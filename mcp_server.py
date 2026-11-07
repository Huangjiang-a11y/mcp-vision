#!/usr/bin/env python3
"""mcp-vision MCP 服务器（stdio 或 --http PORT）。手写 JSON-RPC，无 mcp SDK 依赖。"""
import argparse, base64, hashlib, json, math, os, re, shutil, sys, time, http.server, socketserver, threading
from pathlib import Path
import engine

ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE = ".mvv-cache"


def _load_env_local(path=None):
    """读仓库根 env.local（KEY=VALUE），只补「还没设」的变量。

    MCP 客户端启动 python 时拿不到 shell 里 export 的东西，
    所以模型路径这类配置落文件，进程自己补进 os.environ。
    """
    f = Path(path) if path else (Path(__file__).resolve().parent / "env.local")
    try:
        lines = f.read_text(encoding="utf-8").splitlines()
    except Exception:
        return {}
    applied = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and not os.environ.get(k):
            os.environ[k] = v
            applied[k] = v
    return applied


_load_env_local()
KEEP_WORKSPACES = int(os.environ.get("MVV_KEEP_WORKSPACES") or "5")


def _prune_workspaces(root, keep=None):
    """只留最近打开的 N 个工作区，多出来的整目录删掉。返回删掉的名字。"""
    keep = KEEP_WORKSPACES if keep is None else int(keep)
    root = Path(root)
    if keep <= 0 or not root.exists():
        return []
    def _mt(d):
        m = d / "meta.json"
        try:
            return m.stat().st_mtime
        except Exception:
            return d.stat().st_mtime
    dirs = sorted((d for d in root.iterdir() if d.is_dir()), key=_mt, reverse=True)
    victims = dirs[keep:]
    gone = []
    for d in victims:
        try:
            shutil.rmtree(d)
            gone.append(d.name)
        except Exception:
            pass
    return gone


def _mask(s):
    return s


def _dump(obj, limit=0):
    """序列化。默认**不裁剪**：调用方是模型，宁可长也别悄悄少一半。

    limit>0 时仍保留旧的逐半裁剪逻辑，应急用。
    """
    txt = json.dumps(obj, ensure_ascii=False, indent=2)
    if limit <= 0 or len(txt) <= limit:
        return txt
    import copy
    o = copy.deepcopy(obj)
    for key in ("segments", "results", "timeline", "cues", "frames"):
        v = o.get(key)
        if isinstance(v, list) and v:
            o.setdefault("_truncated", {})[key] = len(v)
    while len(json.dumps(o, ensure_ascii=False, indent=2)) > limit:
        cand = [k for k in o if isinstance(o[k], list) and len(o[k]) > 1]
        if not cand:
            return json.dumps({"_truncated": True, "note": "结果过大"}, ensure_ascii=False, indent=2)
        k = max(cand, key=lambda x: len(json.dumps(o[x], ensure_ascii=False)))
        o[k] = o[k][: max(1, len(o[k]) // 2)]
        o.setdefault("_truncated", {})[k] = "只保留前 %d 条" % len(o[k])
    return json.dumps(o, ensure_ascii=False, indent=2)


MAX_ANALYZE_SECONDS = 900.0   # 单次分析硬上限 15 分钟


def _window(a, asset, default_span=None, allow_end=False):
    """统一的取窗：先定时间戳 time，再定看多久 seconds。

    - time：从哪开始（默认 0）
    - seconds：看多久；不给则看满 default_span 或上限
    - end_seconds：只有 schema 里声明了它的工具才认（audio / subtitles）；video 传了也不生效
    - max_seconds：单次上限，默认 900（15 分钟）
    返回 (start, end, note)，note 非空表示被夹过。
    """
    dur = float(asset.get("duration") or 0)
    start = max(0.0, float(a.get("time") or 0))
    cap = float(a.get("max_seconds") or MAX_ANALYZE_SECONDS)
    if cap <= 0:
        cap = MAX_ANALYZE_SECONDS

    span = None
    if a.get("seconds") is not None:
        span = float(a["seconds"])
    elif allow_end and a.get("end_seconds") is not None:
        span = max(0.0, float(a["end_seconds"]) - start)
    elif default_span:
        span = float(default_span)

    if span is None:
        span = cap
    notes = []
    if span > cap:
        span = cap
        notes.append("单次上限 %.0f 秒（max_seconds 可调）" % cap)
    end = start + span
    if dur > 0 and end > dur:
        end = dur
        notes.append("已到视频结尾 %.0f 秒" % dur)
    if dur > 0 and start >= dur:
        notes.append("time 超出时长，回退到 0")
        start, end = 0.0, min(dur, cap)
    return start, max(start, end), "；".join(notes)


def _load_asset(cwd, vid=None):
    """按新布局加载 asset：workspace/<title>/meta.json。"""
    root = Path(cwd) / "workspace"
    if vid:
        cand = root / vid
        if cand.is_dir() and (cand / "meta.json").exists():
            try:
                return json.loads((cand / "meta.json").read_text())
            except Exception:
                pass
        if root.exists():
            for d in root.iterdir():
                if not d.is_dir():
                    continue
                m = d / "meta.json"
                if m.exists():
                    try:
                        a = json.loads(m.read_text())
                        if a.get("id") == vid:
                            return a
                    except Exception:
                        pass
        return None
    return engine.load_latest(cwd)


def _missing(a):
    v = a.get("video_id")
    return ("找不到视频文件夹: %s（用 workspace.list 看有哪些）" % v) if v else "先 open 一个视频"


def _normalize_source(src):
    # 借自 dsh-video-vision/lib/ytdlp.js normalizeSource：规范化短链接
    try:
        from urllib.parse import urlparse, parse_qs
        u = urlparse(src)
        host = (u.hostname or "").lower()
        if host == "youtu.be":
            vid = u.path.strip("/").split("/")[0]
            if vid and len(vid) == 11:
                return "https://www.youtube.com/watch?v=" + vid
        if host in ("youtube.com", "www.youtube.com", "m.youtube.com"):
            q = parse_qs(u.query)
            if "v" in q and len(q["v"][0]) == 11:
                return "https://www.youtube.com/watch?v=" + q["v"][0]
    except Exception:
        pass
    return src




def _convert_cookies_txt(cwd):
    # cookies.txt: 单行 name=value; ... 转 Netscape 格式，写到 .cookies_netscape.txt
    root = Path(cwd)
    fpath = root / "cookies.txt"
    if not fpath.exists():
        return None
    try:
        text = fpath.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    if not text:
        return None
    if text.startswith("# Netscape") or text.startswith("# HTTP Cookie"):
        return str(fpath)
    domain = ".bilibili.com"
    lines = ["# Netscape HTTP Cookie File", "# https://curl.haxx.se/rfc/cookie_spec.html", ""]
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        eq = part.find("=")
        if eq <= 0:
            continue
        name = part[:eq].strip()
        value = part[eq + 1:].strip()
        if not name:
            continue
        lines.append(domain + "\tTRUE\t/\tFALSE\t0\t" + name + "\t" + value)
    out = root / ".cookies_netscape.txt"
    try:
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception:
        return None
    return str(out)


def _QUALITY_MAP(q):
    return {"low": 480, "medium": 720, "high": 1080, "max": None}.get((q or "medium").lower(), 720)


def tool_open(a):
    src = (a.get("source") or "").strip()
    if not src:
        return "要给 source（视频路径或链接）", []
    # 本地文件用 file:// 协议
    if src.lower().startswith("file://"):
        src = src[7:]
        if not src.startswith("/"):
            src = "/" + src
    src = _normalize_source(src)
    cwd = Path(__file__).resolve().parent
    # 本地文件
    pth = Path(src)
    if not pth.is_absolute():
        pth = (Path(cwd) / pth).resolve()
    if pth.exists():
        asset = engine.open_local(cwd, str(pth))
        return ("已打开 %s\n时长 %.2fs  %sx%s  音频=%s" % (
            asset["title"], asset["duration"], asset["width"], asset["height"],
            "有" if asset["has_audio"] else "无"), [])
    # 链接：用 yt-dlp 下载
    try:
        import yt_dlp
    except ImportError:
        return "需要 yt-dlp（pip install yt-dlp）才能下载链接", []
    out_dir = Path(cwd) / DEFAULT_CACHE / "downloads"
    out_dir.mkdir(parents=True, exist_ok=True)
    max_h = _QUALITY_MAP(a.get("quality"))
    # yt-dlp 的 format 表达式里没有 min()，竖屏也不能只看 height（那是长边）。
    # 所以不写死过滤条件，改成 bv*+ba 全拿，再用 format_sort 把分辨率往上限上靠。
    # 编码优先 h264：同分辨率下 CPU 软解最快（720p 实测 0.12 秒/帧，hevc 0.20，av1 0.27）。
    fmt = "bv*+ba/b"
    _sort = ["vcodec:h264"]
    if max_h:
        _sort.insert(0, "res:%d" % max_h)
    ydl_opts = {
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "format": fmt,
        "format_sort": _sort,
        "merge_output_format": "mp4",
        "quiet": True, "no_warnings": True, "noplaylist": True,
        "retries": 3, "fragment_retries": 3,
    }
    # cookies: 真值表；cookies.txt 放在仓库根目录
    if a.get("cookies"):
        converted = _convert_cookies_txt(cwd)
        if converted:
            ydl_opts["cookiefile"] = converted
        else:
            # 若转换失败或未找到，不强制报错，让 yt-dlp 自己决定
            pass
    # 规范化链接
    src = _normalize_source(src)
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(src, download=True)
            fname = ydl.prepare_filename(info)
            if not Path(fname).exists():
                for cand in out_dir.glob("%s.*" % info.get("id")):
                    if cand.suffix in (".mp4", ".mkv", ".webm"):
                        fname = str(cand); break
    except Exception as e:
        return ("下载失败: %s" % str(e)[:300], [])
    asset = engine.open_local(cwd, fname, title=info.get("title"), url=src)
    return ("已下载并打开 %s\n时长 %.2fs  %sx%s" % (
        asset["title"], asset["duration"], asset["width"], asset["height"]), [])


def tool_audio(a):
    cwd = str(Path(__file__).resolve().parent)
    asset = _load_asset(cwd, a.get("video_id"))
    if asset is None:
        return _missing(a), []
    start, end, note = _window(a, asset, default_span=30.0, allow_end=True)
    res = engine.analyze_audio(asset["media_path"], start, end, out_dir=asset.get("audio_dir"))
    if note:
        res = dict(res, note=note)
    return (_dump(res), [])


def tool_ocr(a):
    """单图 OCR。没有 image 就直说，不再偷偷去 OCR 最近打开的视频。"""
    cwd = str(Path(__file__).resolve().parent)
    try:
        import ocr as ocrmod
    except Exception as e:
        return ("OCR 不可用: %s（先 pkg install python-onnxruntime，并把 PP-OCRv4 放进 models/，见 INSTALL.md）" % e, [])
    src = a.get("image")
    if not src:
        return ("ocr 需要 image（图片路径）。要读视频里的字幕/文字，用 action=band，"
                "或者先 video action=frame 取一帧再喂进来", [])
    p = Path(str(src))
    if not p.is_absolute():
        p = (Path(cwd) / p).resolve()
    if not p.exists():
        return ("找不到图片: %s" % p, [])
    try:
        rgb = ocrmod.load_rgb(p)
        o = ocrmod.get_ppocr()
        res = o.ocr_image(rgb)
    except Exception as e:
        return ("OCR 失败: %s" % e, [])
    return (json.dumps({"image": str(p), "providers": o.providers,
                        "count": len(res), "results": res}, ensure_ascii=False), [])


def _to_srt(segs):
    """字幕段 → SRT。时间轴只有秒级精度，毫秒位一律补 000，不伪造精度。"""
    def to_sec(v):
        # 段里的时间既可能是秒数，也可能是 "00:06.000" / "1:02:03.000"
        if isinstance(v, (int, float)):
            return float(v)
        parts = [float(x) for x in str(v).replace(",", ".").split(":")]
        sec = 0.0
        for p in parts:
            sec = sec * 60 + p
        return sec

    def tc(sec):
        t = int(round(max(0.0, to_sec(sec))))
        return "%02d:%02d:%02d,000" % (t // 3600, t // 60 % 60, t % 60)
    lines = []
    for i, g in enumerate(segs, 1):
        lines.append("%d" % i)
        lines.append("%s --> %s" % (tc(g["from"]), tc(g["to"])))
        lines.append(g["text"])
        lines.append("")
    return "\n".join(lines)


def _parse_band(v):
    """band 可以是 auto / [y0,y1] / "0.10,0.27" / "[0.10, 0.27]"。

    schema 里这个参数没锁类型，客户端送过来的多半是字符串，
    原来直接 band[0] 取下标，传 "0.10,0.27" 就成了 float("[") 报错。
    """
    if v is None or str(v).strip() in ("", "auto"):
        return "auto"
    if isinstance(v, (list, tuple)):
        if len(v) != 2:
            raise ValueError("band 需要两个数：[y0, y1]")
        y0, y1 = float(v[0]), float(v[1])
    else:
        txt = str(v).strip().strip("[](){}")
        parts = [p for p in re.split(r"[,;\s]+", txt) if p]
        if len(parts) != 2:
            raise ValueError("band 要写成 auto 或 \"0.10,0.27\"（纵向起止比例）")
        y0, y1 = float(parts[0]), float(parts[1])
    if not (0.0 <= y0 < y1 <= 1.0):
        raise ValueError("band 需满足 0 ≤ y0 < y1 ≤ 1")
    return (y0, y1)


def tool_subtitles(a):
    """字幕带 OCR：裁下 30% 画面 + 变帧跳过，输出折叠后的字幕段。"""
    cwd = str(Path(__file__).resolve().parent)
    try:
        import ocr as ocrmod
    except Exception as e:
        return ("OCR 不可用: %s" % e, [])
    asset = _load_asset(cwd, a.get("video_id"))
    if asset is None:
        return (_missing(a), [])
    media = asset["media_path"]
    start, end, window_note = _window(a, asset, allow_end=True)
    try:
        band = _parse_band(a.get("band", "auto"))
    except ValueError as e:
        return (str(e), [])
    try:
        res = ocrmod.ocr_subtitles(
            media, start, end,
            step=float(a.get("step", 1.0)),
            band=band,
            width=int(a["width"]) if a.get("width") else None,
            skip_th=float(a.get("skip_th", 1.0)),
            text_th=float(a.get("text_th", 223)),
            engine=ocrmod.get_engine())
    except Exception as e:
        return ("字幕提取失败: %s" % e, [])
    segs = ocrmod.fold_segments(res["results"], step=float(a.get("step", 1.0)))
    out = {"frames": res["frames"], "ocr_runs": res["ocr_runs"], "skipped": res["skipped"], "band": res.get("band"), "band_auto": res.get("band_auto"),
           "band_px": res["band_px"], "segment_count": len(segs), "segments": segs}
    if not segs and band == "auto":
        # 自动检测只搜下方 40%；一个字的没有时，多半是字幕根本不在下面，别让人以为这片子没字幕
        out["hint"] = ("自动检测在下方 40%% 没找到字幕（band=%s）。"
                       "字幕可能在上方或画面中部：先 video action=frame 看一眼位置，"
                       "再用 band=\"y0,y1\" 手动指定，例如 \"0.10,0.27\"。" % (res.get("band"),))
    if window_note:
        out["note"] = window_note
    fmt = str(a.get("format") or "json").lower()
    if fmt == "srt":
        srt = _to_srt(segs)
        dest = Path(a.get("save_to") or (Path(asset["video_dir"]).parent / "out" / "subtitles.srt"))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(srt, encoding="utf-8")
        out["srt_path"] = str(dest)
        out["srt"] = srt
    return (_dump(out), [])


def tool_list(a):
    cwd = str(Path(__file__).resolve().parent)
    st = engine.status(cwd)
    return (json.dumps(st, ensure_ascii=False, indent=2), [])




TOOLS = [
    {"name": "workspace",
     "description": "Workspace: open or download a video from a URL or local path; list opened videos.",
     "inputSchema": {"type": "object", "properties": {
         "action": {"type": "string", "enum": ["open", "list"],
                    "description": "open = download and open (default); list = show opened videos"},
         "source": {"type": "string",
                    "description": "Video path or URL; use the file:// scheme for local files"},
         "quality": {"type": "string", "enum": ["low", "medium", "high", "max"],
                     "description": "low=480p medium=720p high=1080p max=source, filtered by the short edge"},
         "cookies": {"type": "boolean", "description": "true = use cookies.txt from the repo root"}}}},

    {"name": "video",
     "description": (
         "Look at the picture: either the whole clip on one page, or one exact frame.\n"
         "\n"
         "analyze - the whole clip at a glance.\n"
         "Give a window and a frame count; it samples that many frames evenly from the window start to "
         "its end (both ends included), lays them out left to right, top to bottom, and returns the page "
         "plus a timeline list.\n"
         "Example: quantity=12, seconds=60 on a 1280x720 clip gives a 1185x888 png laid out 3 columns x 4 rows "
         "- 12 frames, every cell filled, each cell 395x222 - and timeline = [{t:0.0,...}, {t:5.455,...}, "
         "... {t:60.0,...}]. You get exactly `quantity` frames; the grid only decides the layout: it prefers a "
         "shape close to square and, when the count splits cleanly, no empty cells (a prime count like 13 falls "
         "back to a grid with a couple of dark cells). The sheet has no timestamps burned in: cell order matches "
         "timeline order 1:1, which is how you tell which moment a thumbnail is. The page is always sent as an "
         "image, with sheetJpg / sheetKB describing the file.\n"
         "\n"
         "frame - one moment, or a short burst after it.\n"
         "Without `seconds` you get the single frame at `time`. With `seconds` (and optionally `quantity`, "
         "default 1) it becomes a burst: `quantity` frames evenly spaced from `time` to `time`+`seconds`, each "
         "cropped the same way, returned as separate images plus items[{t,path}].\n"
         "Example: time=123.5, crop=1280x260+0+440 returns that slice as an image plus its saved path; "
         "time=42, seconds=2, quantity=4 returns frames at 42.0 / 42.667 / 43.333 / 44.0. Use frame when the "
         "page is too small to read a board, a scoreboard or a subtitle line.\n"
         "\n"
         "Shared: quality is an area cap, not a resolution - medium means about one megapixel total. For "
         "analyze it caps the whole page (cells get smaller as the grid grows); for frame it caps the "
         "cropped area. It never upscales. The window is time + seconds, capped at max_seconds "
         "(900) and clamped to the clip, with a note in the reply when that happens. Outputs land in "
         "workspace/<video>/video/ (frames t<ms>-<w>x<h>.jpg, page sheet_*.png, single frame_<ms>-<hash>.jpg) "
         "and repeat calls reuse them."
     ),
     "inputSchema": {"type": "object", "properties": {
         "action": {"type": "string", "enum": ["analyze", "frame"],
                    "description": "analyze = grid page for the clip (default); frame = one frame"},
         "video_id": {"type": "string", "description": "Folder name under workspace/; default: most recently opened"},
         "time": {"type": "number",
                  "description": "frame: the moment, or the start of a burst; analyze: window start, default 0"},
         "quantity": {"type": "integer",
                      "description": "analyze: sampled frames, default 15, max 30; frame: burst size, default 1"},
         "seconds": {"type": "number",
                     "description": "analyze: window length; frame: burst length starting at time (omit for a single frame)"},
         "max_seconds": {"type": "number", "description": "window cap in seconds, default 900"},
         "crop": {"type": "string", "description": "frame: crop as WxH+X+Y, e.g. 720x300+0+880"},
         "quality": {"type": "string", "enum": ["low", "medium", "high", "max"],
                     "description": "pixel cap: low=512^2 medium=1024^2 high=1536^2 max=2048^2; scales down only"}}}},

    {"name": "audio",
     "description": (
         "Listen to the audio. levels: loudness profile. "
         "transcribe: speech to text (silero-vad + SenseVoice)."
     ),
     "inputSchema": {"type": "object", "properties": {
         "action": {"type": "string", "enum": ["levels", "transcribe"],
                    "description": "levels = loudness profile (default); transcribe = speech to text"},
         "video_id": {"type": "string", "description": "Folder name under workspace/; default: most recently opened"},
         "time": {"type": "number", "description": "window start in seconds, default 0"},
         "seconds": {"type": "number", "description": "window length in seconds; takes precedence over end_seconds"},
         "end_seconds": {"type": "number", "description": "window end in seconds (legacy; prefer seconds)"},
         "max_seconds": {"type": "number", "description": "window cap in seconds, default 900"},
         "language": {"type": "string", "description": "transcribe: zh / en / ja / ko, default zh"},
         "max_speech": {"type": "number", "description": "transcribe: longest segment in seconds, default 20"}}}},

    {"name": "subtitles",
     "description": (
         "Read on-screen text. ocr reads one image; band OCRs a window frame by frame and folds the "
         "result into cues (from/to/text/conf). Pass band=\"0.10,0.27\" to pin the band manually."
     ),
     "inputSchema": {"type": "object", "properties": {
         "action": {"type": "string", "enum": ["ocr", "band"],
                    "description": "ocr = one image (default); band = subtitle cues for a window"},
         "video_id": {"type": "string", "description": "Folder name under workspace/; default: most recently opened"},
         "image": {"type": "string", "description": "ocr: path to the image"},
         "time": {"type": "number", "description": "window start in seconds, default 0"},
         "seconds": {"type": "number", "description": "band: window length in seconds; takes precedence over end_seconds"},
         "end_seconds": {"type": "number", "description": "band: window end in seconds (legacy; prefer seconds)"},
         "max_seconds": {"type": "number", "description": "window cap in seconds, default 900"},
         "step": {"type": "number", "description": "band: sampling interval in seconds, default 1.0"},
         "band": {"type": "string",
                  "description": "band range: 'auto' (default) or ratios like \"0.10,0.27\""},
         "skip_th": {"type": "number",
                     "description": "skip threshold as percent change of the in-band text mask, default 1.0; -1 disables"},
         "text_th": {"type": "number", "description": "highlight threshold on grayscale, default 223"},
         "format": {"type": "string", "enum": ["json", "srt"], "description": "srt = also write a subtitle file"},
         "save_to": {"type": "string", "description": "srt output path, default workspace/<video>/out/subtitles.srt"},
         "width": {"type": "integer", "description": "resize the band before OCR; default: source resolution"}}}},

]


VIDEO_QUALITY_PX = {"low": 512, "medium": 1024, "high": 1536, "max": 2048}


def _max_pixels(a):
    n = VIDEO_QUALITY_PX.get(str(a.get("quality") or "medium").lower(), 1024)
    return float(n) * n


def _fit(src_w, src_h, max_px):
    # 只缩不放；按面积上限等比缩，宽高取偶
    px = float(src_w) * float(src_h)
    if px <= 0 or px <= max_px:
        return int(src_w), int(src_h)
    k = math.sqrt(max_px / px)
    w = max(2, int(round(src_w * k)) // 2 * 2)
    h = max(2, int(round(src_h * k)) // 2 * 2)
    return w, h


def _parse_crop(spec):
    # WxH+X+Y -> ffmpeg crop=W:H:X:Y
    if not spec:
        return None
    m = re.match(r"^(\d+)x(\d+)\+(\d+)\+(\d+)$", str(spec).strip())
    if not m:
        raise ValueError("crop 格式应为 WxH+X+Y")
    w, h, x, y = (int(v) for v in m.groups())
    if w <= 0 or h <= 0:
        raise ValueError("crop 宽高须为正")
    return "crop=%d:%d:%d:%d" % (w, h, x, y)


def tool_analyze(a):
    cwd = str(Path(__file__).resolve().parent)
    asset = _load_asset(cwd, a.get("video_id"))
    if asset is None:
        return (_missing(a), [])
    duration = float(asset.get("duration") or 0)
    if duration <= 0:
        return ("视频时长为 0，无法抽帧", [])
    vw = float(asset.get("width") or 16)
    vh = float(asset.get("height") or 9)
    aspect = vw / vh
    fw, fh = _fit(vw, vh, _max_pixels(a))
    qty = int(a.get("quantity") or 15)
    n0 = max(1, min(qty, 30))
    # 先定网格，让整张图接近正方形；再按 cols*rows 抽帧，避免空格
    best = None
    for c in range(1, n0 + 1):
        base_r = (n0 + c - 1) // c
        for r in (base_r, base_r + 1):
            empty = c * r - n0
            if empty > 3:
                continue
            flip = (c < r) if aspect >= 1 else (c > r)
            score = abs(math.log((c / r) * aspect)) + 0.15 * empty + (0.25 if flip else 0)
            key = (round(score, 3), empty)
            if best is None or key < best[0]:
                best = (key, c, r)
    cols, rows = best[1], best[2]
    # 要几张就抽几张：网格只决定排版，不足的最后一排留空，不再靠多抽样把格子填满
    need = min(n0, 30)
    win_start, win_end, window_note = _window(a, asset)
    ts = engine.sample_times(win_start, win_end, need)
    frames = engine.extract_frames(asset["media_path"], ts, width=fw, size=(fw, fh),
                                   out_dir=asset.get("video_dir"), duration=duration)
    sheet_path = None
    if frames:
        # 格子尺寸由 quality 档位（像素上限）反推：整页 ≈ cols*rows 格 ≈ 上限像素数
        # 短边 s 满足 s*s*aspect*cols*rows = cap，即 s = sqrt(cap/(aspect*cols*rows))
        cap = _max_pixels(a)
        short = int(round(math.sqrt(cap / max(1e-6, aspect * cols * rows))))
        short = max(80, min(short, 640))
        if aspect < 1:
            cell_w, cell_h = short, max(1, int(round(short / aspect)))
        else:
            cell_h, cell_w = short, max(1, int(round(short * aspect)))
        sp = Path(asset.get("video_dir") or ".")
        sp.mkdir(parents=True, exist_ok=True)
        # 文件名带上片源和网格参数：同名会互相覆盖，多 P 同目录时尤其危险
        key = "%s|%s|%s|%s|%dx%d|%dx%d" % (asset.get("id") or "", a.get("quantity"),
                                           a.get("quality"), window_note, cols, rows, cell_w, cell_h)
        tag = hashlib.md5(key.encode()).hexdigest()[:6]
        tc = engine.timecode(duration).replace(":", "").replace(".", "")
        sheet_path = str(sp / ("sheet_%s_%dx%d_%s.png" % (tc, cols, rows, tag)))
        try:
            engine.compose_sheet(asset["media_path"], [f["seconds"] for f in frames], cols, rows,
                                 cell_w=cell_w, cell_h=cell_h, out_path=sheet_path,
                                 title=asset.get("title"),
                                 frame_paths=[f["path"] for f in frames])
        except Exception:
            sheet_path = None
    images = []
    out = {
        "ok": True,
        "title": asset.get("title"),
        "duration": round(duration, 2),
        "width": asset.get("width"),
        "height": asset.get("height"),
        "frameSize": [fw, fh],
        "frames": len(frames),
        "timeline": [{"t": f["seconds"], "path": f["path"]} for f in frames],
        "sheet": sheet_path,
        "sheetJpg": None,
        "attached": False,
        "videoDir": asset.get("video_dir"),
        "nextStep": "描述画面前先看 sheet 或 timeline 里的图；细节用 action=frame 取原帧。",
    }
    if sheet_path:
        jp = Path(sheet_path).with_suffix(".jpg")
        try:
            if (not jp.exists()) or jp.stat().st_mtime < Path(sheet_path).stat().st_mtime:
                engine._run(["ffmpeg", "-y", "-v", "error", "-i", sheet_path,
                             "-q:v", "4", str(jp)], timeout=120)
            if jp.exists():
                out["sheetJpg"] = str(jp)
                out["attached"] = True
                out["sheetKB"] = int(round(jp.stat().st_size / 1024))
                images.append({"type": "image",
                               "data": base64.b64encode(jp.read_bytes()).decode(),
                               "mimeType": "image/jpeg"})
        except Exception:
            out["attached"] = False
    return (json.dumps(out, ensure_ascii=False, indent=2), images)


def tool_frame(a):
    """单帧；给了 seconds 就是「从 time 起这段范围」里的连拍。"""
    cwd = str(Path(__file__).resolve().parent)
    asset = _load_asset(cwd, a.get("video_id"))
    if asset is None:
        return (_missing(a), [])
    t = a.get("time")
    if t is None:
        return ("frame 需要 time（秒）", [])
    t = float(t)
    spec = str(a.get("crop") or "").strip()
    try:
        crop = _parse_crop(spec)
    except ValueError as e:
        return (str(e), [])
    if spec:
        m = re.match(r"^(\d+)x(\d+)\+(\d+)\+(\d+)$", spec)
        sw, sh = float(m.group(1)), float(m.group(2))
    else:
        sw = float(asset.get("width") or 16)
        sh = float(asset.get("height") or 9)
    tw, th = _fit(sw, sh, _max_pixels(a))
    vf = ([crop] if crop else []) + ["scale=%d:%d" % (tw, th)]

    # 窗口：不给 seconds 就是 time 那一帧；给了就是 time → time+seconds 里等距取 quantity 张
    span = a.get("seconds")
    n = max(1, min(int(a.get("quantity") or 1), 30))
    if span:
        cap = float(a.get("max_seconds") or MAX_ANALYZE_SECONDS)
        span = min(float(span), cap if cap > 0 else MAX_ANALYZE_SECONDS)
        ts = engine.sample_times(t, t + span, n)
    else:
        ts = [t]

    out_dir = Path(asset.get("video_dir") or ".")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = hashlib.md5(",".join(vf).encode()).hexdigest()[:6]
    items, images = [], []
    for tt in ts:
        dest = out_dir / ("frame_%s-%s.jpg" % (engine.timecode(tt).replace(":", ""), tag))
        if not dest.exists():
            r = engine._run(["ffmpeg", "-y", "-v", "error", "-ss", "%.3f" % tt, "-i", asset["media_path"],
                             "-frames:v", "1", "-vf", ",".join(vf), str(dest)], timeout=120)
            if not dest.exists():
                return ("取帧失败 @%.3fs: %s" % (tt, r or "ffmpeg 无输出"), [])
        items.append({"t": round(tt, 3), "path": str(dest)})
        images.append({"type": "image",
                       "data": base64.b64encode(dest.read_bytes()).decode(),
                       "mimeType": "image/jpeg"})

    if len(items) == 1:
        return (json.dumps({"ok": True, "t": items[0]["t"], "save": items[0]["path"],
                            "size": [tw, th], "quality": a.get("quality") or "medium"},
                           ensure_ascii=False), images[:1])
    out = {"ok": True, "frames": len(items), "range": [round(ts[0], 3), round(ts[-1], 3)],
           "size": [tw, th], "quality": a.get("quality") or "medium",
           "crop": spec or None, "items": items}
    return (json.dumps(out, ensure_ascii=False, indent=2), images)


def dispatch_workspace(a):
    act = (a.get("action") or "open").lower()
    if act == "open":
        return tool_open(a)
    if act == "list":
        return tool_list(a)
    return ("unknown action: %s" % act, [])


def dispatch_video(a):
    act = (a.get("action") or "analyze").lower()
    if act == "analyze":
        return tool_analyze(a)
    if act == "frame":
        return tool_frame(a)
    return ("unknown action: %s" % act, [])


def tool_transcribe(a):
    cwd = str(Path(__file__).resolve().parent)
    asset = _load_asset(cwd, a.get("video_id"))
    if asset is None:
        return (_missing(a), [])
    dur_total = float(asset.get("duration") or 0)
    if dur_total > 0 and float(a.get("time") or 0) >= dur_total:
        return (json.dumps({"ok": False,
                            "error": "time=%.1f 超出时长 %.1f" % (float(a["time"]), dur_total)},
                           ensure_ascii=False), [])
    start, end, window_note = _window(a, asset, allow_end=True)
    language = str(a.get("language") or "zh")
    max_speech = float(a.get("max_speech") or 20)
    t0 = time.time()
    try:
        segs, dur = engine.transcribe(asset["media_path"], start, end,
                                      language=language, max_speech=max_speech)
    except RuntimeError as e:
        return (json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False, indent=2), [])
    out = {
        "ok": True,
        "engine": "silero-vad + sensevoice",
        "range": [round(start, 2), round(start + dur, 2)],
        "duration": round(dur, 2),
        "segments": len(segs),
        "cues": segs,
        "text": "".join(x["text"] for x in segs),
        "elapsed": round(time.time() - t0, 2),
    }
    if window_note:
        out["note"] = window_note
    return (json.dumps(out, ensure_ascii=False, indent=2), [])


def dispatch_audio(a):
    act = (a.get("action") or "levels").lower()
    if act == "levels":
        return tool_audio(a)
    if act == "transcribe":
        return tool_transcribe(a)
    return ("unknown action: %s" % act, [])


def dispatch_subtitles(a):
    act = (a.get("action") or "band").lower()
    if act == "ocr":
        return tool_ocr(a)
    if act == "band":
        return tool_subtitles(a)
    return ("unknown action: %s" % act, [])


HANDLERS = {
    "workspace": dispatch_workspace,
    "video": dispatch_video,
    "audio": dispatch_audio,
    "subtitles": dispatch_subtitles,
}


# 同一个工具在没跑完之前不重入；不同工具之间互不阻塞。
# 手机内存有限：video 和 audio 同时跑 ffmpeg + 模型，两路就够，别放成四路。
_TOOL_LOCKS = {}
_TOOL_LOCKS_GUARD = threading.Lock()


def _tool_lock(name):
    with _TOOL_LOCKS_GUARD:
        lk = _TOOL_LOCKS.get(name)
        if lk is None:
            lk = _TOOL_LOCKS[name] = threading.Lock()
        return lk


def handle(req):
    if not isinstance(req, dict):
        return {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "invalid request"}}
    rid = req.get("id")
    method = req.get("method")
    params = req.get("params") or {}
    is_note = "id" not in req
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": rid,
                "result": {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
                           "serverInfo": {"name": "mcp-vision", "version": "0.3.0"}}}
    if method == "notifications/initialized":
        return None if is_note else {"jsonrpc": "2.0", "id": rid, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}}
    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        handler = HANDLERS.get(name)
        if not handler:
            return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "unknown tool %s" % name}}
        try:
            with _tool_lock(name):
                text, images = handler(args)
        except Exception as e:
            return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32000, "message": str(e)}}
        content = [{"type": "text", "text": text}]
        for im in (images or []):
            content.append(im)
        return {"jsonrpc": "2.0", "id": rid, "result": {"content": content}}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": rid, "result": {}}
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "method not found: %s" % method}}


def serve_stdio():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            continue
        resp = handle(req)
        if resp is None:
            continue
        sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
        sys.stdout.flush()


def serve_http(port, host="127.0.0.1"):
    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(n)
            try:
                req = json.loads(body)
            except Exception:
                self.send_response(400); self.end_headers(); return
            try:
                resp = handle(req)
            except Exception as e:      # 别让一个请求炸掉整条连接线程
                resp = {"jsonrpc": "2.0", "id": None,
                        "error": {"code": -32603, "message": "internal: %s" % e}}
            if resp is None:
                self.send_response(204); self.end_headers(); return
            data = json.dumps(resp, ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        def log_message(self, *a):
            pass
    class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True               # 主线程退出时不等长请求
        allow_reuse_address = True          # 重启不用等端口释放

    httpd = Server((host, port), H)
    httpd.serve_forever()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--http", type=int, default=0)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    try:
        if args.http:
            serve_http(args.http, args.host)
        else:
            serve_stdio()
    except KeyboardInterrupt:
        # Ctrl+C 正常退出，不打 traceback
        sys.stderr.write("\n[mcp-vision] 已停止\n")


if __name__ == "__main__":
    main()
