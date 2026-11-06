#!/usr/bin/env python3
"""mcp-vision MCP 服务器（stdio 或 --http PORT）。手写 JSON-RPC，无 mcp SDK 依赖。"""
import argparse, base64, hashlib, json, math, os, re, sys, time, http.server, socketserver, threading
from pathlib import Path
import engine

ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE = ".mvv-cache"


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


def _window(a, asset, default_span=None):
    """统一的取窗：先定时间戳 start_seconds，再定看多久 seconds。

    - start_seconds：从哪开始（默认 0）
    - seconds / analyze_seconds：看多久；不给则看满 default_span 或上限
    - end_seconds：仍兼容，等价于 sec = end - start
    - max_seconds：单次上限，默认 900（15 分钟）
    返回 (start, end, note)，note 非空表示被夹过。
    """
    dur = float(asset.get("duration") or 0)
    start = max(0.0, float(a.get("start_seconds") or 0))
    cap = float(a.get("max_seconds") or MAX_ANALYZE_SECONDS)
    if cap <= 0:
        cap = MAX_ANALYZE_SECONDS

    span = None
    if a.get("seconds") is not None:
        span = float(a["seconds"])
    elif a.get("analyze_seconds") is not None:
        span = float(a["analyze_seconds"])
    elif a.get("end_seconds") is not None:
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
        notes.append("start_seconds 超出时长，回退到 0")
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
    src = (a.get("source") or a.get("file") or "").strip()
    if not src:
        return "要给 source（视频路径或链接）", []
    # 本地文件用 file:// 协议
    if src.lower().startswith("file://"):
        src = src[7:]
        if not src.startswith("/"):
            src = "/" + src
    src = _normalize_source(src)
    cwd = Path(a.get("cwd") or (Path(__file__).resolve().parent))
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
    fmt = "bv*+ba/b"
    ydl_opts = {
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "format": fmt,
        "format_sort": ["res:%d" % max_h] if max_h else None,
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

def tool_frames(a):
    cwd = str(Path(__file__).resolve().parent)
    asset = _load_asset(cwd, a.get("video_id"))
    if asset is None:
        return _missing(a), []
    media = asset["media_path"]
    width = int(a.get("width") or 1280)
    frames = []
    if a.get("timestamps"):
        ts = [float(x) for x in a["timestamps"]]
        frames = engine.extract_frames(media, ts, width=width, out_dir=asset.get("video_dir"), duration=asset["duration"])
    else:
        start = float(a.get("start_seconds", asset.get("cursor", 0)))
        end = float(a.get("end_seconds", min(start + 30, asset["duration"])))
        count = int(a.get("frames", 6))
        count = max(1, min(engine.MAX_FRAMES, count))
        ts = engine.sample_times(start, end, count)
        frames = engine.extract_frames(media, ts, width=width, out_dir=asset.get("video_dir"), duration=asset["duration"])
    # update cursor
    if frames and not a.get("timestamps"):
        asset["cursor"] = frames[-1]["seconds"]
        meta = Path(asset["workspace"]) / "meta.json"
        try:
            meta.write_text(json.dumps(asset, ensure_ascii=False, indent=2))
        except Exception:
            pass
    # build content blocks
    blocks = []
    head = "frames=%d  range=%s" % (len(frames), ",".join("%.1f" % f["seconds"] for f in frames))
    for f in frames:
        data = base64.b64encode(Path(f["path"]).read_bytes()).decode()
        blocks.append({"type": "image", "data": data, "mimeType": "image/jpeg"})
    return (head, blocks)


def tool_sheet(a):
    cwd = str(Path(__file__).resolve().parent)
    asset = _load_asset(cwd, a.get("video_id"))
    if asset is None:
        return _missing(a), []
    media = asset["media_path"]
    cols = int(a.get("cols", 3))
    rows = int(a.get("rows", 3))
    n = cols * rows
    dur = asset["duration"]
    ts = engine.sample_times(0, dur, n)
    out = Path(asset.get("video_dir", cwd)) / "sheet.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cell_w = 400
    cell_h = 225
    engine.compose_sheet(media, ts, cols, rows, cell_w, cell_h, out)
    data = base64.b64encode(out.read_bytes()).decode()
    return ("sheet: %s  (%dx%d grid, %d frames)" % (out, cols, rows, len(ts)),
            [{"type": "image", "data": data, "mimeType": "image/png"}])


def tool_audio(a):
    cwd = str(Path(__file__).resolve().parent)
    asset = _load_asset(cwd, a.get("video_id"))
    if asset is None:
        return _missing(a), []
    start, end, note = _window(a, asset, default_span=30.0)
    res = engine.analyze_audio(asset["media_path"], start, end, out_dir=asset.get("audio_dir"))
    if note:
        res = dict(res, note=note)
    return (_dump(res), [])


def tool_ocr(a):
    cwd = str(Path(__file__).resolve().parent)
    try:
        import ocr as ocrmod
    except Exception as e:
        return ("OCR 不可用: %s（先 pkg install python-onnxruntime，并把 PP-OCRv4 放进 models/，见 INSTALL.md）" % e, [])
    src = a.get("image") or a.get("path")
    if src:
        p = Path(src)
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
        return (json.dumps({"image": str(p), "providers": o.providers, "count": len(res), "results": res}), [])
    asset = engine.load_latest(cwd)
    if asset is None:
        return ("先 open 一个视频，或给 image 参数", [])
    media = asset["media_path"]
    if a.get("timestamps"):
        ts = [float(x) for x in a["timestamps"]]
    else:
        start = float(a.get("start_seconds", 0))
        end = float(a.get("end_seconds", min(start + 30, asset["duration"])))
        ts = engine.sample_times(start, end, max(1, int(a.get("frames", 5))))
    try:
        o = ocrmod.get_ppocr()
    except Exception as e:
        return ("OCR 不可用: %s" % e, [])
    frames = engine.extract_frames(media, ts, width=int(a.get("width") or 1280), duration=asset["duration"])
    out = []
    for f in frames:
        try:
            res = o.ocr_image(ocrmod.load_rgb(f["path"]))
        except Exception:
            res = []
        out.append({"seconds": f["seconds"], "timestamp": f["timestamp"],
                    "lines": [r["text"] for r in res]})
    return (_dump({"frames": len(out), "providers": o.providers, "results": out}), [])


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
    start, end, window_note = _window(a, asset)
    band = a.get("band", "auto")
    try:
        res = ocrmod.ocr_subtitles(
            media, start, end,
            step=float(a.get("step", 1.0)),
            band=band if band == "auto" else (float(band[0]), float(band[1])),
            width=int(a.get("width") or 960),
            skip_th=float(a.get("skip_th", 1.0)),
            text_th=float(a.get("text_th", 220)),
            engine=ocrmod.get_engine())
    except Exception as e:
        return ("字幕提取失败: %s" % e, [])
    segs = ocrmod.fold_segments(res["results"], step=float(a.get("step", 1.0)))
    out = {"frames": res["frames"], "ocr_runs": res["ocr_runs"], "skipped": res["skipped"], "band": res.get("band"), "band_auto": res.get("band_auto"),
           "band_px": res["band_px"], "segment_count": len(segs), "segments": segs}
    if window_note:
        out["note"] = window_note
    fmt = str(a.get("format") or "json").lower()
    if fmt in ("srt", "subrip") or a.get("save_srt"):
        srt = _to_srt(segs)
        dest = Path(a.get("save_to") or (Path(asset["video_dir"]).parent / "out" / "subtitles.srt"))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(srt, encoding="utf-8")
        out["srt_path"] = str(dest)
        out["srt"] = srt if fmt in ("srt", "subrip") else None
    return (_dump(out), [])


def tool_list(a):
    cwd = str(Path(__file__).resolve().parent)
    st = engine.status(cwd)
    return (json.dumps(st, ensure_ascii=False, indent=2), [])


# 兼容别名：status/clean 仍可调用，但建议直接用文件系统
def tool_status(a):
    return tool_list(a)


def tool_clean(a):
    cwd = str(Path(__file__).resolve().parent)
    root = Path(cwd) / "workspace"
    if root.exists():
        shutil.rmtree(root)
        return ("已清空 %s" % root, [])
    return ("无 workspace", [])


TOOLS = [
    {"name": "workspace", "description": "工作区：打开/下载视频、列出已打开视频。纯 Python，无 Node。",
     "inputSchema": {"type": "object", "properties": {
         "action": {"type": "string", "enum": ["open", "list"]},
         "source": {"type": "string", "description": "视频路径或链接；本地用 file:// 协议"},
         "quality": {"type": "string", "enum": ["low", "medium", "high", "max"], "description": "low=480p medium=720p high=1080p max=源最大；按短边筛选，竖屏不误杀"},
         "cookies": {"type": "boolean", "description": "true 时使用仓库根目录的 cookies.txt"}}}},
    {"name": "video", "description": "视频：analyze 整片理解（时间线抽帧 + 近似正方形网格页）；frame 单帧核对。",
     "inputSchema": {"type": "object", "properties": {
         "action": {"type": "string", "enum": ["analyze", "frame"]},
         "video_id": {"type": "string", "description": "工作区里的视频文件夹名（workspace/<名>），不传用最近打开"},
         "time": {"type": "number", "description": "frame 的时间点（秒）"},
         "quantity": {"type": "integer", "description": "analyze 抽帧数，默认 15"},
         "start_seconds": {"type": "number", "description": "从哪个时间戳开始，默认 0"},
         "seconds": {"type": "number", "description": "看多久（秒）；不传则为 15 分钟上限内全长"},
         "max_seconds": {"type": "number", "description": "单次上限秒数，默认 900（15 分钟）"},
         "crop": {"type": "string", "description": "frame 裁剪：WxH+X+Y，如 720x300+0+880"},
         "quality": {"type": "string", "enum": ["low", "medium", "high", "max"],
                     "description": "像素上限（数值平方=最大像素数）：low=512 medium=1024 high=1536 max=2048"}}}},
    {"name": "audio", "description": "音频：levels 电平分析 / extract 导出 / transcribe 转写（silero-vad 自动断句 + SenseVoice）。",
     "inputSchema": {"type": "object", "properties": {
         "action": {"type": "string", "enum": ["levels", "extract", "transcribe"]},
         "video_id": {"type": "string", "description": "工作区里的视频文件夹名，不传用最近打开"},
         "start_seconds": {"type": "number", "description": "起始秒，默认 0"},
         "seconds": {"type": "number", "description": "看多久（秒）；优先于 end_seconds"},
         "end_seconds": {"type": "number", "description": "结束秒（兼容用）"},
         "max_seconds": {"type": "number", "description": "单次上限秒数，默认 900（15 分钟）"},
         "format": {"type": "string", "description": "extract 的输出格式，如 wav / mp3"},
         "language": {"type": "string", "description": "transcribe 语言：zh / en / ja / ko，默认 zh"},
         "max_speech": {"type": "number", "description": "transcribe 单段上限秒数，默认 20"},
         }}},
    {"name": "subtitles", "description": "字幕：单图 OCR / 字幕带逐帧 OCR + 折叠。",
     "inputSchema": {"type": "object", "properties": {
         "action": {"type": "string", "enum": ["ocr", "band"]},
         "video_id": {"type": "string", "description": "工作区里的视频文件夹名，不传用最近打开"},
         "image": {"type": "string", "description": "要 OCR 的图片路径"},
         "start_seconds": {"type": "number", "description": "从哪个时间戳开始，默认 0"},
         "seconds": {"type": "number", "description": "看多久（秒）；优先于 end_seconds"},
         "end_seconds": {"type": "number", "description": "结束秒（兼容用）"},
         "max_seconds": {"type": "number", "description": "单次上限秒数，默认 900（15 分钟）"},
         "format": {"type": "string", "enum": ["json", "srt"], "description": "srt 额外写字幕文件"},
         "save_to": {"type": "string", "description": "srt 落盘路径，默认 workspace/<视频>/out/subtitles.srt"},
         "step": {"type": "number", "description": "采样间隔秒，默认 1.0"},
         "band": {"description": "字幕带高度比例，默认 'auto'"},
         "skip_th": {"type": "number", "description": "跳帧阈值（带内文字块变化百分比），默认 1.0；设 -1 关闭跳帧"},
         "text_th": {"type": "number", "description": "文字高亮阈值，默认 220"},
         "width": {"type": "integer"},
         }}},
]

def tool_probe(a):
    cwd = str(Path(__file__).resolve().parent)
    asset = _load_asset(cwd, a.get("video_id"))
    if asset is None:
        return (_missing(a), [])
    return (json.dumps({k: asset.get(k) for k in ("id", "title", "duration", "width", "height", "fps", "has_audio", "media_path", "workspace") if asset.get(k) is not None},
                       ensure_ascii=False, indent=2), [])


def tool_peek(a):
    cwd = str(Path(__file__).resolve().parent)
    asset = _load_asset(cwd, a.get("video_id"))
    if asset is None:
        return (_missing(a), [])
    t = float(a.get("at") or a.get("seconds") or 0)
    ts = [t]
    frames = engine.extract_frames(asset["media_path"], ts, width=int(a.get("width") or 640),
                                  out_dir=asset.get("video_dir"), duration=asset["duration"])
    if not frames:
        return ("取帧失败", [])
    blocks = []
    for f in frames:
        data = base64.b64encode(Path(f["path"]).read_bytes()).decode()
        blocks.append({"type": "image", "data": data, "mimeType": "image/jpeg"})
    return ("peek t=%.2f" % frames[0]["seconds"], blocks)


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
    fw, _fh = _fit(vw, vh, _max_pixels(a))
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
            score = abs(math.log((c / r) * aspect)) + 0.05 * empty + (0.25 if flip else 0)
            key = (round(score, 3), empty)
            if best is None or key < best[0]:
                best = (key, c, r)
    cols, rows = best[1], best[2]
    need = min(cols * rows, 30)
    win_start, win_end, window_note = _window(a, asset)
    ts = engine.sample_times(win_start, win_end, need)
    frames = engine.extract_frames(asset["media_path"], ts, width=fw,
                                   out_dir=asset.get("video_dir"), duration=duration)
    sheet_path = None
    if frames:
        short = 190
        if aspect < 1:
            cell_w, cell_h = short, max(1, int(round(short / aspect)))
        else:
            cell_h, cell_w = short, max(1, int(round(short * aspect)))
        sp = Path(asset.get("video_dir") or ".")
        sp.mkdir(parents=True, exist_ok=True)
        sheet_path = str(sp / ("sheet_%s.png" % engine.timecode(duration).replace(":", "")))
        try:
            engine.compose_sheet(asset["media_path"], [f["seconds"] for f in frames], cols, rows,
                                 cell_w=cell_w, cell_h=cell_h, out_path=sheet_path,
                                 title=asset.get("title"))
        except Exception:
            sheet_path = None
    out = {
        "ok": True,
        "title": asset.get("title"),
        "duration": round(duration, 2),
        "width": asset.get("width"),
        "height": asset.get("height"),
        "frameSize": [fw, int(round(fw / aspect)) if aspect else None],
        "frames": len(frames),
        "timeline": [{"t": f["seconds"], "path": f["path"]} for f in frames],
        "sheet": sheet_path,
        "videoDir": asset.get("video_dir"),
        "nextStep": "描述画面前先看 sheet 或 timeline 里的图；细节用 action=frame 取原帧。",
    }
    return (json.dumps(out, ensure_ascii=False, indent=2), [])


def tool_frame(a):
    cwd = str(Path(__file__).resolve().parent)
    asset = _load_asset(cwd, a.get("video_id"))
    if asset is None:
        return (_missing(a), [])
    t = a.get("time", a.get("t"))
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
    out_dir = Path(asset.get("video_dir") or ".")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = hashlib.md5(",".join(vf).encode()).hexdigest()[:6]
    dest = out_dir / ("frame_%s-%s.jpg" % (engine.timecode(t).replace(":", ""), tag))
    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", "%.3f" % t, "-i", asset["media_path"],
           "-frames:v", "1", "-vf", ",".join(vf), str(dest)]
    r = engine._run(cmd, timeout=120)
    if not dest.exists():
        return ("取帧失败: %s" % (r or "ffmpeg 无输出"), [])
    data = base64.b64encode(dest.read_bytes()).decode()
    return (json.dumps({"ok": True, "t": round(t, 3), "save": str(dest),
                        "size": [tw, th], "quality": a.get("quality") or "medium"},
                       ensure_ascii=False),
            [{"type": "image", "data": data, "mimeType": "image/jpeg"}])


def dispatch_workspace(a):
    act = (a.get("action") or "open").lower()
    if act == "open":
        return tool_open(a)
    if act == "list":
        return tool_list(a)
    return ("未知 action: %s" % act, [])


def dispatch_video(a):
    act = (a.get("action") or "analyze").lower()
    if act in ("analyze", "frames", "watch", "probe", "sheet"):
        return tool_analyze(a)
    if act in ("frame", "peek"):
        return tool_frame(a)
    return ("未知 action: %s" % act, [])


def tool_transcribe(a):
    cwd = str(Path(__file__).resolve().parent)
    asset = _load_asset(cwd, a.get("video_id"))
    if asset is None:
        return (_missing(a), [])
    dur_total = float(asset.get("duration") or 0)
    if dur_total > 0 and float(a.get("start_seconds") or 0) >= dur_total:
        return (json.dumps({"ok": False,
                            "error": "start_seconds=%.1f 超出时长 %.1f" % (float(a["start_seconds"]), dur_total)},
                           ensure_ascii=False), [])
    start, end, window_note = _window(a, asset)
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
    if act in ("levels", "analyze"):
        return tool_audio(a)
    if act in ("extract",):
        return tool_audio(a)
    if act in ("transcribe", "transcript", "stt"):
        return tool_transcribe(a)
    return ("未知 action: %s" % act, [])


def dispatch_subtitles(a):
    act = (a.get("action") or "band").lower()
    if act == "ocr":
        return tool_ocr(a)
    if act in ("band", "subtitles"):
        return tool_subtitles(a)
    if act == "search":
        return ("search 待实现", [])
    return ("未知 action: %s" % act, [])


def dispatch(a):
    # 兼容旧入口：按 action 或 source 推断
    act = (a.get("action") or "").lower()
    if not act:
        return dispatch_workspace(a) if a.get("source") else dispatch_workspace(a)
    # 旧 action 名映射
    if act in ("open", "list", "status", "clean"):
        return dispatch_workspace(a)
    if act in ("probe", "frames", "sheet", "peek", "watch"):
        return dispatch_video(a)
    if act in ("levels", "analyze", "extract", "transcribe", "transcript", "audio"):
        return dispatch_audio(a)
    if act in ("ocr", "band", "subtitles", "search"):
        return dispatch_subtitles(a)
    return ("未知 action: %s" % act, [])


HANDLERS = {
    "workspace": dispatch_workspace,
    "video": dispatch_video,
    "audio": dispatch_audio,
    "subtitles": dispatch_subtitles,
    # 兼容旧名
    "video_vision": dispatch,
}


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
                           "serverInfo": {"name": "mcp-vision", "version": "0.2.0"}}}
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
            resp = handle(req)
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
    httpd = socketserver.TCPServer((host, port), H)
    httpd.serve_forever()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--http", type=int, default=0)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    if args.http:
        serve_http(args.http, args.host)
    else:
        serve_stdio()


if __name__ == "__main__":
    main()
