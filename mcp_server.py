#!/usr/bin/env python3
"""mcp-video-vision MCP 服务器（stdio 或 --http PORT）。手写 JSON-RPC，无 mcp SDK 依赖。"""
import argparse, base64, json, math, os, re, sys, time, http.server, socketserver, threading
from pathlib import Path
import engine

ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE = ".mvv-cache"


def _mask(s):
    return s


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
    if max_h is None:
        fmt = "bestvideo+bestaudio/best"
    else:
        # 用 min(w,h) 而非 height：竖屏视频 height 是长边，会误杀
        fmt = "bestvideo[min(w,h)<=%d]+bestaudio/bestvideo+bestaudio/best" % max_h
    ydl_opts = {
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "format": fmt,
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
    asset = engine.open_local(cwd, fname)
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
    start = float(a.get("start_seconds", 0))
    end = float(a.get("end_seconds", min(start + 30, asset["duration"])))
    res = engine.analyze_audio(asset["media_path"], start, end, out_dir=asset.get("audio_dir"))
    return (json.dumps(res, ensure_ascii=False, indent=2)[:8000], [])


def tool_ocr(a):
    cwd = str(Path(__file__).resolve().parent)
    try:
        import ocr as ocrmod
    except Exception as e:
        return ("OCR 不可用: %s（先 pkg install python-onnxruntime 并跑 scripts/fetch-models.py）" % e, [])
    src = a.get("image") or a.get("path")
    if src:
        p = Path(src)
        if not p.is_absolute():
            p = (Path(cwd) / p).resolve()
        if not p.exists():
            return ("找不到图片: %s" % p, [])
        try:
            rgb = ocrmod.load_rgb(p)
            o = ocrmod.PPOCR()
            res = o.ocr_image(rgb)
        except Exception as e:
            return ("OCR 失败: %s" % e, [])
        return (json.dumps({"image": str(p), "providers": o.providers, "count": len(res), "results": res},
                           ensure_ascii=False, indent=2)[:8000], [])
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
        o = ocrmod.PPOCR()
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
    return (json.dumps({"frames": len(out), "providers": o.providers, "results": out},
                       ensure_ascii=False, indent=2)[:8000], [])


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
    start = float(a.get("start_seconds", 0))
    end = a.get("end_seconds")
    end = float(end) if end is not None else float(asset["duration"])
    band = a.get("band", "auto")
    try:
        res = ocrmod.ocr_subtitles(
            media, start, end,
            step=float(a.get("step", 1.0)),
            band=band if band == "auto" else (float(band[0]), float(band[1])),
            width=int(a.get("width") or 960),
            skip_th=float(a.get("skip_th", 0.5)),
            text_th=float(a.get("text_th", 220)),
            engine=ocrmod.get_engine())
    except Exception as e:
        return ("字幕提取失败: %s" % e, [])
    segs = ocrmod.fold_segments(res["results"])
    out = {"frames": res["frames"], "ocr_runs": res["ocr_runs"], "skipped": res["skipped"], "band": res.get("band"), "band_auto": res.get("band_auto"),
           "band_px": res["band_px"], "segment_count": len(segs), "segments": segs}
    return (json.dumps(out, ensure_ascii=False, indent=2)[:8000], [])


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
    {"name": "video", "description": "视频：analyze 整片理解（时间线抽帧 + 近似正方形网格页）；frame 单帧核对（可裁剪/缩放）。",
     "inputSchema": {"type": "object", "properties": {
         "action": {"type": "string", "enum": ["analyze", "frame"]},
         "video_id": {"type": "string", "description": "工作区里的视频文件夹名（workspace/<名>），不传用最近打开"},
         "video": {"type": "string", "description": "视频路径或链接；不传则用最近打开的视频"},
         "baseFps": {"type": "number", "description": "analyze 采样帧率，默认 5，上限 12 帧"},
         "t": {"type": "number", "description": "frame 的时间点（秒）"},
         "crop": {"type": "string", "description": "frame 裁剪：WxH+X+Y，如 720x300+0+880"},
         "scale": {"type": "integer", "description": "frame 缩放宽度（像素）"},
         }}},
    {"name": "audio", "description": "音频：分析 (RMS/峰值/静音/节拍)。transcribe 待实现。",
     "inputSchema": {"type": "object", "properties": {
         "action": {"type": "string", "enum": ["levels", "extract"]},
         "video_id": {"type": "string", "description": "工作区里的视频文件夹名，不传用最近打开"},
         "start_seconds": {"type": "number"},
         "end_seconds": {"type": "number"},
         "format": {"type": "string"},
         }}},
    {"name": "subtitles", "description": "字幕：单图 OCR / 字幕带逐帧 OCR + 折叠。",
     "inputSchema": {"type": "object", "properties": {
         "action": {"type": "string", "enum": ["ocr", "band"]},
         "video_id": {"type": "string", "description": "工作区里的视频文件夹名，不传用最近打开"},
         "image": {"type": "string", "description": "要 OCR 的图片路径"},
         "start_seconds": {"type": "number"},
         "end_seconds": {"type": "number"},
         "step": {"type": "number", "description": "采样间隔秒，默认 1.0"},
         "band": {"description": "字幕带高度比例，默认 'auto'"},
         "skip_th": {"type": "number", "description": "跳帧阈值，默认 0.5"},
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
    base_fps = float(a.get("baseFps") or a.get("base_fps") or 5)
    vw = float(asset.get("width") or 16)
    vh = float(asset.get("height") or 9)
    aspect = vw / vh
    n0 = max(1, min(12, int(round(duration * base_fps)) or 1))
    # 先定网格，让整张图接近正方形；再按 cols*rows 抽帧，避免空格
    best = None
    for c in range(1, n0 + 1):
        base_r = (n0 + c - 1) // c
        for r in (base_r, base_r + 1):
            empty = c * r - n0
            if empty > 3:
                continue
            score = abs(math.log((c / r) * aspect)) + 0.05 * empty
            key = (round(score, 3), empty)
            if best is None or key < best[0]:
                best = (key, c, r)
    cols, rows = best[1], best[2]
    need = min(cols * rows, 18)
    ts = engine.sample_times(0, duration, need)
    frames = engine.extract_frames(asset["media_path"], ts, width=int(a.get("width") or 640),
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
    if a.get("t") is None:
        return ("frame 需要 t（秒）", [])
    t = float(a.get("t"))
    width = int(a.get("scale") or a.get("width") or 0) or None
    try:
        crop = _parse_crop(a.get("crop"))
    except ValueError as e:
        return (str(e), [])
    vf = [x for x in (crop,) if x]
    if width:
        vf.append("scale=%d:-1" % width)
    out_dir = Path(asset.get("video_dir") or ".")
    out_dir.mkdir(parents=True, exist_ok=True)
    import hashlib
    tag = hashlib.md5((",".join(vf) or "raw").encode()).hexdigest()[:6]
    dest = out_dir / ("frame_%s-%s.jpg" % (engine.timecode(t).replace(":", ""), tag))
    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", "%.3f" % t, "-i", asset["media_path"],
           "-frames:v", "1"]
    if vf:
        cmd += ["-vf", ",".join(vf)]
    cmd.append(str(dest))
    r = engine._run(cmd, timeout=120)
    if not dest.exists():
        return ("取帧失败: %s" % (r or "ffmpeg 无输出"), [])
    data = base64.b64encode(dest.read_bytes()).decode()
    return (json.dumps({"ok": True, "t": round(t, 3), "path": str(dest)}, ensure_ascii=False),
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


def dispatch_audio(a):
    act = (a.get("action") or "levels").lower()
    if act in ("levels", "analyze"):
        return tool_audio(a)
    if act in ("extract",):
        return tool_audio(a)
    if act in ("transcribe", "transcript"):
        return ("transcribe/transcript 待实现（sherpa-onnx）", [])
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
                           "serverInfo": {"name": "mcp-video-vision", "version": "0.1.0"}}}
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
