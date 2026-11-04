#!/usr/bin/env python3
"""mcp-video-vision MCP 服务器（stdio 或 --http PORT）。手写 JSON-RPC，无 mcp SDK 依赖。"""
import argparse, base64, json, os, sys, time, http.server, socketserver, threading
from pathlib import Path
import engine

ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE = ".mvv-cache"


def _mask(s):
    return s


def tool_open(a):
    src = (a.get("source") or a.get("file") or "").strip()
    if not src:
        return "要给 source（视频路径或链接）"
    cwd = a.get("cwd") or os.getcwd()
    # 本地文件
    p = Path(src)
    if not p.is_absolute():
        p = (Path(cwd) / p).resolve()
    if p.exists():
        asset = engine.open_local(cwd, str(p))
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
    ydl_opts = {"outtmpl": str(out_dir / "%(id)s.%(ext)s"), "format": "best[height<=720]/best", "quiet": True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(src, download=True)
            fname = ydl.prepare_filename(info)
    except Exception as e:
        return ("下载失败: %s" % e, [])
    asset = engine.open_local(cwd, fname)
    return ("已下载并打开 %s\n时长 %.2fs  %sx%s" % (
        asset["title"], asset["duration"], asset["width"], asset["height"]), [])


def tool_frames(a):
    cwd = a.get("cwd") or os.getcwd()
    asset = None
    vid = a.get("video_id")
    if vid:
        folder = Path(cwd) / DEFAULT_CACHE / "videos" / vid
        man = folder / "manifest.json"
        if man.exists():
            asset = json.loads(man.read_text())
    if asset is None:
        asset = engine.load_latest(cwd)
    if asset is None:
        return "先 open 一个视频", []
    media = asset["media_path"]
    width = int(a.get("width") or 1280)
    frames = []
    if a.get("timestamps"):
        ts = [float(x) for x in a["timestamps"]]
        frames = engine.extract_frames(media, ts, width=width, duration=asset["duration"])
    else:
        start = float(a.get("start_seconds", asset.get("cursor", 0)))
        end = float(a.get("end_seconds", min(start + 30, asset["duration"])))
        count = int(a.get("frames", 6))
        count = max(1, min(engine.MAX_FRAMES, count))
        ts = engine.sample_times(start, end, count)
        frames = engine.extract_frames(media, ts, width=width, duration=asset["duration"])
    # update cursor
    if frames and not a.get("timestamps"):
        asset["cursor"] = frames[-1]["seconds"]
        Path(asset and (Path(cwd)/DEFAULT_CACHE/"videos"/asset["id"]/"manifest.json")).write_text(
            json.dumps(asset, ensure_ascii=False, indent=2))
    # build content blocks
    blocks = []
    head = "frames=%d  range=%s" % (len(frames), ",".join("%.1f" % f["seconds"] for f in frames))
    for f in frames:
        data = base64.b64encode(Path(f["path"]).read_bytes()).decode()
        blocks.append({"type": "image", "data": data, "mimeType": "image/jpeg"})
    return (head, blocks)


def tool_sheet(a):
    cwd = a.get("cwd") or os.getcwd()
    asset = engine.load_latest(cwd)
    if asset is None:
        return "先 open 一个视频", []
    media = asset["media_path"]
    cols = int(a.get("cols", 3))
    rows = int(a.get("rows", 3))
    n = cols * rows
    dur = asset["duration"]
    ts = engine.sample_times(0, dur, n)
    out = Path(cwd) / "out" / "sheet.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cell_w = 400
    cell_h = 225
    engine.compose_sheet(media, ts, cols, rows, cell_w, cell_h, out)
    data = base64.b64encode(out.read_bytes()).decode()
    return ("sheet: %s  (%dx%d grid, %d frames)" % (out, cols, rows, len(ts)),
            [{"type": "image", "data": data, "mimeType": "image/png"}])


def tool_audio(a):
    cwd = a.get("cwd") or os.getcwd()
    asset = None
    vid = a.get("video_id")
    if vid:
        man = Path(cwd) / DEFAULT_CACHE / "videos" / vid / "manifest.json"
        if man.exists():
            asset = json.loads(man.read_text())
    if asset is None:
        asset = engine.load_latest(cwd)
    if asset is None:
        return "先 open 一个视频", []
    start = float(a.get("start_seconds", 0))
    end = float(a.get("end_seconds", min(start + 30, asset["duration"])))
    res = engine.analyze_audio(asset["media_path"], start, end)
    return (json.dumps(res, ensure_ascii=False, indent=2)[:8000], [])


def tool_ocr(a):
    cwd = a.get("cwd") or os.getcwd()
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
    cwd = a.get("cwd") or os.getcwd()
    try:
        import ocr as ocrmod
    except Exception as e:
        return ("OCR 不可用: %s" % e, [])
    asset = None
    vid = a.get("video_id")
    if vid:
        man = Path(cwd) / DEFAULT_CACHE / "videos" / vid / "manifest.json"
        if man.exists():
            asset = json.loads(man.read_text())
    if asset is None:
        asset = engine.load_latest(cwd)
    if asset is None:
        return ("先 open 一个视频", [])
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
            skip_th=float(a.get("skip_th", 3.0)),
            engine=ocrmod.get_engine())
    except Exception as e:
        return ("字幕提取失败: %s" % e, [])
    segs = []
    for e in res["results"]:
        txt = " ".join(x for x in e["lines"] if x)
        if segs and segs[-1]["text"] == txt:
            segs[-1]["_to"] = e["seconds"]
        else:
            segs.append({"from": ocrmod._tc(e["seconds"]), "to": ocrmod._tc(e["seconds"]),
                         "_to": e["seconds"], "text": txt})
    for sg in segs:
        sg["to"] = ocrmod._tc(sg.pop("_to"))
    out = {"frames": res["frames"], "ocr_runs": res["ocr_runs"], "skipped": res["skipped"], "band": res.get("band"), "band_auto": res.get("band_auto"),
           "band_px": res["band_px"], "segment_count": len(segs), "segments": segs}
    return (json.dumps(out, ensure_ascii=False, indent=2)[:8000], [])


def tool_status(a):
    cwd = a.get("cwd") or os.getcwd()
    st = engine.status(cwd)
    asset = engine.load_latest(cwd)
    info = {"cache": st}
    if asset:
        info["latest"] = {"id": asset["id"], "title": asset.get("title"),
                          "duration": asset["duration"], "resolution": "%sx%s" % (asset["width"], asset["height"]),
                          "cursor": asset.get("cursor", 0)}
    return (json.dumps(info, ensure_ascii=False, indent=2), [])


def tool_clean(a):
    cwd = a.get("cwd") or os.getcwd()
    root = Path(cwd) / DEFAULT_CACHE
    if root.exists():
        shutil.rmtree(root)
        return ("已清空 %s" % root, [])
    return ("无缓存", [])


TOOLS = [
    {"name": "video_vision", "description": "观看视频：抽帧/故事板/音频分析/状态。纯 Python，无 Node。",
     "inputSchema": {"type": "object", "properties": {
         "action": {"type": "string", "enum": ["open", "frames", "sheet", "audio", "ocr", "subtitles", "status", "clean"]},
         "source": {"type": "string", "description": "视频路径或链接"},
         "image": {"type": "string", "description": "要 OCR 的图片路径"},
         "video_id": {"type": "string"},
         "start_seconds": {"type": "number"},
         "end_seconds": {"type": "number"},
         "timestamps": {"type": "array", "items": {"type": "number"}},
         "frames": {"type": "integer"},
         "width": {"type": "integer"},
         "cols": {"type": "integer"},
         "step": {"type": "number", "description": "采样间隔秒，默认 1.0"},
         "band": {"description": "字幕带高度比例，默认 'auto'（自动定位），也可指定如 [0.75, 0.95]"},
         "skip_th": {"type": "number", "description": "跳帧阈值，默认 3.0"},
         "rows": {"type": "integer"},
         "cwd": {"type": "string"}}}}]


def dispatch(a):
    act = (a.get("action") or ("open" if a.get("source") else "status")).lower()
    if act == "open":
        return tool_open(a)
    if act in ("frames", "watch"):
        return tool_frames(a)
    if act == "sheet":
        return tool_sheet(a)
    if act == "audio":
        return tool_audio(a)
    if act == "ocr":
        return tool_ocr(a)
    if act == "subtitles":
        return tool_subtitles(a)
    if act == "status":
        return tool_status(a)
    if act == "clean":
        return tool_clean(a)
    return ("未知 action: %s" % act, [])


HANDLERS = {"video_vision": dispatch}


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
