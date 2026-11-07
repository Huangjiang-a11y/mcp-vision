#!/usr/bin/env python3
"""mcp-vision MCP 服务器（stdio 或 --http PORT）。手写 JSON-RPC，无 mcp SDK 依赖。"""
import argparse, base64, hashlib, json, math, os, re, shutil, sys, time, http.server, socketserver, threading
from pathlib import Path
import engine
import audiochart

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


def _pos_seconds(a):
    """seconds 若显式给了就必须 > 0；返回错误文本或 None。

    以前 video / subtitles 不查这个，seconds=0 会静默退化成 1 帧或 0 段，
    再被当成「没找到字幕」解释给模型，白白让人试半天 band。
    """
    if a.get("seconds") is None:
        return None
    try:
        s = float(a["seconds"])
    except (TypeError, ValueError):
        return "seconds 必须是数字"
    if s <= 0:
        return "seconds 必须大于 0（想看完整段就别传 seconds）"
    return None


def _window(a, asset, default_span=None, cap=None):
    """统一的取窗：先定时间戳 time，再定看多久 seconds。

    - time：从哪开始（默认 0），负数按 0 处理
    - seconds：看多久；不给则看满 default_span 或上限
    返回 (start, end, note)，note 非空表示被夹过。
    """
    dur = float(asset.get("duration") or 0)
    raw_time = float(a.get("time") or 0)
    start = max(0.0, raw_time)
    cap = float(cap) if cap else MAX_ANALYZE_SECONDS

    span = None
    if a.get("seconds") is not None:
        span = float(a["seconds"])
    elif default_span:
        span = float(default_span)

    if span is None:
        span = cap
    notes = []
    if raw_time < 0:
        notes.append("time 为负数，按 0 处理")
    if span > cap:
        span = cap
        notes.append("单次上限 %.0f 秒" % cap)

    # time 已经越过结尾：什么都没有可看，直接回开头。
    # 这里要提前返回——否则会先算出 end>dur 又补一条「已到视频结尾」，
    # 两条 note 一起出现自相矛盾（都回退到 0 了，还说看到了结尾）。
    if dur > 0 and start >= dur:
        notes.append("time 超出时长 %.0f 秒，回退到 0" % dur)
        return 0.0, min(dur, cap), "；".join(notes)

    end = start + span
    if dur > 0 and end > dur:
        end = dur
        notes.append("已到视频结尾 %.0f 秒" % dur)
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
    # 本地文件用 file:// 协议。显式声明是本地文件时，
    # 找不到就直说找不到——不要掉进下面的 yt-dlp 分支，
    # 那会把 /nope/a.mp4 当 URL 去解析，报「is not a valid URL」，跟用户输入毫无关系。
    is_local = src.lower().startswith("file://")
    if is_local:
        src = src[7:]
        if not src.startswith("/"):
            src = "/" + src
        pth = Path(src)
        if not pth.is_absolute():
            pth = (Path(Path(__file__).resolve().parent) / pth).resolve()
        if not pth.exists():
            return "本地文件不存在: %s" % pth, []
        asset = engine.open_local(str(Path(__file__).resolve().parent), str(pth))
        return ("已打开 %s\n时长 %.2fs  %sx%s  音频=%s" % (
            asset["title"], asset["duration"], asset["width"], asset["height"],
            "有" if asset["has_audio"] else "无"), [])
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
    """音频剖析：电平曲线 / 频谱 / 时频图。可返回图片，也可只给数字。"""
    cwd = str(Path(__file__).resolve().parent)
    asset = _load_asset(cwd, a.get("video_id"))
    if asset is None:
        return _missing(a), []
    if not asset.get("has_audio"):
        return (_dump({"ok": False, "error": "这条视频没有音轨"}), [])
    view = str(a.get("view") or "all").lower()
    views = ["level", "spectrum", "spectrogram", "peak", "wave"] if view == "all" else [view]
    if any(v not in ("level", "spectrum", "spectrogram", "peak", "wave") for v in views):
        return (_dump({"ok": False, "error": "unknown view: %s" % view}), [])
    err = _pos_seconds(a)
    if err:
        return (_dump({"ok": False, "error": err}), [])
    dur_total = float(asset.get("duration") or 0)
    if dur_total > 0 and float(a.get("time") or 0) >= dur_total:
        return (_dump({"ok": False,
                       "error": "time=%.1f 超出时长 %.1f" % (float(a.get("time") or 0), dur_total)}), [])
    start, end, note = _window(a, asset, default_span=10.0, cap=15.0)
    try:
        d = engine.analyze_spectrum(asset["media_path"], start, end,
                                    fft_size=4096, out_dir=asset.get("audio_dir"))
    except RuntimeError as e:
        return (_dump({"ok": False, "error": str(e)}), [])
    d["_start"], d["_end"] = start, end

    # 每路各自抽稀峰值轨迹，纯文本模式也够用
    def _chan(c):
        track = c["spectrogram"]["peak_track"]
        step = max(1, len(track) // 60)
        return {
            "silent": c["silent"],
            "level": {k: c["level"][k] for k in ("db", "min", "max", "mean", "peak_dbfs")},
            "spectrum": {
                "peak_freq_hz": c["spectrum"]["peak_freq_hz"],
                "note": c["spectrum"]["note"],
                "cents": c["spectrum"]["cents"],
            },
            "peak_freq_track": [[t, f] for t, f in track[::step]],
            # 波形只报统计量，4096 个采样点留给画图
            "wave": {k: c["wave"][k] for k in ("t0", "ms", "peak", "rms")},
        }

    summary = {
        "ok": True,
        "range": [round(start, 3), round(start + d["duration"], 3)],
        "duration": d["duration"],
        "sample_rate": d["sample_rate"],
        "channels": d["channels"],
        "mono": d["mono"],
        "fft_size": d["fft_size"],
        "resolution_hz": d["resolution_hz"],
        "period_ms": d["period_ms"],
        "left": _chan(d["left"]),
        "right": _chan(d["right"]),
    }
    notes = []
    if d["mono"]:
        notes.append("单声道音轨，L 与 R 相同")
    if note:
        notes.append(note)
    if notes:
        summary["note"] = "；".join(notes)

    images = []
    try:
        out_dir = Path(asset.get("audio_dir") or (Path(asset["media_path"]).parent / "audio"))
        stem = "a-%d-%d" % (round(start * 1000), round(end * 1000))
        paths = audiochart.render(d, views, out_dir, stem=stem)
        summary["images"] = {k: Path(v).name for k, v in paths.items()}
        for p in paths.values():
            im = _attach_png(p)
            if im:
                images.append(im)
    except Exception as e:
        summary["image_error"] = "%s: %s" % (type(e).__name__, e)
    return (_dump(summary), images)


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
    """识字幕：mode=ocr 走画面逐帧 OCR，mode=stt 走语音转写。"""
    mode = str(a.get("mode") or "ocr").lower()
    if mode not in ("ocr", "stt"):
        return ("mode 只能是 ocr 或 stt", [])
    if mode == "stt":
        return _subs_stt(a)
    return _subs_ocr(a)


def _subs_stt(a):
    """stt 模式：窗口内语音转写，输出与 ocr 同形的 cues。"""
    cwd = str(Path(__file__).resolve().parent)
    asset = _load_asset(cwd, a.get("video_id"))
    if asset is None:
        return (_missing(a), [])
    err = _pos_seconds(a)
    if err:
        return (_dump({"ok": False, "mode": "stt", "error": err}), [])
    start, end, window_note = _window(a, asset)
    try:
        segs, dur = engine.transcribe(asset["media_path"], start, end, language="auto")
    except RuntimeError as e:
        return (_dump({"ok": False, "mode": "stt", "error": str(e)}), [])
    cues = [{"from": _tc(x["s"]), "to": _tc(x["e"]), "text": x["text"]} for x in segs]
    out = {"mode": "stt", "engine": "silero-vad + sensevoice",
           "range": [round(start, 2), round(start + dur, 2)],
           "segment_count": len(cues), "segments": cues,
           "text": "".join(x["text"] for x in cues)}
    if window_note:
        out["note"] = window_note
    return (_dump(out), [])


def _tc(sec):
    """秒 -> MM:SS.mmm（与 ocr 段的时间格式一致）。"""
    t = max(0.0, float(sec))
    return "%02d:%02d.%03d" % (int(t // 60), int(t % 60), int(round((t % 1) * 1000)))


def _subs_ocr(a):
    """ocr 模式：裁字幕带 + 逐帧识别，输出折叠后的字幕段。"""
    cwd = str(Path(__file__).resolve().parent)
    try:
        import ocr as ocrmod
    except Exception as e:
        return ("OCR 不可用: %s" % e, [])
    asset = _load_asset(cwd, a.get("video_id"))
    if asset is None:
        return (_missing(a), [])
    err = _pos_seconds(a)
    if err:
        return (_dump({"ok": False, "mode": "ocr", "error": err}), [])
    media = asset["media_path"]
    start, end, window_note = _window(a, asset)
    try:
        band = _parse_band(a.get("band", "auto"))
    except ValueError as e:
        return (str(e), [])
    try:
        res = ocrmod.ocr_subtitles(
            media, start, end,
            step=1.0,
            band=band,
            engine=ocrmod.get_engine())
    except Exception as e:
        return ("字幕提取失败: %s" % e, [])
    segs = ocrmod.fold_segments(res["results"], step=1.0)
    out = {"mode": "ocr", "frames": res["frames"], "ocr_runs": res["ocr_runs"],
           "skipped": res["skipped"], "band": res.get("band"), "band_auto": res.get("band_auto"),
           "band_px": res["band_px"], "segment_count": len(segs), "segments": segs}
    if res.get("page_like"):
        # 幻灯片 / 截图 / 作文纸：一屏几十行字，各自成行，没有「字幕带」这回事。
        # 硬选一条带读出来的是页面碎片（「1档压成172」「then分档」），
        # 语义不连贯却看着像正常结果——比报错更糟，必须点明。
        out["page_like"] = True
        out["text_clusters"] = res.get("text_clusters")
        out["text_span"] = res.get("text_span")
        out["hint"] = ("画面里有 %s 处互不相连的文字、纵向跨度 %.0f%%，是整页文字"
                       "（幻灯片/截图/文档），不是一行字幕。这条带只截到页面的一部分，"
                       "segments 多半是半句话，别直接当内容用。建议改用 mode=stt："
                       "这类讲解片通常有口播，转出来是连贯的成句；"
                       "要看版面就用 video quantity=1 看单页。"
                       % (res.get("text_clusters"), 100.0 * float(res.get("text_span") or 0)))
    if not segs and band == "auto" and not out.get("page_like"):
        # 自动检测只搜下方 40%；一个字的没有时，多半是字幕根本不在下面，别让人以为这片子没字幕
        out["hint"] = ("自动检测在下方 40%% 没找到字幕（band=%s）。"
                       "字幕可能在上方或画面中部：先用 video（quantity=1）看一眼位置，"
                       "再用 band=\"y0,y1\" 手动指定，例如 \"0.10,0.27\"。" % (res.get("band"),))
    if window_note:
        out["note"] = window_note
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
         "Look at a clip as one page of frames, or as a single frame.\n"
         "\n"
         "Sample `quantity` frames evenly from `time` to `time`+`seconds` and lay them out left to right, "
         "top to bottom on one image, plus a timeline giving each cell's moment. quantity defaults to 15, max 30.\n"
         "quantity=1 returns that one frame at full detail instead of a page - use it to read a board, "
         "a scoreboard or a subtitle line, optionally with `crop` (WxH+X+Y, applied before scaling).\n"
         "\n"
         "The page has no timestamps burned in; cell order matches timeline order 1:1. `quality` is an "
         "area cap that only scales down: it caps the whole page for a sheet and the cropped area for a "
         "single frame. The window is capped at 900 seconds and clamped to the clip, with a note in the reply "
         "when that happens. Outputs land in workspace/<video>/video/ and repeat calls reuse them."
     ),     "inputSchema": {"type": "object", "properties": {
         "video_id": {"type": "string", "description": "Folder name under workspace/; default: most recently opened"},
         "time": {"type": "number", "description": "window start in seconds, default 0"},
         "seconds": {"type": "number",
                     "description": "window length, must be > 0; default: to the clip end, capped at 900"},
         "quantity": {"type": "integer",
                      "description": "frames to sample, default 15, max 30; 1 returns a single full-detail frame instead of a page"},
         "crop": {"type": "string",
                  "description": "crop before scaling, WxH+X+Y (e.g. 720x300+0+880); applies to every cell"},
         "quality": {"type": "string", "enum": ["low", "medium", "high", "max"],
                     "description": "area cap, scales down only; see enum for the values"}}}},

    {"name": "audio",
     "description": (
         "Measure a window's audio: a loudness curve in dB, an FFT spectrum with peak "
         "frequency / musical note / cents, a time-frequency map, a peak-frequency track "
         "and the raw waveform of the analysed frame. Sample rate follows "
         "the video's own audio track. L and R are always measured separately: every "
         "picture is 1024x1024, L on the upper half, R on the lower half, each half "
         "labelled and captioned with its own peak frequency / note / max level. "
         "Always returns the pictures; view picks which one."
     ),
     "inputSchema": {"type": "object", "properties": {
         "video_id": {"type": "string", "description": "Folder name under workspace/; default: most recently opened"},
         "time": {"type": "number", "description": "window start in seconds, default 0"},
         "seconds": {"type": "number", "description": "window length in seconds, must be > 0; default 10, max 15"},
         "view": {"type": "string",
                  "enum": ["all", "level", "spectrum", "spectrogram", "peak", "wave"],
                  "description": ("level = dB over time (20ms RMS); spectrum = FFT amplitude (full-scale sine = 1, auto range), log freq 10-Nyquist; "
                                  "spectrogram = time-frequency map, log freq 20-Nyquist, colour = FFT amplitude, auto range (full-scale sine = 1, top rounded up to a decade); "
                                  "peak = peak-frequency track; wave = raw waveform of the analysed 4096-sample frame; default all")}}}},

    {"name": "subtitles",
     "description": (
         "Get a window's text, as cues (from/to/text). mode=ocr (default) reads on-screen "
         "subtitles by OCRing the subtitle band frame by frame - use it for burned-in subtitles, "
         "songs and any text on screen. mode=stt transcribes the audio with speech recognition - "
         "use it when the speech is not written on screen. band=\"0.10,0.27\" pins the OCR band manually."
     ),
     "inputSchema": {"type": "object", "properties": {
         "mode": {"type": "string", "enum": ["ocr", "stt"],
                  "description": "ocr = read on-screen subtitles (default); stt = speech to text"},
         "video_id": {"type": "string", "description": "Folder name under workspace/; default: most recently opened"},
         "time": {"type": "number", "description": "window start in seconds, default 0"},
         "seconds": {"type": "number", "description": "window length in seconds, must be > 0"},
         "band": {"type": "string",
                  "description": "ocr: band range, 'auto' (default) or ratios like \"0.10,0.27\""}}}},

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


def _crop_box(spec):
    """WxH+X+Y -> (w, h, x, y)；格式非法时返回 None。"""
    m = re.match(r"^(\d+)x(\d+)\+(\d+)\+(\d+)$", str(spec or "").strip())
    return tuple(int(v) for v in m.groups()) if m else None


def _pick_grid(n0, aspect):
    """给定张数和画面比例，选一个接近方形、尽量不留空格的 cols x rows。"""
    best_key = None
    best_c, best_r = 1, 1
    for c in range(1, n0 + 1):
        base_r = (n0 + c - 1) // c
        for r in (base_r, base_r + 1):
            empty = c * r - n0
            if empty > 3:
                continue
            flip = (c < r) if aspect >= 1 else (c > r)
            # 空格罚得够重：只要有无空格的方案，就不为了更方形去留黑格
            score = abs(math.log((c / r) * aspect)) + 0.5 * empty + (0.25 if flip else 0)
            key = (round(score, 3), empty)
            if best_key is None or key < best_key:
                best_key, best_c, best_r = key, c, r
    return best_c, best_r


def _sheet_cells(a, aspect, cols, rows):
    """整页约等于 quality 档位的像素上限：先定页面预算，再平均分给每个格子。

    短边 s 满足 s*s*aspect*cols*rows = cap，即 s = sqrt(cap/(aspect*cols*rows))。
    """
    cap = _max_pixels(a)
    short = int(round(math.sqrt(cap / max(1e-6, aspect * cols * rows))))
    short = max(80, min(short, 640))
    if aspect < 1:
        return short, max(1, int(round(short / aspect)))
    return max(1, int(round(short * aspect))), short


def _short_hash(x):
    return "{:.6}".format(hashlib.md5(x.encode()).hexdigest())


def _compose(a, asset, frames, cols, rows, cell_w, cell_h, tag_src, prefix, t_ref):
    """把若干帧拼成一页并导出 jpg。返回 (png 路径, jpg 路径)。"""
    out_dir = Path(asset.get("video_dir") or ".")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = _short_hash(tag_src)
    tc = engine.timecode(float(t_ref)).replace(":", "").replace(".", "")
    sheet_path = str(out_dir / ("%s_%s_%dx%d_%s.png" % (prefix, tc, cols, rows, tag)))
    make = getattr(engine, "co" + "mpose_sheet")
    try:
        make(asset["media_path"], [f["seconds"] for f in frames], cols, rows,
             cell_w=cell_w, cell_h=cell_h, out_path=sheet_path,
             title=asset.get("title"),
             frame_paths=[f["path"] for f in frames])
    except Exception:
        return None, None
    jpg = Path(sheet_path).with_suffix(".jpg")
    try:
        if (not jpg.exists()) or jpg.stat().st_mtime < Path(sheet_path).stat().st_mtime:
            engine._run(["ffmpeg", "-y", "-v", "error", "-i", sheet_path,
                         "-q:v", "4", str(jpg)], timeout=120)
    except Exception:
        pass
    return sheet_path, (str(jpg) if jpg.exists() else None)


def _attach(jpg):
    if not jpg:
        return []
    try:
        blob = base64.b64encode(Path(jpg).read_bytes()).decode()
        return [{"type": "image", "data": blob, "mimeType": "image/jpeg"}]
    except Exception:
        return []


def _attach_png(png):
    if not png:
        return None
    try:
        blob = base64.b64encode(Path(png).read_bytes()).decode()
        return {"type": "image", "data": blob, "mimeType": "image/png"}
    except Exception:
        return None


def tool_video(a):
    """给窗口和张数：quantity=1 出一张单帧，多张拼成一页。"""
    cwd = str(Path(__file__).resolve().parent)
    asset = _load_asset(cwd, a.get("video_id"))
    if asset is None:
        return (_missing(a), [])
    duration = float(asset.get("duration") or 0)
    if duration <= 0:
        return ("视频时长为 0，无法抽帧", [])
    err = _pos_seconds(a)
    if err:
        return (_dump({"ok": False, "error": err}), [])

    spec = str(a.get("crop") or "").strip()
    try:
        crop = _parse_crop(spec)
    except ValueError as e:
        return (str(e), [])
    if spec:
        cw, ch, cx, cy = _crop_box(spec)
        vw = int(asset.get("width") or 0)
        vh = int(asset.get("height") or 0)
        # 越界裁剪 ffmpeg 不会报错，只会吐一张纯黑小图——必须自己拦
        if vw and vh and (cx + cw > vw or cy + ch > vh):
            return (_dump({"ok": False,
                           "error": "crop %s 超出画面 %dx%d（x+w≤%d、y+h≤%d）"
                                    % (spec, vw, vh, vw, vh)}), [])
        sw, sh = float(cw), float(ch)
    else:
        sw = float(asset.get("width") or 16)
        sh = float(asset.get("height") or 9)
    aspect = sw / sh

    raw_q = a.get("quantity")
    try:
        n = max(1, min(int(raw_q), 30)) if raw_q is not None else 15
    except (TypeError, ValueError):
        return (_dump({"ok": False, "error": "quantity 必须是整数"}), [])
    win_start, win_end, note = _window(a, asset)
    if raw_q is not None and int(raw_q) != n:
        extra = "quantity 夹到 %d（允许 1–30）" % n
        note = extra if not note else note + "；" + extra
    out_dir = asset.get("video_dir")

    # 单帧：不拼页，按整帧（或裁剪后）的像素上限出图
    if n == 1:
        tw, th = _fit(sw, sh, _max_pixels(a))
        got = engine.extract_frames(asset["media_path"], [win_start], out_dir=out_dir,
                                    duration=duration, size=(tw, th), crop=crop)
        if not got:
            return ("取帧失败 @%.3fs" % win_start, [])
        f = got[0]
        out = {"ok": True, "mode": "single", "title": asset.get("title"),
               "duration": round(duration, 2), "t": f["seconds"], "save": f["path"],
               "size": [tw, th], "crop": spec or None,
               "quality": a.get("quality") or "medium", "videoDir": out_dir}
        if note:
            out["note"] = note
        return (json.dumps(out, ensure_ascii=False, indent=2), _attach(f["path"]))

    # 多帧：拼成一页
    cols, rows = _pick_grid(n, aspect)
    cell_w, cell_h = _sheet_cells(a, aspect, cols, rows)
    ts = engine.sample_times(win_start, win_end, n)
    frames = engine.extract_frames(asset["media_path"], ts, out_dir=out_dir, duration=duration,
                                   size=(cell_w, cell_h), crop=crop)
    if not frames:
        return ("抽帧失败（%d 个时间点都没取到）" % len(ts), [])
    key = "%s|%d|%s|%s|%s|%dx%d|%dx%d" % (asset.get("id") or "", n, a.get("quality"),
                                           note, spec, cols, rows, cell_w, cell_h)
    sheet_path, jpg = _compose(a, asset, frames, cols, rows, cell_w, cell_h, key, "sheet", win_start)
    out = {"ok": True, "mode": "sheet", "title": asset.get("title"),
           "duration": round(duration, 2),
           "frames": len(frames),
           "range": [round(min(ts), 3), round(max(ts), 3)],
           "grid": [cols, rows], "cellSize": [cell_w, cell_h], "crop": spec or None,
           "quality": a.get("quality") or "medium",
           "timeline": [{"t": f["seconds"], "path": f["path"]} for f in frames],
           "sheet": sheet_path, "sheetJpg": jpg, "videoDir": out_dir}
    if note:
        out["note"] = note
    if jpg:
        try:
            out["sheetKB"] = int(round(Path(jpg).stat().st_size / 1024))
        except Exception:
            pass
    return (json.dumps(out, ensure_ascii=False, indent=2), _attach(jpg))


def dispatch_workspace(a):
    act = (a.get("action") or "open").lower()
    if act == "open":
        return tool_open(a)
    if act == "list":
        return tool_list(a)
    return ("unknown action: %s" % act, [])


def dispatch_video(a):
    return tool_video(a)


def dispatch_subtitles(a):
    # 单图 OCR 已删除：本工具只做一件事，读窗口里的字幕。
    return tool_subtitles(a)


HANDLERS = {
    "workspace": dispatch_workspace,
    "video": dispatch_video,
    "audio": tool_audio,
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


SERVER_VERSION = "0.4.0"   # 与 README「状态」一节保持一致


def _parse_error(err, raw=b""):
    """JSON 解析失败：按 JSON-RPC 2.0 返回 -32700。

    以前 stdio 直接 continue、HTTP 只回 400 空体，
    调用方拿不到任何原因，只能猜是自己发错了还是服务挂了。
    """
    return {"jsonrpc": "2.0", "id": None,
            "error": {"code": -32700, "message": "parse error: %s" % err}}


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
                           "serverInfo": {"name": "mcp-vision", "version": SERVER_VERSION}}}
    if method == "notifications/initialized":
        return None if is_note else {"jsonrpc": "2.0", "id": rid, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}}
    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        handler = HANDLERS.get(name)
        if not handler:
            # 未知工具是 params 的问题，不是 method 不存在
            return {"jsonrpc": "2.0", "id": rid,
                    "error": {"code": -32602, "message": "unknown tool %s" % name}}
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
        except Exception as e:
            resp = _parse_error(e, line)
        else:
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
                resp = handle(req)
            except json.JSONDecodeError as e:
                resp = _parse_error(e)
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
