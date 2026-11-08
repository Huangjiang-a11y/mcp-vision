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
        # 优先读 meta.json 里的 opened_at，而不是文件的 mtime。
        # mtime 会被任何一次写 meta 的操作刷新（改 cursor 之类），
        # 和「用户什么时候打开过」不是一回事。
        try:
            at = json.loads((d / "meta.json").read_text()).get("opened_at") or ""
            if at:
                return time.mktime(time.strptime(at, "%Y-%m-%dT%H:%M:%S"))
        except Exception:
            pass
        try:
            return (d / "meta.json").stat().st_mtime
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


def _dump(obj, limit=0, compact=False):
    """序列化。默认**不裁剪**：调用方是模型，宁可长也别悄悄少一半。

    compact=True 时不缩进。纯数值数组（音频峰值轨迹那种）缩进后
    每个 [t, f] 要占 6 行，字符翻一倍多，而缩进对它毫无可读性收益。
    limit>0 时仍保留旧的逐半裁剪逻辑，应急用。
    """
    ind = None if compact else 2
    txt = json.dumps(obj, ensure_ascii=False, indent=ind)
    if limit <= 0 or len(txt) <= limit:
        return txt
    import copy
    o = copy.deepcopy(obj)
    for key in ("segments", "results", "timeline", "cues", "frames"):
        v = o.get(key)
        if isinstance(v, list) and v:
            o.setdefault("_truncated", {})[key] = len(v)
    while len(json.dumps(o, ensure_ascii=False, indent=ind)) > limit:
        cand = [k for k in o if isinstance(o[k], list) and len(o[k]) > 1]
        if not cand:
            return json.dumps({"_truncated": True, "note": "结果过大"}, ensure_ascii=False, indent=ind)
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


def _window(a, asset, default_span=None, cap=None, single=False):
    """统一的取窗：先定时间戳 time，再定看多久 seconds。

    - time：从哪开始（默认 0），负数按 0 处理
    - seconds：看多久；不给则看满 default_span 或上限
    - single：只要一个时间点。此时结尾根本不参与取图，
      再报「已到视频结尾」就是误导——用户只要 t=100 那一帧，
      却说「到结尾了」，等于凭空制造一个假的越界警告。
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

    # time 已经越过结尾：什么都没有可看，直接回开头。
    # 这里要提前返回——否则会先算出 end>dur 又补一条「已到视频结尾」，
    # 两条 note 一起出现自相矛盾（都回退到 0 了，还说看到了结尾）。
    if dur > 0 and start >= dur:
        notes.append("time 超出时长 %.0f 秒，回退到 0" % dur)
        return 0.0, min(dur, cap), "；".join(notes)

    if single:
        # 单帧只用起点，seconds 和结尾都不参与取图。
        # 此处若还报「单次上限」「已到视频结尾」，都是在描述一个不存在的窗口。
        return start, start, "；".join(notes)

    if span > cap:
        span = cap
        notes.append("单次上限 %.0f 秒" % cap)

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


def _opened(asset):
    """打开成功后的统一收尾：报一句状态，并按保留数清理旧工作区。

    清理要放在这里，不能只定义 _prune_workspaces 就完事——
    它之前一直是死代码（定义在 line 40，全项目零调用），
    于是 MVV_KEEP_WORKSPACES=5 从来没有生效过，目录只增不减。
    """
    gone = _prune_workspaces(engine.workspace_root(Path(__file__).resolve().parent))
    msg = ("已打开 %s\n时长 %.2fs  %sx%s  音频=%s" % (
        asset["title"], asset["duration"], asset["width"], asset["height"],
        "有" if asset["has_audio"] else "无"))
    if gone:
        msg += "\n（按 MVV_KEEP_WORKSPACES=%d 清理了 %d 个旧工作区：%s）" % (
            KEEP_WORKSPACES, len(gone), "、".join(gone))
    return msg


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
        return _opened(asset), []
    src = _normalize_source(src)
    cwd = Path(__file__).resolve().parent
    # 本地文件
    pth = Path(src)
    if not pth.is_absolute():
        pth = (Path(cwd) / pth).resolve()
    if pth.exists():
        asset = engine.open_local(cwd, str(pth))
        return _opened(asset), []
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
    return _opened(asset), []


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
        "range": [round(start, 3), round(start + d["duration"], 3)],
        "duration": d["duration"],
        "sample_rate": d["sample_rate"],
        "channels": d["channels"],
        "mono": d["mono"],
        "fft_size": d["fft_size"],
        "resolution_hz": d["resolution_hz"],
        "period_ms": d["period_ms"],
        "left": _chan(d["left"]),
    }
    # 单声道时 right 和 left 逐字节相同（note 里也这么说），
    # 两条 peak_freq_track 就白占 1000 多字符，别发两份。
    if not d["mono"]:
        summary["right"] = _chan(d["right"])
    notes = []
    if d["mono"]:
        notes.append("单声道音轨，只有 left")
    if note:
        notes.append(note)
    if notes:
        summary["note"] = "；".join(notes)

    images = []
    try:
        out_dir = Path(asset.get("audio_dir") or (Path(asset["media_path"]).parent / "audio"))
        stem = "a-%d-%d" % (round(start * 1000), round(end * 1000))
        paths = audiochart.render(d, views, out_dir, stem=stem)
        for p in paths.values():
            im = _attach_png(p)
            if im:
                images.append(im)
    except Exception as e:
        summary["image_error"] = "%s: %s" % (type(e).__name__, e)
    return (_dump(summary, compact=True), images)


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
        segs, _dur = engine.transcribe(asset["media_path"], start, end, language="auto")
    except RuntimeError as e:
        return (_dump({"ok": False, "mode": "stt", "error": str(e)}), [])
    cues = []
    for x in segs:
        c = {"from": _tc(x["s"]), "to": _tc(x["e"]), "text": x["text"]}
        # SenseVoice 同一次前向给出的情感 / 音频事件 / 语种，非默认才写。
        for k in ("emotion", "event", "lang"):
            if x.get(k):
                c[k] = x[k]
        cues.append(c)
    # 不写 engine（stt 只有一种引擎，属已知事实，见 schema）
    # 不写 range（就是调用方传的 time/seconds；真被夹了，note 会说）
    # 不写 text（等于把 segments 的文本再拼一遍，实测占响应 13%）
    out = {"mode": "stt", "segments": cues}
    # 汇总：一眼看出这首歌有没有 BGM、几处在笑/鼓掌、语种混没混。
    agg = {}
    for k in ("emotion", "event", "lang"):
        vals = [c[k] for c in cues if c.get(k)]
        if vals:
            agg[k] = {v: vals.count(v) for v in sorted(set(vals), key=vals.count, reverse=True)}
    if agg:
        out["summary"] = agg
    # 「event 是什么、不代表声道分离」属于字段语义，写进 schema 一次就够，
    # 不占每次响应。这里只在真的掉字时提醒一句。
    #
    # 两条 note 要合并，不能各写各的 out["note"]——
    # 之前就是 window_note 那条在后面无条件覆盖，BGM 提示被吞了。
    notes = []
    if window_note:
        notes.append(window_note)
    if any(c.get("event") == "BGM" for c in cues):
        notes.append("带 BGM 的段转写可能掉字")
    if notes:
        out["note"] = "；".join(notes)
    return (_dump(out), [])


def _tc(sec):
    """秒 -> MM:SS。

    只到秒。ms 位对定位没用（没人会拿它去切帧），却每段多花 4 个 token，
    97 段字幕就是 388 个——省下来更值。原始秒数在 seconds / s·e 字段里另有。
    """
    t = max(0.0, float(sec))
    t = int(math.floor(t + 0.5))
    return "%02d:%02d" % (t // 60, t % 60)


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
    # band_auto 恒等于 band（自动检测时两者同源，显式指定时它必为 None），零额外信息
    out = {"mode": "ocr", "frames": res["frames"], "ocr_runs": res["ocr_runs"],
           "skipped": res["skipped"], "band": res.get("band"),
           "band_px": res["band_px"], "segments": segs}
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
         "Samples `quantity` frames across the window into one page, "
         "plus a timeline mapping each cell to its moment. quantity defaults to 15, max 30.\n"
         "quantity=1 returns one frame at full detail instead of a page - use it to read a board, "
         "a scoreboard or a subtitle line, optionally with `crop` (WxH+X+Y, applied before scaling).\n"
         "\n"
         "`quality` is an area cap that only scales down. The window is capped at 900 seconds "
         "and clamped to the clip, with a note when that happens."
     ),     "inputSchema": {"type": "object", "properties": {
         "video_id": {"type": "string", "description": "Folder name under workspace/; default: most recently opened"},
         "time": {"type": "number", "description": "window start in seconds, default 0"},
         "seconds": {"type": "number",
                     "description": "window length, must be > 0; default: to the clip end, capped at 900"},
         "quantity": {"type": "integer",
                      "description": "frames to sample, default 15, max 30; 1 returns a single full-detail frame instead of a page"},
         "pick": {"type": "string", "enum": ["scene", "even"],
                  "description": ("scene (default) takes the frames where the picture changes most, so a sheet "
                                  "shows distinct content; even spaces them by time, for a sense of pacing")},
         "crop": {"type": "string",
                  "description": "crop before scaling, WxH+X+Y (e.g. 720x300+0+880); applies to every cell"},
         "quality": {"type": "string", "enum": ["low", "medium", "high", "max"],
                     "description": "area cap, scales down only; see enum for the values"}}}},

    {"name": "audio",
     "description": (
         "Measure a window's audio. Returns pictures plus numbers: loudness curve, FFT "
         "spectrum (peak frequency / note / cents), time-frequency map, pitch track, "
         "waveform. Sample rate follows the video's audio track. L and R are measured "
         "separately, L on the upper half of each picture, R on the lower. view picks "
         "which picture; the numbers always come back for both channels."
     ),
     "inputSchema": {"type": "object", "properties": {
         "video_id": {"type": "string", "description": "Folder name under workspace/; default: most recently opened"},
         "time": {"type": "number", "description": "window start in seconds, default 0"},
         "seconds": {"type": "number", "description": "window length in seconds, must be > 0; default 10, max 15"},
         "view": {"type": "string",
                  "enum": ["all", "level", "spectrum", "spectrogram", "peak", "wave"],
                  "description": ("level = loudness over time; spectrum = frequency content at one frame; "
                                  "spectrogram = frequency over time; peak = pitch track; wave = raw samples. "
                                  "Which to pick: how loud -> level, what notes -> spectrum or peak, how it changes -> spectrogram. Default all")}}}},

    {"name": "subtitles",
     "description": (
         "Get a window's text, as cues (from/to/text). mode=ocr (default) reads on-screen "
         "subtitles by OCRing the subtitle band frame by frame - use it for burned-in subtitles, "
         "songs and any text on screen. mode=stt transcribes the audio with speech recognition - "
         "use it when the speech is not written on screen. "
         "ONLY mode=stt carries acoustic fields: per-cue emotion, "
         "audio event (BGM = music "
         "outweighed speech, so words may be missing), language, and a summary count. "
         "mode=ocr reads pixels and cannot see them; it returns a per-cue conf instead. "
         
         
         "band=\"0.10,0.27\" pins the OCR band manually. "
         "Cue times are MM:SS (second precision)."
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
    """按 quality 档位给**整页**定像素预算，再平均分给每个格子。

    设 cell 短边 = s，整页 = (s·cols·aspect) × (s·rows)（宽屏），
    面积 s²·aspect·cols·rows = cap，即 s = sqrt(cap/(aspect·cols·rows))。

    注意这个 s 是**单格**短边，因为已经除过 cols·rows 了。
    以前代码注释写的是「短边」，实现也当整页短边用，于是竖屏拼页
    3.31MP 是预算 1.05MP 的 316%——横屏恰好 100%，所以一直没发现。
    """
    cap = _max_pixels(a)
    # 预算按**整页**算：单格面积 = cap/(cols·rows)，且单格宽高比 = 画面比例。
    #   h² · aspect = cap/(cols·rows)  →  h = sqrt(cap/(cols·rows·aspect))
    #
    # 之前把 sqrt(cap/(aspect·cols·rows)) 当成"短边"用，公式里已经除过 cols·rows，
    # 得到的其实是单格窄边；竖屏再拿它当页面宽，于是整页 3.31MP = 预算的 316%。
    # 横屏恰好 100%，所以这个偏差一直没暴露——只有竖屏才看得出来。
    per_cell = cap / max(1, cols * rows)
    ch = math.sqrt(per_cell / max(1e-6, aspect))
    cw = ch * aspect
    if aspect < 1:
        cw, ch = cw, ch
    # 单格尺寸兜底，避免极小窗口算出 0
    cw = max(48, min(cw, 1280))
    ch = max(48, min(ch, 1280))
    return max(1, int(round(cw))), max(1, int(round(ch)))


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


def _fill(anchors, n, start, end):
    """锚点不够 n 个时补点：在锚点之间的空隙里按长度比例均匀插值。

    补出来的点会贴着既有变化点分布——变化密集的地方补得少，
    空白的地方补得多，而不是无视内容硬套等分。
    """
    anchors = sorted(set(anchors))
    need = n - len(anchors)
    if need <= 0:
        return anchors[:n]
    edges = [start] + anchors + [end]
    gaps = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    gaps = [g for g in gaps if g[1] - g[0] > 1e-6]
    if not gaps:
        return anchors
    total = sum(b - a for a, b in gaps)
    # 按长度比例分配，最大余数法保证总数正好是 need
    share = [need * (b - a) / total for a, b in gaps]
    alloc = [int(x) for x in share]
    for i in sorted(range(len(gaps)), key=lambda i: share[i] - alloc[i], reverse=True)[:need - sum(alloc)]:
        alloc[i] += 1
    out = list(anchors)
    for (a, b), k in zip(gaps, alloc):
        for j in range(1, k + 1):
            out.append(round(a + (b - a) * j / (k + 1), 3))
    return sorted(set(out))[:n]


def _pick_scene(scores, n, start, end):
    """按画面变化取 n 个时间点。返回 (times, 补齐了几个)。

    1. 分数从高到低取，够 n 个就停
    2. 取一个就把它 ±min_gap 内的候选划掉——
       同一瞬间的 20 帧分数都高，全取进来等于浪费格子
    3. 有效变化点不足 n 个，剩下的按间隔均匀插值补齐
    """
    if n <= 1:
        return [float(start)], 0
    span = max(1e-6, end - start)
    min_gap = span / n
    cand = sorted(scores, key=lambda x: -x[1])
    # 相对下限：分数比最高分小两个量级的，是压缩抖动一类的噪声，不是"变化"。
    # 不设这条的话 NMS 会把满屏噪声点全选进来——它们分数虽低，但离已选点远，
    # 于是永远能凑满 n 个，补齐分支永远不触发，警告也就永远不会出现。
    # 用相对值而非固定阈值：不同片子、不同编码的动态范围差很多。
    if cand:
        floor = cand[0][1] * 0.05
        cand = [x for x in cand if x[1] >= floor]
    picked = []
    for t, _sc in cand:
        if all(abs(t - p) >= min_gap for p in picked):
            picked.append(t)
            if len(picked) >= n:
                break
    if len(picked) >= n:
        return sorted(picked)[:n], 0
    got = len(picked)
    return _fill(picked, n, start, end), n - got


def _scene_times(a, asset, start, end, n, timeout=None):
    """取 n 个时间点：默认按画面变化挑，pick=even 则老办法等分。

    返回 (times, 补齐数, 退化原因)。扫描结果按窗口缓存，
    同一段重复调用不再解一遍码。
    """
    if str(a.get("pick") or "scene").lower() == "even":
        return engine.sample_times(start, end, n), 0, None
    media = asset.get("media_path")
    key = "%s|%.3f|%.3f" % (media, start, end)
    cache = None
    try:
        cache = Path(asset.get("video_dir") or ".") / (".scene-%s.json" % _short_hash(key))
        if cache.exists():
            scores = [(float(t), float(sc)) for t, sc in json.loads(cache.read_text())]
            ts, filled = _pick_scene(scores, n, start, end)
            return ts, filled, None
    except Exception:
        cache = None
    try:
        scores = engine.scene_scores(media, start, end, timeout=timeout)
    except Exception as e:
        # 退回等分，但要说出来——否则用户以为看到的是变化点，其实是等分
        return engine.sample_times(start, end, n), 0, "画面扫描失败（%s），已退回按时间等分" % e
    if not scores:
        return engine.sample_times(start, end, n), 0, "窗口内没扫到帧，已退回按时间等分"
    ts, filled = _pick_scene(scores, n, start, end)
    if cache is not None:
        try:
            # 只存两位小数，8279 帧的 json 约 120KB，够用且省
            cache.write_text(json.dumps([[round(t, 3), round(sc, 4)] for t, sc in scores]))
        except Exception:
            pass
    return ts, filled, None


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
    win_start, win_end, note = _window(a, asset, single=(n == 1))
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
        # 路径只留文件名：图片本身走 image 内容块，模型读不了本地路径。
        # 绝对路径前缀（workspace/<标题>/video/）每处 55 字符，纯重复。
        # 只留模型用得上的：这一帧在几秒、片长多少、有没有被截断。
        # ok/mode/title 是回显或恒定值，size 看图就知道，file 它打不开。
        out = {"t": f["seconds"], "duration": round(duration, 2)}
        if note:
            out["note"] = note
        return (json.dumps(out, ensure_ascii=False, indent=2), _attach(f["path"]))

    # 多帧：拼成一页
    cols, rows = _pick_grid(n, aspect)
    cell_w, cell_h = _sheet_cells(a, aspect, cols, rows)
    ts, filled, degraded = _scene_times(a, asset, win_start, win_end, n)
    frames = engine.extract_frames(asset["media_path"], ts, out_dir=out_dir, duration=duration,
                                   size=(cell_w, cell_h), crop=crop)
    if not frames:
        return ("抽帧失败（%d 个时间点都没取到）" % len(ts), [])
    # 缓存键必须含实际取到的帧时间点。
    # 以前等分采样时 ts 完全由 (win_start, win_end, n) 决定，键里不写也对；
    # 现在按画面变化取，同样窗口两次调用可能取到完全不同的帧，
    # 不把 ts 写进键就等于拿旧图冒充新结果。
    ts_key = ",".join("%.3f" % t for t in ts)
    key = "%s|%d|%s|%s|%s|%dx%d|%dx%d|%s" % (asset.get("id") or "", n, a.get("quality"),
                                              note, spec, cols, rows, cell_w, cell_h, ts_key)
    sheet_path, jpg = _compose(a, asset, frames, cols, rows, cell_w, cell_h, key, "sheet", win_start)
    # 只留模型用得上的：片长、实际窗口、每格时间点、提示。
    # 删掉的几类——
    #   ok        永远是 true，失败走另一条返回路径
    #   mode      由 quantity 决定，调用方自己知道
    #   title     回显它刚传的 video_id
    #   frames    = len(timeline)
    #   grid/cellSize  排版细节，图上看得出来
    #   crop/quality   回显入参
    #   file      图片就在 image 块里，它没有文件系统
    #   sheetKB   它没法据此做决定
    out = {"duration": round(duration, 2),
           "range": [round(min(ts), 3), round(max(ts), 3)],
           "timeline": [f["seconds"] for f in frames]}
    for msg in ((degraded or ""),
                ("窗口里只找到 %d 处明显画面变化，其余 %d 格是在变化点之间插值补的，"
                 "不代表新的画面" % (n - filled, filled)) if filled else ""):
        if msg:
            note = msg if not note else note + "；" + msg
    if note:
        out["note"] = note
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
