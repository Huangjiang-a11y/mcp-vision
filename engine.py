"""mcp-vision 引擎：probe / 抽帧 / 网格页 / 音频分析 / 工作区。纯 Python，无 Node。"""
import os, re, json, subprocess, time, math, struct, zlib, hashlib, shutil, urllib.request
from pathlib import Path
import numpy as np

WORKSPACE_DIRNAME = "workspace"
CACHE_DIRNAME = WORKSPACE_DIRNAME  # 兼容别名
DEFAULT_FRAME_WIDTH = 1280
MAX_FRAMES = 12


def _run(args, timeout=300, cwd=None):
    p = subprocess.run(args, capture_output=True, text=True, timeout=timeout, cwd=cwd)
    if p.returncode != 0:
        raise RuntimeError((p.stderr or p.stdout or "command failed")[:2000])
    return p.stdout


def timecode(seconds):
    s = max(0.0, float(seconds))
    return "%02d:%06.3f" % (int(s // 60), s % 60)


def sample_times(start, end, count):
    if count <= 1:
        return [float(start)]
    return [round(start + (end - start) * i / (count - 1), 3) for i in range(count)]


# ---------- probe ----------
def probe(path):
    out = _run(["ffprobe", "-v", "error", "-show_entries",
                "format=duration:stream=codec_type,width,height,avg_frame_rate,duration,start_time",
                "-of", "json", str(path)], timeout=60)
    info = json.loads(out)
    streams = info.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not v:
        raise RuntimeError("no video stream")
    def _dur(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None
    dur_candidates = [_dur(v.get("duration")), _dur(info.get("format", {}).get("duration"))]
    duration = next((d for d in dur_candidates if d and d > 0), None)
    if not duration:
        raise RuntimeError("no duration")
    # tail frame to refine duration
    try:
        origin = float(v.get("start_time") or 0)
        tail = json.loads(_run(["ffprobe", "-v", "error", "-select_streams", "v:0",
            "-read_intervals", "%s%%" % (origin + max(0, duration - 1)),
            "-show_entries", "frame=best_effort_timestamp_time,duration_time,pkt_duration_time",
            "-of", "json", str(path)], timeout=60))
        last = None
        for f in tail.get("frames", []):
            pos = float(f.get("best_effort_timestamp_time", 0)) - origin
            if pos < 0:
                continue
            last = max(last or 0, pos)
            length = f.get("duration_time") or f.get("pkt_duration_time")
            try:
                length = float(length)
                duration = max(duration, pos + (length if length > 0 else 0))
            except (TypeError, ValueError):
                pass
    except Exception:
        pass
    fps_str = v.get("avg_frame_rate", "0/1")
    try:
        a, b = fps_str.split("/")
        fps = float(a) / float(b) if float(b) else 0.0
    except Exception:
        fps = 0.0
    return {
        "duration": round(duration, 6),
        "width": int(v["width"]),
        "height": int(v["height"]),
        "fps": round(fps, 3),
        "has_audio": any(s.get("codec_type") == "audio" for s in streams),
        "timeline_start": float(v.get("start_time") or 0),
    }


# ---------- frame extraction ----------
def _extract_jpeg(media, t, width, out_path, timeout=120):
    _run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
          "-ss", str(t), "-i", str(media), "-frames:v", "1", "-an",
          "-vf", "scale=w='min(%d,iw)':h=-2:out_range=full" % width,
          "-pix_fmt", "yuvj420p", "-q:v", "2", str(out_path)], timeout=timeout)


def extract_frames(media, times, width=DEFAULT_FRAME_WIDTH, out_dir=None, signal=None, duration=None):
    media = Path(media)
    if out_dir is None:
        out_dir = media.parent / (media.stem + "_frames")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    media_mtime = media.stat().st_mtime
    result = []
    seen = set()
    for t in times:
        if signal is not None and getattr(signal, "aborted", False):
            raise RuntimeError("aborted")
        t = float(t)
        if duration is not None and t >= duration:
            t = max(0.0, duration - 0.05)
        if t < 0:
            t = 0.0
        path = out_dir / ("t%09d-w%d.jpg" % (round(t * 1000), width))
        # reuse if fresh
        if not (path.exists() and path.stat().st_size > 0 and path.stat().st_mtime >= media_mtime):
            ok = False
            for cand in (t, t-0.1, t-0.2, t-0.5):
                if cand < 0:
                    continue
                try:
                    _extract_jpeg(media, cand, width, path)
                    if path.exists() and path.stat().st_size > 0:
                        t = cand
                        ok = True
                        break
                except Exception:
                    continue
            if not ok:
                continue
        if not path.exists() or path.stat().st_size == 0:
            continue
        if t in seen:
            continue
        seen.add(t)
        result.append({"seconds": round(t, 3), "timestamp": timecode(t), "path": str(path)})
    return result


# ---------- png writer (zlib, no Pillow) ----------
def _write_png(path, rgb, w, h):
    # rgb: numpy array shape (h,w,3) uint8
    def chunk(typ, data):
        c = struct.pack(">I", len(data)) + typ + data
        crc = zlib.crc32(typ + data) & 0xffffffff
        return struct.pack(">I", len(data)) + typ + data + struct.pack(">I", crc)
    # proper chunk assembly
    def mk(typ, data):
        return struct.pack(">I", len(data)) + typ + data + struct.pack(">I", zlib.crc32(typ + data) & 0xffffffff)
    raw = bytearray()
    for y in range(h):
        raw.append(0)  # filter 0
        raw.extend(rgb[y].tobytes())
    compressed = zlib.compress(bytes(raw), 9)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(mk(b"IHDR", ihdr))
        f.write(mk(b"IDAT", compressed))
        f.write(mk(b"IEND", b""))


def _load_raw_rgb(media, t, w, h, timeout=120):
    # decode one frame to raw rgb24 at target size
    out = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
         "-ss", str(t), "-i", str(media), "-frames:v", "1", "-an",
         "-vf", "scale=%d:%d:force_original_aspect_ratio=decrease,pad=%d:%d:(ow-iw)/2:(oh-ih)/2" % (w, h, w, h),
         "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
        capture_output=True, timeout=timeout)
    if out.returncode != 0:
        raise RuntimeError((out.stderr or b"").decode("utf-8", "replace")[:500])
    buf = out.stdout
    expected = w * h * 3
    if len(buf) < expected:
        # pad with black
        buf = buf + b"\x00" * (expected - len(buf))
    arr = np.frombuffer(buf[:expected], dtype=np.uint8).reshape(h, w, 3)
    return arr


def compose_sheet(media, times, cols, rows, cell_w, cell_h, out_path, title=None):
    canvas_w = cols * cell_w
    canvas_h = rows * cell_h
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:] = (24, 24, 28)
    for idx, t in enumerate(times):
        if idx >= cols * rows:
            break
        r, c = divmod(idx, cols)
        try:
            cell = _load_raw_rgb(media, t, cell_w, cell_h)
            canvas[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w] = cell
        except Exception:
            pass
    _write_png(out_path, canvas, canvas_w, canvas_h)
    return str(out_path)


# ---------- audio analysis ----------
def _extract_pcm(media, start, end, out_path, timeout=300):
    _run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
          "-ss", str(start), "-i", str(media), "-t", str(end - start),
          "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", str(out_path)],
         timeout=timeout)


def analyze_audio(media, start, end, out_dir=None):
    media = Path(media)
    if out_dir is None:
        out_dir = media.parent / (media.stem + "_audio")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "a-%d-%d" % (round(start * 1000), round(end * 1000))
    pcm_path = out_dir / (stem + ".pcm")
    wav_path = out_dir / (stem + ".wav")
    _extract_pcm(media, start, end, pcm_path)
    pcm = open(pcm_path, "rb").read()
    if not pcm or len(pcm) % 2:
        raise RuntimeError("no complete pcm samples")
    samples = len(pcm) // 2
    arr = np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768.0
    sr = 16000
    hop = int(sr * 0.02)
    nframes = (samples + hop - 1) // hop
    rms = np.zeros(nframes)
    peak = 0.0
    for i in range(nframes):
        sl = arr[i*hop:(i+1)*hop]
        if sl.size == 0:
            continue
        rms[i] = np.sqrt(np.mean(sl * sl))
        peak = max(peak, float(np.max(np.abs(sl))))
    def db(x):
        return round(20 * math.log10(max(x, 1e-6)), 3)
    duration = samples / sr
    step = hop / sr
    # quiet ranges
    pauses = []
    qstart = None
    for i in range(nframes):
        if db(rms[i]) < -45:
            if qstart is None:
                qstart = i
        else:
            if qstart is not None and (i - qstart) * step >= 0.3:
                pauses.append({"start_seconds": round(start + qstart * step, 3),
                               "end_seconds": round(start + i * step, 3)})
            qstart = None
    if qstart is not None and (duration - qstart * step) >= 0.3:
        pauses.append({"start_seconds": round(start + qstart * step, 3),
                       "end_seconds": round(start + duration, 3)})
    # levels (every 50 frames)
    levels = []
    for i in range(0, nframes, 50):
        sl = rms[i:i+50]
        if sl.size == 0:
            continue
        v = np.sqrt(np.mean(sl * sl))
        levels.append({"seconds": round(start + i * step, 3), "rms_dbfs": db(v)})
    avg_rms = float(np.mean(rms)) if nframes else 0.0
    modulation = (float(np.std(rms)) / max(avg_rms, 1e-6)) if nframes else 0.0
    novelty = np.maximum(0, np.diff(rms, prepend=rms[0]))
    mean_n = float(np.mean(novelty)) if nframes else 0.0
    dev_n = float(np.std(novelty)) if nframes else 0.0
    attacks = []
    for i in range(2, max(2, nframes - 2)):
        v = float(novelty[i])
        if v <= max(0.001, avg_rms * 0.08, mean_n + dev_n * 0.5):
            continue
        window = novelty[max(0, i-2):i+3]
        if v < float(np.max(window)):
            continue
        ev = {"seconds": round(start + i * step, 3), "strength": round(v, 6)}
        if attacks and ev["seconds"] - attacks[-1]["seconds"] < 0.12:
            if v > attacks[-1]["strength"]:
                attacks[-1] = ev
        else:
            attacks.append(ev)
    # tempo via autocorrelation on novelty
    candidates = []
    lags = range(max(1, int(60/(180*step))), int(60/(60*step)) + 1)
    for lag in lags:
        a = novelty[lag:] - mean_n
        b = novelty[:-lag] - mean_n if lag > 0 else novelty - mean_n
        # align lengths
        n = min(len(a), len(b))
        if n <= 0:
            continue
        a = a[:n]; b = b[:n]
        denom = math.sqrt(float(np.sum(a*a)) * float(np.sum(b*b)))
        score = float(np.sum(a*b)) / denom if denom > 0 else 0.0
        if lag < len(correlations := []) :
            pass
        candidates.append((lag, score))
    peaks = []
    for i, (lag, score) in enumerate(candidates):
        if score > 0.15 and score >= (candidates[i-1][1] if i > 0 else 0) and score >= (candidates[i+1][1] if i < len(candidates)-1 else 0):
            peaks.append({"bpm": round(60/(lag*step), 1), "periodicity": round(score, 3)})
    peaks.sort(key=lambda x: (-x["periodicity"], abs(x["bpm"] - 120)))
    peaks = peaks[:3]
    silent = db(float(np.sqrt(np.mean(arr*arr)))) < -60 if samples else True
    periodic = (not silent and modulation >= 0.08 and duration >= 8 and
                len(attacks) >= 8 and bool(peaks) and peaks[0]["periodicity"] >= 0.2)
    strongest = sorted(attacks, key=lambda x: -x["strength"])[:32]
    strongest.sort(key=lambda x: x["seconds"])
    # write wav
    import wave
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(pcm)
    return {
        "state": "available",
        "source_kind": "embedded",
        "clip_path": str(wav_path),
        "analyzed_range_seconds": [round(start, 3), round(start + duration, 3)],
        "next_start_seconds": (start + duration) if (start + duration) < (start + (end - start)) else None,
        "analysis": {
            "source": "decoded-audio-signal",
            "method": "20ms RMS envelope and onset autocorrelation",
            "start_seconds": round(start, 3),
            "end_seconds": round(start + duration, 3),
            "duration_seconds": round(duration, 3),
            "sample_rate": sr,
            "rms_dbfs": db(float(np.sqrt(np.mean(arr * arr)))),
            "peak_dbfs": db(peak),
            "silent": bool(silent),
            "quiet_range_count": len(pauses),
            "onset_count": len(attacks),
            "tempo": {
                "state": "estimated" if periodic else "no-stable-pulse",
                "bpm": (peaks[0]["bpm"] if (periodic and peaks) else None),
                "candidates": peaks if periodic else [],
                "timing_resolution_seconds": step,
                "interpretation": "Pulse estimates and audio attacks for editing; compare with frames.",
            },
            "level_timeline": levels,
            "quiet_ranges": pauses[:32],
            "prominent_onsets_seconds": [a["seconds"] for a in strongest],
        },
    }


# ---------- cache ----------
def _sanitize_title(name):
    # 保留中文/字母/数字/常见标点，去掉路径分隔与控制字符
    bad = set('/\\:*?"<>|\x00')
    out = "".join(c for c in (name or "") if c not in bad).strip(" .")
    return out or "video"


def workspace_root(cwd):
    return Path(cwd) / WORKSPACE_DIRNAME


def cache_root(cwd):
    return workspace_root(cwd)


def status(cwd):
    root = workspace_root(cwd)
    if not root.exists():
        return {"workspace": str(root), "videos": [], "bytes": 0}
    videos = []
    total = 0
    for d in root.iterdir():
        if not d.is_dir():
            continue
        meta = d / "meta.json"
        if meta.exists():
            try:
                m = json.loads(meta.read_text())
                videos.append({"id": m.get("id"), "title": m.get("title"),
                               "duration": m.get("duration"), "path": m.get("media_path")})
            except Exception:
                pass
        for pp in d.rglob("*"):
            if pp.is_file():
                total += pp.stat().st_size
    return {"workspace": str(root), "videos": videos, "bytes": total}


def open_local(cwd, path):
    p = Path(path)
    if not p.is_absolute():
        p = (Path(cwd) / p).resolve()
    if not p.exists():
        raise FileNotFoundError(str(p))
    info = probe(p)
    title = _sanitize_title(p.stem)
    folder = workspace_root(cwd) / title
    folder.mkdir(parents=True, exist_ok=True)
    for sub in ("video", "audio", "subtitles"):
        (folder / sub).mkdir(exist_ok=True)
    dest = folder / p.name
    if dest.resolve() != p.resolve() and not dest.exists():
        shutil.copy2(p, dest)
    media_path = str(dest if dest.exists() else p)
    asset = {"id": title, "source": "local", "title": p.name, "source_url": None,
             "media_path": media_path, "workspace": str(folder),
             "video_dir": str(folder / "video"),
             "audio_dir": str(folder / "audio"),
             "subtitles_dir": str(folder / "subtitles"),
             "cursor": 0, "opened_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
             **info}
    (folder / "meta.json").write_text(json.dumps(asset, ensure_ascii=False, indent=2))
    (workspace_root(cwd) / "latest.txt").write_text(media_path)
    return asset


def load_latest(cwd):
    lp = workspace_root(cwd) / "latest.txt"
    if not lp.exists():
        return None
    try:
        path = Path(lp.read_text().strip())
    except Exception:
        return None
    if not path.exists():
        return None
    meta = path.parent / "meta.json"
    if meta.exists():
        try:
            return json.loads(meta.read_text())
        except Exception:
            pass
    info = probe(path)
    return {"id": path.parent.name, "title": path.name, "source": "local",
            "media_path": str(path), "workspace": str(path.parent),
            "video_dir": str(path.parent / "video"),
            "audio_dir": str(path.parent / "audio"),
            "subtitles_dir": str(path.parent / "subtitles"),
            **info}


# ---------- STT：silero-vad 自动断句 + SenseVoice 识别 ----------
_ASR_CACHE = {}

# 转写用的三个模型路径，全部由环境变量给出，文件名不限：
#   MVV_ASR_MODEL   SenseVoice 的 onnx
#   MVV_ASR_TOKENS  对应的 tokens.txt
#   MVV_VAD_MODEL   silero VAD 的 onnx
ASR_ENV = ("MVV_ASR_MODEL", "MVV_ASR_TOKENS", "MVV_VAD_MODEL")


def asr_models():
    """返回 (model, tokens, vad) 的绝对路径。"""
    if "paths" in _ASR_CACHE:
        return _ASR_CACHE["paths"]
    vals = []
    for name in ASR_ENV:
        v = (os.environ.get(name) or "").strip()
        if not v:
            raise RuntimeError("未设置 %s。转写需要三个模型路径，见 INSTALL.md" % name)
        p = Path(v).expanduser()
        if not p.is_file():
            raise RuntimeError("%s 指向的文件不存在：%s" % (name, p))
        vals.append(str(p))
    _ASR_CACHE["paths"] = tuple(vals)
    return _ASR_CACHE["paths"]


def _asr_recognizer(threads=2, language="zh"):
    key = ("rec", threads, language)
    if key in _ASR_CACHE:
        return _ASR_CACHE[key]
    import sherpa_onnx
    model, tokens, _ = asr_models()
    rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=model, tokens=tokens, num_threads=int(threads),
        use_itn=True, language=language, debug=False)
    _ASR_CACHE[key] = rec
    return rec


def pcm16k(media, start=0.0, end=None, timeout=1800):
    """ffmpeg 直出 16k 单声道 float32，不落盘。"""
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
           "-ss", "%.3f" % max(0.0, float(start))]
    if end is not None:
        cmd += ["-t", "%.3f" % (float(end) - float(start))]
    cmd += ["-i", str(media), "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"]
    p = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if p.returncode != 0:
        err = (p.stderr or b"").decode("utf-8", "ignore")[:500]
        raise RuntimeError("ffmpeg 提取音频失败：%s" % err)
    if not p.stdout:
        raise RuntimeError("ffmpeg 没有输出音频（该视频可能没有音轨）")
    return np.frombuffer(p.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def _is_noise_only(text):
    t = re.sub(r"[，。！？、,.!?…~～\s]", "", text or "")
    if not t:
        return True
    if len(t) > 3:
        return False
    return re.fullmatch(r"[啊哦嗯呃唉哎哈嘿呀哇喔噢欸呢呐啦哟]+", t) is not None


def transcribe(media, start=0.0, end=None, language="zh", threads=2,
               threshold=0.4, min_silence=0.4, max_speech=20.0, min_speech=0.25,
               drop_noise=True):
    """按 VAD 自动断句，逐段识别。返回 [{s, e, text}]（秒，相对整片）。"""
    import sherpa_onnx
    samples = pcm16k(media, start, end)
    if samples.size == 0:
        return [], 0.0
    _, _, vad_path = asr_models()
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = vad_path
    cfg.silero_vad.threshold = float(threshold)
    cfg.silero_vad.min_silence_duration = float(min_silence)
    cfg.silero_vad.max_speech_duration = float(max_speech)
    cfg.silero_vad.min_speech_duration = float(min_speech)
    cfg.sample_rate = 16000
    cfg.num_threads = 1
    vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=int(max_speech + 30))
    rec = None  # 懒加载：VAD 真切出语音了再建识别器（229 MB，没人说话就别占）
    sr = 16000
    win = 512  # silero 硬性窗口
    base = float(start)

    out = []
    dropped = 0

    def _emit(seg):
        nonlocal dropped, rec
        audio = np.array(seg.samples, dtype=np.float32).reshape(-1)
        dur = len(audio) / float(sr)
        if dur < min_speech:
            return
        if rec is None:
            rec = _asr_recognizer(threads, language)
        s = rec.create_stream()
        s.accept_waveform(sr, audio)
        rec.decode_stream(s)
        text = re.sub(r"\s+", "", str(s.result.text or "")).strip()
        if not text:
            return
        if drop_noise and _is_noise_only(text):
            dropped += 1
            return
        a = base + seg.start / float(sr)
        out.append({"s": round(a, 2), "e": round(a + dur, 2), "text": text})

    for i in range(0, samples.size, win):
        vad.accept_waveform(samples[i:i + win])
        while not vad.empty():
            seg = vad.front
            _emit(seg)
            vad.pop()
    vad.flush()
    while not vad.empty():
        seg = vad.front
        _emit(seg)
        vad.pop()
    return out, samples.size / float(sr)
