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
def _extract_jpeg(media, t, width, out_path, timeout=120, size=None, crop=None):
    # size 显式指定宽高时直接给死，别再让 h=-2 自己取整，否则实际尺寸和报出去的对不上
    if size:
        vf = "scale=%d:%d:out_range=full" % (int(size[0]), int(size[1]))
    else:
        vf = "scale=w='min(%d,iw)':h=-2:out_range=full" % width
    # crop 走 ffmpeg 的 crop 滤镜，语法就是 WxH+X+Y；先裁再缩，比例不会变形
    if crop:
        vf = "%s,%s" % (crop, vf)
    _run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
          "-ss", str(t), "-i", str(media), "-frames:v", "1", "-an",
          "-vf", vf, "-pix_fmt", "yuvj420p", "-q:v", "2", str(out_path)], timeout=timeout)


def extract_frames(media, times, width=DEFAULT_FRAME_WIDTH, out_dir=None, signal=None, duration=None,
                   size=None, crop=None):
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
        # 不同裁剪区域可能裁出同样的宽高，文件名带上裁剪的短 hash 才不会互相覆盖
        csuf = "" if not crop else "-" + hashlib.md5(str(crop).encode()).hexdigest()[:4]
        if size:
            path = out_dir / ("t%09d-%dx%d%s.jpg" % (round(t * 1000), int(size[0]), int(size[1]), csuf))
        else:
            path = out_dir / ("t%09d-w%d%s.jpg" % (round(t * 1000), width, csuf))
        # reuse if fresh
        if not (path.exists() and path.stat().st_size > 0 and path.stat().st_mtime >= media_mtime):
            ok = False
            for cand in (t, t-0.1, t-0.2, t-0.5):
                if cand < 0:
                    continue
                try:
                    _extract_jpeg(media, cand, width, path, size=size, crop=crop)
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
    # decode one frame to raw rgb24 at target size；t=None 时直接把 media 当图片读，不 seek
    head = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
    if t is not None:
        head += ["-ss", str(t)]
    out = subprocess.run(
        head + ["-i", str(media), "-frames:v", "1", "-an",
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


def compose_sheet(media, times, cols, rows, cell_w, cell_h, out_path, title=None, frame_paths=None):
    """frame_paths 给了就直接读这些 jpg（抽帧时已经解码过一次），
    否则退回按 times 回原片逐帧解码 —— 后者会把每个时间点 seek 两遍。"""
    canvas_w = cols * cell_w
    canvas_h = rows * cell_h
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:] = (24, 24, 28)
    srcs = [(p, None) for p in (frame_paths or [])] or [(media, t) for t in times]
    for idx, (src, t) in enumerate(srcs):
        if idx >= cols * rows:
            break
        r, c = divmod(idx, cols)
        try:
            cell = _load_raw_rgb(src, t, cell_w, cell_h)
            canvas[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w] = cell
        except Exception:
            pass
    _write_png(out_path, canvas, canvas_w, canvas_h)
    return str(out_path)


# ---------- audio analysis ----------
# ---------- 音频频谱分析（level / spectrum / spectrogram）----------
_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def audio_info(media):
    """音轨原始采样率与声道数；无音轨返回空。"""
    out = _run(["ffprobe", "-v", "error", "-select_streams", "a:0",
                "-show_entries", "stream=sample_rate,channels",
                "-of", "json", str(media)], timeout=60)
    st = ((json.loads(out).get("streams")) or [{}])[0]
    try:
        sr = int(st.get("sample_rate") or 0)
    except (TypeError, ValueError):
        sr = 0
    try:
        ch = int(st.get("channels") or 0)
    except (TypeError, ValueError):
        ch = 0
    return {"sample_rate": sr, "channels": ch}


def freq_to_note(freq):
    """频率 -> (音名, 音分偏差)。A4=440，十二平均律。"""
    if not freq or freq <= 0:
        return "", 0.0
    midi = 69 + 12 * math.log2(float(freq) / 440.0)
    n = int(round(midi))
    cents = (midi - n) * 100.0
    return _NOTE_NAMES[n % 12] + str(n // 12 - 1), cents


def _peak_delta(mag, k):
    """峰值 bin 相对真实频率的偏移（-0.5..0.5），对数域抛物线插值。"""
    if k < 1 or k + 1 >= mag.size:
        return 0.0
    a, b, c = (20.0 * np.log10(np.maximum(mag[j], 1e-12)) for j in (k - 1, k, k + 1))
    denom = a - 2.0 * b + c
    if abs(denom) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5))


def _hann_gain(delta):
    """汉宁窗在偏离 bin 中心 delta 格时的幅度损失（0dB 处为 1）。

    |sin(πδ) / (πδ(1-δ²))|，δ=0.5 时约 0.849，正是扇贝损失。
    """
    d = abs(float(delta))
    if d < 1e-9:
        return 1.0
    g = math.sin(math.pi * d) / (math.pi * d * (1.0 - d * d))
    return max(abs(g), 1e-6)


def _peak_bin(mag):
    """幅度谱峰值：抛物线插值，返回 (小数 bin, 峰值幅度)。

    幅度按窗函数的扇贝损失修正过，纯音不论落在 bin 哪个位置都能报准。
    """
    if mag.size < 3:
        return 0, 0.0
    k = int(np.argmax(mag[1:])) + 1
    delta = _peak_delta(mag, k)
    return k + delta, float(mag[k]) / _hann_gain(delta)


def _channel_metrics(x, sr, start, size):
    """单个声道的样本 -> 电平 / 频谱 / 时频图。L、R 各算一遍，参数完全一致。"""
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        raise RuntimeError("窗口内没有音频样本")

    # 电平曲线：20ms 一帧的 RMS
    hop = max(1, int(sr * 0.02))
    nfr = x.size // hop
    if nfr < 1:
        nfr, hop = 1, x.size
    frames = x[: nfr * hop].reshape(nfr, hop).astype(np.float64)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    lsb = 1.0 / 32768.0          # 16bit 量化底噪，比这更小没有意义
    db = 20.0 * np.log10(np.maximum(rms, lsb))
    lt = float(start) + np.arange(nfr) * (hop / float(sr))
    silent = bool(float(np.max(np.abs(x))) < lsb * 2)
    peak_dbfs = 20.0 * math.log10(max(float(np.max(np.abs(x))), lsb))

    # 窗口比 FFT 还短（比如只切了 30ms）：补零到整帧，
    # 否则窗函数长度和 STFT 分块长度对不上，下面会直接崩
    if x.size < size:
        x = np.pad(x, (0, size - x.size))

    # 频谱：取最响的一帧做 FFT
    loud = int(np.argmax(rms)) * hop
    lo = max(0, min(x.size - size, loud - size // 2))
    # 原始波形：和上面这段 FFT 用的是同一处声音，40ms，方便对照着看
    wn = min(x.size, max(16, int(sr * 0.040)))
    w0 = max(0, min(x.size - wn, (lo + size // 2) - wn // 2))
    wseg = x[w0: w0 + wn].astype(np.float64)
    wave = {
        "t0": round(float(start) + w0 / float(sr), 6),
        "ms": round(1000.0 * wn / float(sr), 3),
        "samples": [round(float(v), 8) for v in wseg],
        "peak": round(float(np.max(np.abs(wseg))), 6) if wseg.size else 0.0,
        "rms": round(float(np.sqrt(np.mean(wseg ** 2))), 6) if wseg.size else 0.0,
    }
    seg = x[lo: lo + size].astype(np.float64)
    win = np.hanning(seg.size)
    spec = np.abs(np.fft.rfft(seg * win))
    freqs = np.fft.rfftfreq(seg.size, 1.0 / sr)
    k, _ = _peak_bin(spec)
    peak_hz = float(k * sr / seg.size)
    note, cents = freq_to_note(peak_hz)
    # 幅度谱取绝对值：满幅正弦 = 1（2|X|/sum(win)），不再按窗口峰值归一化，
    # 这样不同时间窗之间能横向比较，超过 1 也不奇怪（方波基波就有 1.27）
    wsum = float(np.sum(win))
    mag = 2.0 * spec / wsum
    # 峰值那格按扇贝损失修正：纯音落在两个 bin 正中间时单格读数会低 1.4dB。
    # 只修峰值一格——其余 bin 本来就只是采样点，不知道真实频率在哪。
    kf, _ = _peak_bin(spec)
    kb = int(round(kf))
    if 0 <= kb < mag.size:
        mag[kb] /= _hann_gain(kf - kb)
    mag_db = 20.0 * np.log10(np.maximum(mag, 1e-9))

    # 时频图：STFT
    # 87.5% 重叠：时频图在 1024 宽的画布上放大好几倍，
    # 帧太稀会看到一块块的小方格
    hop2 = max(1, size // 8)
    ncol = max(1, (x.size - size) // hop2 + 1) if x.size >= size else 1
    cols = np.empty((freqs.size, ncol), dtype=np.float32)
    for i in range(ncol):
        chunk = x[i * hop2: i * hop2 + size].astype(np.float64)
        if chunk.size < size:
            chunk = np.pad(chunk, (0, size - chunk.size))
        cols[:, i] = np.abs(np.fft.rfft(chunk * win))
    # 每帧峰值那一格补回扇贝损失：纯音落在 bin 中间时单格只有 0.849 倍，
    # 不补的话峰值轨迹和亮点会随音高移动忽明忽暗地跳 1.4dB。
    # 只动峰值一格——别的格本来就只是采样点，不知道真实频率在哪。
    for i in range(ncol):
        col = cols[:, i].astype(np.float64)
        if col.size < 3:
            continue
        k = int(np.argmax(col[1:])) + 1
        cols[k, i] /= _hann_gain(_peak_delta(col, k))
    amp = 2.0 * cols / wsum
    spec_db = 20.0 * np.log10(np.maximum(amp, 1e-9))   # dBFS，满幅正弦 = 0dB
    st = float(start) + np.arange(ncol) * (hop2 / float(sr))
    # 峰值轨迹：只在有说话的帧上取峰，静音段不跟噪底跑
    frame_db = 20.0 * np.log10(np.maximum(np.sqrt(np.mean(cols.astype(np.float64) ** 2, axis=0)), 1e-9))
    gate = max(-70.0, float(np.max(frame_db)) - 45.0) if frame_db.size else -70.0
    track = []
    for i in range(ncol):
        if frame_db[i] < gate:
            continue
        kk, _ = _peak_bin(cols[1:, i].astype(np.float64))
        f = float((kk + 1) * sr / size)
        if f > 0:
            track.append([round(float(st[i]), 3), round(f, 2)])

    return {
        "silent": silent,
        "level": {
            "db": round(float(db[-1]), 1),
            "min": round(float(np.min(db)), 1),
            "max": round(float(np.max(db)), 1),
            "mean": round(float(np.mean(db)), 1),
            "peak_dbfs": round(peak_dbfs, 1),
            "times": lt, "values": db,
        },
        "spectrum": {
            "freqs": freqs, "mag_db": mag_db,
            "peak_freq_hz": round(peak_hz, 2),
            "note": note, "cents": round(cents, 2),
        },
        "wave": wave,
        "spectrogram": {
            "spec_db": spec_db, "freqs": freqs, "times": st,
            "peak_track": track,
        },
    }


def analyze_spectrum(media, start, end, fft_size=4096, out_dir=None):
    """窗口内的音频剖析：L、R 分别算电平 / 频谱 / 时频图。

    一次解码出双声道（-ac 2），采样率跟视频音轨走，不重采样。
    单声道音轨会被 ffmpeg 复制成 L=R，结果里 mono=True 标出来。
    """
    media = Path(media)
    info = audio_info(media)
    sr = info["sample_rate"]
    if sr <= 0:
        raise RuntimeError("没有音轨")
    if out_dir is None:
        out_dir = media.parent / (media.stem + "_audio")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pcm_path = out_dir / ("s-%d-%d.pcm" % (round(start * 1000), round(end * 1000)))
    _run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
          "-ss", "%.3f" % float(start), "-i", str(media),
          "-t", "%.3f" % max(0.0, float(end) - float(start)),
          "-vn", "-ac", "2", "-ar", str(sr), "-f", "s16le", str(pcm_path)], timeout=300)
    raw = open(pcm_path, "rb").read()
    # 交错 L/R 16bit：先按样本对齐，再切出两路
    keep = len(raw) - (len(raw) % 4)
    if keep <= 0:
        raise RuntimeError("音频解码失败")
    arr = np.frombuffer(raw[:keep], dtype="<i2").astype(np.float32) / 32768.0
    pair = arr.reshape(-1, 2)
    nsmp = pair.shape[0]
    dur = nsmp / float(sr)
    size = int(min(max(64, fft_size), nsmp))

    left = _channel_metrics(pair[:, 0], sr, start, size)
    right = _channel_metrics(pair[:, 1], sr, start, size)

    return {
        "sample_rate": sr,
        "channels": info["channels"],
        "mono": int(info["channels"] or 0) < 2,
        "duration": round(dur, 3),
        "fft_size": int(size),
        "resolution_hz": round(sr / float(size), 2),
        "period_ms": round(1000.0 * size / float(sr), 2),
        "left": left,
        "right": right,
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


def open_local(cwd, path, title=None, url=None):
    p = Path(path)
    if not p.is_absolute():
        p = (Path(cwd) / p).resolve()
    if not p.exists():
        raise FileNotFoundError(str(p))
    info = probe(p)
    display = (title or "").strip() or p.stem
    title = _sanitize_title(display)
    folder = workspace_root(cwd) / title
    folder.mkdir(parents=True, exist_ok=True)
    for sub in ("video", "audio", "subtitles"):
        (folder / sub).mkdir(exist_ok=True)
    dest = folder / p.name
    if dest.resolve() != p.resolve() and not dest.exists():
        shutil.copy2(p, dest)
    media_path = str(dest if dest.exists() else p)
    asset = {"id": title, "source": "local", "title": display, "source_url": url,
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
