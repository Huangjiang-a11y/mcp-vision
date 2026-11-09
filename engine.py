"""mcp-vision 引擎：probe / 抽帧 / 网格页 / 音频分析 / 工作区。纯 Python，无 Node。"""
import os, re, json, subprocess, time, math, struct, zlib, hashlib, shutil
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


_RE_PTS = re.compile(r"pts_time:([0-9.]+)")
_RE_SCORE = re.compile(r"lavfi\.scene_score=([0-9.]+)")


def scene_scores(path, start, end, timeout=None):
    """窗口内逐帧算画面变化分数，返回按时间排序的 [(t, 秒), ...]。

    t 是绝对秒数（加了 start 偏移）。分数是 ffmpeg scene 滤镜的口径：
    相邻两帧的差异，0 = 完全相同，1 = 完全不同。

    成本全在解码上（275 秒的片子约 18 秒，缩分辨率或降帧率都更慢），
    所以一次扫描就要把整窗都算完，不要按需重扫。
    """
    start, end = float(start), float(end)
    dur = end - start
    if dur <= 0:
        return []
    cmd = ["ffmpeg", "-nostdin", "-hide_banner",
           "-ss", "%.3f" % start, "-t", "%.3f" % dur, "-i", str(path),
           "-vf", "select='gte(scene,0)',metadata=print", "-f", "null", "-"]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout or 900)
    if proc.returncode != 0:
        # 别把「ffmpeg 跑挂了」当成「这段没有变化」，两者下游没法区分
        tail = (proc.stderr or "").strip().splitlines()[-1:] or ["未知错误"]
        raise RuntimeError(tail[0][:120])
    rows, cur = [], None
    for line in proc.stderr.splitlines():
        m = _RE_PTS.search(line)
        if m:
            cur = float(m.group(1))
            continue
        m = _RE_SCORE.search(line)
        if m and cur is not None:
            rows.append((round(start + cur, 3), float(m.group(1))))
            cur = None
    return rows


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
# ---------- 音频分析（level / spectrum / spectrogram / peak / wave / pitch / history）----------
# 口径参考 phyphox 的 Audio Autocorrelation / Frequency History / Audio Amplitude：
# - 分析块 100ms（Frequency History 用 10ms，那是为了刷新率，不看精度）
# - 自相关峰：只留「超过全局最大一半」的局部极大，周期 = 最后一个峰位置 ÷ 峰数
# - RMS 转 dB 用 20*log10（等价 10*log10(rms²)，省一次开方）
# 与 phyphox 不同的一处：本工具的自相关做了能量归一化（跨窗口可比），
# phyphox 靠相对阈值、不归一化。见 README「音频图的读数口径」。
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


# 基频搜索范围。下限 50Hz 够低到男低音（贝斯歌者 ~50Hz）和吉他低音弦；
# 上限 5000Hz 对齐 phyphox 的 Audio Autocorrelation 实验——手机麦克风和
# 大多数乐器的基频都落在 200-5000 Hz，上限再高，周期只有几毫秒，
# autocorrelation 的峰宽和抛物线插值都开始不可靠。
F0_MIN_HZ = 50.0
F0_MAX_HZ = 5000.0
# 有没有周期性：峰值相关系数低于这个数就判无基频（噪声 / 静音）。
# 实测人声 / 乐音帧多在 0.3~0.8，噪声在 0.2 以下。
F0_MIN_CONF = 0.25

# 分析块长度。phyphox 的 Audio Autocorrelation 与 Audio Amplitude 都用
# 100ms，本工具的 spectrum / wave / pitch 沿用这个长度（44.1k 下 4410 点，
# 分辨率约 10Hz）。
# Frequency History 用 20ms 短窗：正好装下 50Hz 的一个整周期（20ms），
# 下限就是搜索下限 50Hz 本身；再短（10ms）就掉到 100Hz，再长（100ms）
# 又会把颤音/滑音抹平。20ms 下 4 秒窗口有 200 个点，趋势也看得清。
FRAME_MS = 100.0
# 电平曲线的 RMS 块长（非重叠）。20ms 是 phyphox Audio Amplitude 的口径。
LEVEL_MS = 20.0
# 轨迹点上限：长窗口逐 20ms 扫太贵（1000 秒 = 5 万帧 x 两声道，自相关要三分钟），
# 超过就把 step 放宽到这个点数——它只用来画趋势，2000 点足够铺满一张图。
MAX_TRACK_FRAMES = 2000
HISTORY_MS = 20.0


def _norm_autocorr(seg, sr):
    """归一化自相关 r(τ)。返回 (r, n)。

    r(τ) = Σ x[i]x[i+τ] / sqrt(Σ x[i]² · Σ x[i+τ]²)，按重叠段能量归一化。
    phyphox 不归一化，靠「最大幅度一半」这种相对阈值找峰；本工具要跨窗口 /
    跨文件横向比读数，归一化后读数与窗口长度无关——这是有意偏离 phyphox 的
    一处，README「音频图的读数口径」里有记。
    不加窗：汉宁窗的包络会把相关峰往小 τ 拉（55Hz 上实测偏 21 音分）。
    """
    x = np.asarray(seg, dtype=np.float64)
    n = x.size
    if n < 64:
        return None, 0
    x = x - x.mean()
    if float(np.dot(x, x)) < 1e-12:
        return None, 0
    nfft = 1 << int(math.ceil(math.log2(2 * n)))
    f = np.fft.rfft(x, nfft)
    ac = np.fft.irfft(f.real ** 2 + f.imag ** 2, nfft)[:n]
    # 重叠能量：τ 越大，两段可用样本越少，分母跟着缩
    e = np.concatenate(([0.0], np.cumsum(x * x)))
    lag = np.arange(n)
    den = np.sqrt(np.maximum(e[n - lag] * (e[n] - e[lag]), 1e-20))
    return ac / den, n


def _local_peaks(r, sr, fmin, fmax):
    """搜索带内的局部极大下标。

    峰可能正好落在搜索区间的端点上（5000Hz 只有 9 个样本），所以极大要在
    [lo-1, hi+1] 里找、再筛回区间内；只在 [lo, hi] 里找会漏掉端点那个峰，
    退到第二个周期，报出低八度。
    """
    n = r.size
    lo = max(1, int(math.floor(sr / float(fmax))))
    hi = min(n - 2, int(math.ceil(sr / float(fmin))))
    if hi <= lo + 1:
        return None
    a0, a1 = max(0, lo - 1), min(n - 1, hi + 1)
    band = r[a0:a1 + 1]
    mid = band[1:-1]
    cand = np.where((mid > band[:-2]) & (mid >= band[2:]))[0] + 1 + a0
    cand = cand[(cand >= lo) & (cand <= hi)]
    return cand if cand.size else None


def _f0_period(r, sr, fmin, fmax):
    """峰挑选（思路见 phyphox 的实验文档）：周期 = 最后一个峰的位置 / 峰数。

    - 门限：只留「超过全局最大一半」的局部极大
    - 位置：取最后一个峰而不是第一个。谐波丰富时第一个峰未必是基频，
      取最后一个 + 除以峰数，等于用整段峰的平均间距估周期，比单看一个峰稳。
    """
    cand = _local_peaks(r, sr, fmin, fmax)
    if cand is None:
        return None
    gmax = float(np.max(r[cand]))
    if gmax < F0_MIN_CONF:          # 静音 / 纯噪声，没有周期可言
        return None
    ok = cand[r[cand] >= 0.5 * gmax]
    if ok.size == 0:
        ok = cand
    last, cnt = int(ok[-1]), int(ok.size)
    # 抛物线插值只作用在「最后一个峰」的位置上：峰数 cnt 是整数，
    # 插了也会被 1/cnt 的粗糙度吃掉；位置插值是为了让平均间距更准
    # （5000Hz 处 1 个样本 = 400Hz，不插值就白搭）
    delta = 0.0
    if 0 < last < r.size - 1:
        a, b, cc = r[last - 1], r[last], r[last + 1]
        d2 = a - 2.0 * b + cc
        if abs(d2) > 1e-12:
            delta = float(np.clip(0.5 * (a - cc) / d2, -0.5, 0.5))
    lag = (last + delta) / float(cnt)
    return lag, float(r[last]), cnt


def _f0_autocorr(seg, sr, fmin=F0_MIN_HZ, fmax=F0_MAX_HZ):
    """归一化自相关测基频：返回可直接上报 / 画图的字段。

    周期用「数峰法」估：最后一个峰的位置 ÷ 峰数（phyphox 文档里描述的同一思路，
    但这里是独立实现，没有引用其 GPL 源码）。比简单取第一个峰稳。
    """
    r, n = _norm_autocorr(seg, sr)
    if r is None:
        return None
    # 上限别越过 Nyquist；参数顺序是 (fmin, fmax)，别传反
    got = _f0_period(r, sr, float(fmin), min(float(fmax), sr * 0.45))
    if got is None:
        return None
    lag, conf, cnt = got
    lag = max(1e-9, lag)
    freq = float(sr) / lag
    period_ms = 1000.0 * lag / float(sr)
    note, cents = freq_to_note(freq)

    # 画图只画搜索范围内的滞后：fmin=50Hz 对应 20ms，正好和参考图同宽。
    # 画到 2.5 个周期会露出搜索范围外的峰——那里若比基频峰还高，
    # 读数却按基频报，图和数据就对不上了。
    span_ms = min(1000.0 * (n - 1) / float(sr), 1000.0 / float(fmin))
    last = int(min(n - 1, max(1, round(span_ms * sr / 1000.0))))
    step = max(1, (last + 1) // 600)
    idx = np.unique(np.append(np.arange(0, last + 1, step), last))
    return {
        "freq_hz": round(freq, 2),
        "period_ms": round(period_ms, 3),
        "note": note,
        "cents": round(cents, 2),
        "confidence": round(conf, 3),
        "peaks": int(cnt),
        "range_hz": [fmin, fmax],
        "span_ms": round(span_ms, 2),
        "curve_ms": [round(float(v), 3) for v in 1000.0 * idx / float(sr)],
        "curve": [round(float(v), 4) for v in r[idx]],
    }


def _pitch_metrics(x, sr, start, size, fmin=F0_MIN_HZ, fmax=F0_MAX_HZ):
    """单个声道的基频：读窗口**起点**那一帧。

    只吃一路样本：分声道还是混单声道由调用方决定。混成单声道会丢掉
    「两路装的是不同声音」这种情况（见 analyze_spectrum 里的实测）。

    **不读「窗口内最响的一帧」**。pitch 是单帧视图，只认 time，窗口给多长
    都不该改变读数；以前拿窗口内最响的 20ms 去定位分析帧，于是同一个 time
    换个 seconds 就得到不同的音高——和 3bba4c1 修的是同一类错。
    两个声道读的是同一个起点，所以天然同一瞬间，不用额外对齐。
    """
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return None
    if x.size < size:
        x = np.pad(x, (0, size - x.size))
    lo = 0
    seg = x[lo:lo + size].astype(np.float64)
    p = _f0_autocorr(seg, sr, fmin, fmax)
    if p is None:
        return None
    p["t0"] = round(float(start) + lo / float(sr), 6)
    p["ms"] = round(1000.0 * seg.size / float(sr), 3)
    return p


def pitch_track(x, sr, start, frame_ms=HISTORY_MS, step_ms=HISTORY_MS,
                fmin=F0_MIN_HZ, fmax=F0_MAX_HZ):
    """phyphox Frequency History：连续短窗测基频，给「音高怎么随时间变」。

    frame_ms 默认 20ms：正好容纳 50Hz 的一个整周期，下限就是 50Hz 本身；
    10ms 会掉到 100Hz，100ms 会把颤音抹平。
    短块对精度来说仍然偏短，读数只用来画轨迹；要准的单点读数看 pitch 视图。
    """
    x = np.asarray(x, dtype=np.float32)
    if x.size < int(sr * float(frame_ms) / 1000.0):
        return []
    flen = max(64, int(sr * float(frame_ms) / 1000.0))
    step = max(1, int(sr * float(step_ms) / 1000.0))
    out = []
    i = 0
    while i + flen <= x.size:
        p = _f0_autocorr(x[i:i + flen].astype(np.float64), sr, fmin, fmax)
        if p is not None:
            out.append([round(float(start) + i / float(sr), 3),
                        round(p["freq_hz"], 2),
                        round(p["confidence"], 3)])
        i += step
    return out


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


def _channel_metrics(x, sr, start, size, level_ms=LEVEL_MS):
    """单个声道的样本 -> 电平 / 频谱 / 时频图。L、R 各算一遍，参数完全一致。

    size 是 FFT 块长（采样点数），level_ms 是电平曲线的 RMS 块长；两者独立，
    因为「频谱要多细」和「电平要多快」不是一个取舍。
    """
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        raise RuntimeError("no audio samples in this window")

    # 电平曲线：level_ms 一帧的 RMS（默认 20ms）
    hop = max(1, int(sr * float(level_ms) / 1000.0))
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
    seg = x[lo: lo + size].astype(np.float64)
    win = np.hanning(seg.size)
    spec = np.abs(np.fft.rfft(seg * win))
    freqs = np.fft.rfftfreq(seg.size, 1.0 / sr)
    # 原始波形就是这个 FFT 窗本身（默认 100ms）：
    # 波形和频谱是同一段数据，不用再换算时间
    wave = {
        "t0": round(float(start) + lo / float(sr), 6),
        "ms": round(1000.0 * seg.size / float(sr), 3),
        "samples": [round(float(v), 8) for v in seg],
        "peak": round(float(np.max(np.abs(seg))), 6) if seg.size else 0.0,
        "rms": round(float(np.sqrt(np.mean(seg ** 2))), 6) if seg.size else 0.0,
    }
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
    # 列数上限 6000：帧短时 size//8 会小到几十个采样，15s 窗口能堆出几万列，
    # 画出来还要再抽稀一遍，纯属浪费
    hop2 = max(1, size // 8, x.size // 6000)
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


def analyze_spectrum(media, start, end, fft_size=None, out_dir=None, track=False,
                     history_ms=HISTORY_MS, level_ms=LEVEL_MS, fft_ms=None,
                     fmin=F0_MIN_HZ):
    """窗口内的音频剖析：L、R 分别算电平 / 频谱 / 时频图 / 基频。

    一次解码出双声道（-ac 2），采样率跟视频音轨走，不重采样。
    单声道音轨会被 ffmpeg 复制成 L=R，结果里 mono=True 标出来。

    三个块长各管一摊，别混：
    fft_ms   频谱 / 波形 / 基频 / 时频图的块长，默认 FRAME_MS（100ms）；
             决定频率分辨率 sr/块长，也决定时频图的列间距。
    level_ms 电平曲线的 RMS 块长，默认 LEVEL_MS（20ms）。
    history_ms  Frequency History 逐帧块长，默认 HISTORY_MS（20ms）。
    fft_size 仍可直接给采样点数，优先于 fft_ms。
    track=True 才逐帧跑 Frequency History 轨迹，只有 history 视图需要；
    fmin 是基频搜索下限，帧太短时由调用方抬高（一帧至少要装下一个周期）。
    """
    media = Path(media)
    info = audio_info(media)
    sr = info["sample_rate"]
    if sr <= 0:
        raise RuntimeError("no audio track")
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
        raise RuntimeError("audio decode failed")
    arr = np.frombuffer(raw[:keep], dtype="<i2").astype(np.float32) / 32768.0
    pair = arr.reshape(-1, 2)
    nsmp = pair.shape[0]
    dur = nsmp / float(sr)
    # 默认块长 100ms（phyphox 的 Audio Autocorrelation / Audio Amplitude 口径）。
    # 调用方可以传 fft_size 覆盖，但给的是「采样点数」而不是时长。
    if not fft_size:
        fft_size = int(round(sr * float(fft_ms or FRAME_MS) / 1000.0))
    size = int(min(max(64, fft_size), nsmp))

    _span_ms = 1000.0 * dur
    _trk_step = float(history_ms)
    if track and _span_ms / max(1e-6, _trk_step) > MAX_TRACK_FRAMES:
        _trk_step = _span_ms / float(MAX_TRACK_FRAMES)

    left = _channel_metrics(pair[:, 0], sr, start, size, level_ms)
    right = _channel_metrics(pair[:, 1], sr, start, size, level_ms)
    # 基频**按声道各测一遍**，不混成单声道。
    # 早先混的理由是「L/R 只差声像和底噪，分开测会给出两个音高，不知道信哪个」——
    # 那只对「同一声源分到两边」成立。故意把两条不同的声音装进 L / R 的素材
    # （双耳素材、左右不同音轨、某些教程）混起来会得到一个两边都没有的音：
    # 实测 t=53.5s，L 是 266.92 Hz C4、R 是 374.99 Hz F#4，混合却报 320.86 Hz E4
    # （−46.65 音分）——自相关把两条周期的最小公倍当成了基频。
    # mono 文件 ffmpeg 会把同一路复制到两边，两边的结果逐字节相同，只报 left。
    mono = int(info["channels"] or 0) < 2
    _empty = lambda: {
        "freq_hz": 0.0, "period_ms": 0.0, "note": "", "cents": 0.0,
        "confidence": 0.0, "range_hz": [fmin, F0_MAX_HZ],
        "span_ms": 20.0, "curve_ms": [], "curve": [],
        "t0": round(float(start), 6), "ms": round(1000.0 * size / float(sr), 3),
    }
    # 两个声道都读窗口起点那一帧：单帧视图只认 time，窗口长度不参与。
    pitch = {"left": _pitch_metrics(pair[:, 0], sr, start, size, fmin) or _empty(),
             "right": _pitch_metrics(pair[:, 1], sr, start, size, fmin) or _empty()}

    return {
        "sample_rate": sr,
        "channels": info["channels"],
        "mono": int(info["channels"] or 0) < 2,
        "duration": round(dur, 3),
        "fft_size": int(size),
        "resolution_hz": round(sr / float(size), 2),
        "period_ms": round(1000.0 * size / float(sr), 2),
        "level_ms": round(1000.0 * max(1, int(sr * float(level_ms) / 1000.0)) / float(sr), 2),
        "pitch": pitch,
        # Frequency History 口径的轨迹：短窗逐帧测基频，看音高怎么变。
        # 精度不如 pitch（块太短），且很费时（10s → 1000 帧），按需才算。
        # 逐帧轨迹同样按声道分开：两路各扫一遍，耗时翻倍，所以仍按需才算。
        "pitch_track": {
            "left": (pitch_track(pair[:, 0], sr, start, frame_ms=history_ms,
                                 step_ms=_trk_step, fmin=fmin) if track else []),
            "right": (pitch_track(pair[:, 1], sr, start, frame_ms=history_ms,
                                  step_ms=_trk_step, fmin=fmin) if track else []),
        },
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
    """列出工作区里的视频。只给模型用得上的：id 和时长。

    不给绝对路径——模型没有文件系统，读不了也拼不了。
    不给 bytes——它没法据此做任何决定。
    id 与 title 绝大多数时候相同，留 id 一个（调用 video 时传的就是它）。
    """
    root = workspace_root(cwd)
    if not root.exists():
        return {"videos": []}
    videos = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        meta = d / "meta.json"
        if not meta.exists():
            continue
        try:
            m = json.loads(meta.read_text())
        except Exception:
            continue
        item = {"id": m.get("id")}
        if m.get("duration"):
            item["duration"] = round(float(m["duration"]), 1)
        videos.append(item)
    return {"videos": videos}


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
            raise RuntimeError("%s is not set; transcription needs three model paths, see INSTALL.md" % name)
        p = Path(v).expanduser()
        if not p.is_file():
            raise RuntimeError("%s points to a missing file: %s" % (name, p))
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
        raise RuntimeError("ffmpeg audio extraction failed: %s" % err)
    if not p.stdout:
        raise RuntimeError("ffmpeg produced no audio (the video may have no audio track)")
    return np.frombuffer(p.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def _tag(v):
    """SenseVoice 的富标签 <|NEUTRAL|> -> NEUTRAL。

    result.emotion / event / lang 都是这个形状，空值或异常一律返回空串。
    """
    t = str(v or "").strip()
    if t.startswith("<|") and t.endswith("|>"):
        t = t[2:-2]
    return t if t and t.upper() != "NONE" else ""


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
    """按 VAD 自动断句，逐段识别。

    返回 [{s, e, text, emotion?, event?, lang?}]（秒，相对整片）。
    emotion / event / lang 来自 SenseVoice 的同一次前向，不额外花钱；
    只有非默认值才出现（NEUTRAL / Speech / EMO_UNKNOWN 省略）。
    """
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
        item = {"s": round(a, 2), "e": round(a + dur, 2), "text": text}
        # SenseVoice 一次前向顺带给出情感 / 音频事件 / 语种，丢掉就白算了。
        # 默认值不写：NEUTRAL、Speech、以及模型给的 EMO_UNKNOWN 都是「没说啥」。
        emo = _tag(getattr(s.result, "emotion", ""))
        ev = _tag(getattr(s.result, "event", ""))
        lang = _tag(getattr(s.result, "lang", ""))
        if emo and emo != "NEUTRAL" and emo != "EMO_UNKNOWN":
            item["emotion"] = emo
        if ev and ev != "Speech":
            item["event"] = ev
        if lang:
            item["lang"] = lang
        out.append(item)

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
