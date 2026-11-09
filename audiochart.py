"""把音频剖析结果画成 PNG。纯 numpy 光栅化 + engine._write_png，不依赖绘图库。"""
import math
import numpy as np

from engine import _write_png

FG = (255, 255, 0)       # 纯黄
GRID = (72, 72, 0)        # 同色相的暗黄，避免网格压过曲线
AXIS = (150, 150, 0)
TXT = (255, 255, 255)     # 白色
DIM = (255, 255, 0)

_F = {
    "A": [".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"],
    "B": ["####.", "#...#", "#...#", "####.", "#...#", "#...#", "####."],
    "C": [".###.", "#...#", "#....", "#....", "#....", "#...#", ".###."],
    "D": ["####.", "#...#", "#...#", "#...#", "#...#", "#...#", "####."],
    "E": ["#####", "#....", "#....", "####.", "#....", "#....", "#####"],
    "F": ["#####", "#....", "#....", "####.", "#....", "#....", "#...."],
    "G": [".###.", "#...#", "#....", "#.###", "#...#", "#...#", ".###."],
    "H": ["#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"],
    "I": ["#####", "..#..", "..#..", "..#..", "..#..", "..#..", "#####"],
    "J": ["..###", "...#.", "...#.", "...#.", "...#.", "#..#.", ".##.."],
    "K": ["#...#", "#..#.", "#.#..", "##...", "#.#..", "#..#.", "#...#"],
    "L": ["#....", "#....", "#....", "#....", "#....", "#....", "#####"],
    "M": ["#...#", "##.##", "#.#.#", "#...#", "#...#", "#...#", "#...#"],
    "N": ["#...#", "##..#", "#.#.#", "#..##", "#...#", "#...#", "#...#"],
    "O": [".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."],
    "P": ["####.", "#...#", "#...#", "####.", "#....", "#....", "#...."],
    "Q": [".###.", "#...#", "#...#", "#...#", "#.#.#", "#..#.", ".##.#"],
    "R": ["####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"],
    "S": [".####", "#....", "#....", ".###.", "....#", "....#", "####."],
    "T": ["#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."],
    "U": ["#...#", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."],
    "V": ["#...#", "#...#", "#...#", "#...#", "#...#", ".#.#.", "..#.."],
    "W": ["#...#", "#...#", "#...#", "#...#", "#.#.#", "##.##", "#...#"],
    "X": ["#...#", "#...#", ".#.#.", "..#..", ".#.#.", "#...#", "#...#"],
    "Y": ["#...#", "#...#", ".#.#.", "..#..", "..#..", "..#..", "..#.."],
    "Z": ["#####", "....#", "...#.", "..#..", ".#...", "#....", "#####"],
    "0": [".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."],
    "1": ["..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."],
    "2": [".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"],
    "3": ["#####", "...#.", "..#..", "...#.", "....#", "#...#", ".###."],
    "4": ["...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."],
    "5": ["#####", "#....", "####.", "....#", "....#", "#...#", ".###."],
    "6": ["..##.", ".#...", "#....", "####.", "#...#", "#...#", ".###."],
    "7": ["#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."],
    "8": [".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."],
    "9": [".###.", "#...#", "#...#", ".####", "....#", "...#.", ".##.."],
    " ": [".....", ".....", ".....", ".....", ".....", ".....", "....."],
    ".": [".....", ".....", ".....", ".....", ".....", ".##..", ".##.."],
    ",": [".....", ".....", ".....", ".....", ".##..", ".##..", ".#..."],
    "-": [".....", ".....", ".....", "#####", ".....", ".....", "....."],
    "+": [".....", "..#..", "..#..", "#####", "..#..", "..#..", "....."],
    ":": [".....", ".##..", ".##..", ".....", ".##..", ".##..", "....."],
    "%": ["##..#", "##..#", "...#.", "..#..", ".#...", "#..##", "#..##"],
    "(": ["...#.", "..#..", ".#...", ".#...", ".#...", "..#..", "...#."],
    ")": [".#...", "..#..", "...#.", "...#.", "...#.", "..#..", ".#..."],
    "/": ["....#", "....#", "...#.", "..#..", ".#...", "#....", "#...."],
    "#": [".#.#.", ".#.#.", "#####", ".#.#.", "#####", ".#.#.", ".#.#."],
}


class Canvas:
    def __init__(self, w, h, bg=(18, 18, 22)):
        self.w, self.h = w, h
        self.buf = np.zeros((h, w, 3), dtype=np.uint8)
        self.buf[:] = bg

    def px(self, x, y, c):
        x, y = int(x), int(y)
        if 0 <= x < self.w and 0 <= y < self.h:
            self.buf[y, x] = c

    def hline(self, y, x0, x1, c, t=1):
        x0, x1 = int(min(x0, x1)), int(max(x0, x1))
        for k in range(t):
            yy = int(y) + k
            if 0 <= yy < self.h:
                self.buf[yy, max(0, x0):min(self.w, x1 + 1)] = c

    def vline(self, x, y0, y1, c, t=1):
        y0, y1 = int(min(y0, y1)), int(max(y0, y1))
        for k in range(t):
            xx = int(x) + k
            if 0 <= xx < self.w:
                self.buf[max(0, y0):min(self.h, y1 + 1), xx] = c

    def line(self, x0, y0, x1, y1, c, t=1):
        n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        xs = np.linspace(x0, x1, n).round().astype(int)
        ys = np.linspace(y0, y1, n).round().astype(int)
        for x, y in zip(xs, ys):
            for k in range(t):
                self.px(x + k, y, c)

    def curve(self, xs, ys, x0, y0, x1, y1, c=FG, ss=4, lw=2.0):
        """按列超采样再缩回，画连续曲线；直线插值在像素格上是台阶，看着全是锯齿。"""
        W = int(x1 - x0 + 1)
        H = int(y1 - y0 + 1)
        gx = np.linspace(x0, x1, W * ss)
        gy = np.interp(gx, xs, ys)
        rows = np.round((gy - y0) * ss).astype(int)
        mask = np.zeros((H * ss, W * ss), dtype=np.float32)
        half = max(1, int(round(lw * ss / 2.0)))
        for i in range(len(rows) - 1):
            a, b = int(rows[i]), int(rows[i + 1])
            lo, hi = (a, b) if a <= b else (b, a)
            lo = max(0, lo - half)
            hi = min(H * ss - 1, hi + half)
            if hi >= lo:
                mask[lo:hi + 1, i] = 1.0
        # 列与列之间斜率突变处会留下半透明的暗段，先横向补一格再缩回
        mask[:, 1:] = np.maximum(mask[:, 1:], mask[:, :-1])
        alpha = mask.reshape(H, ss, W, ss).mean(axis=(1, 3))[..., None]
        alpha = np.clip(alpha * 1.35, 0.0, 1.0)
        sub = self.buf[y0:y0 + H, x0:x0 + W].astype(np.float32)
        col = np.array(c, dtype=np.float32)
        self.buf[y0:y0 + H, x0:x0 + W] = np.clip(
            sub * (1.0 - alpha) + col * alpha, 0, 255).astype(np.uint8)

    def rect(self, x0, y0, x1, y1, c, fill=False):
        if fill:
            self.buf[max(0, y0):min(self.h, y1), max(0, x0):min(self.w, x1)] = c
        else:
            self.hline(y0, x0, x1, c); self.hline(y1, x0, x1, c)
            self.vline(x0, y0, y1, c); self.vline(x1, y0, y1, c)

    def text(self, x, y, s, c=TXT, scale=2):
        cx, y = int(x), int(y)
        for ch in str(s).upper():
            g = _F.get(ch, _F[" "])
            for r, row in enumerate(g):
                for col, bit in enumerate(row):
                    if bit == "#":
                        yy, xx = y + r * scale, cx + col * scale
                        self.buf[yy:yy + scale, xx:xx + scale] = c
            cx += 6 * scale
        return cx

    def ctext(self, x, y, s, c=DIM, scale=2):
        """以 x 为中心写，自动夹在画布内，避免边缘刻度被裁。"""
        tw = self.tw(s, scale)
        xx = min(max(x - tw / 2.0, 4), self.w - tw - 4)
        return self.text(xx, y, s, c, scale)

    def vtext(self, x, y, s, c=DIM, scale=2):
        """竖排：每个字符占一行，用于纵轴标题。"""
        for i, ch in enumerate(str(s).upper()):
            self.text(x, y + i * (7 * scale + 2), ch, c, scale)

    def tw(self, s, scale=2):
        return len(str(s)) * 6 * scale

    def blit(self, rgb, x0, y0):
        h, w = rgb.shape[:2]
        self.buf[y0:y0 + h, x0:x0 + w] = rgb

    def save(self, path):
        _write_png(str(path), self.buf, self.w, self.h)
        return str(path)


def _cmap(v):
    """0..1 -> 黑 / 暗橙 / 橙 / 亮橙 / 白"""
    stops = [0.0, 0.22, 0.5, 0.78, 1.0]
    r = np.interp(v, stops, [0, 70, 200, 255, 255])
    g = np.interp(v, stops, [0, 70, 200, 255, 255])
    b = np.interp(v, stops, [0, 0, 0, 60, 255])
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def _nice_db(v):
    return ("%d" % v) if abs(v - round(v)) < 1e-9 else ("%.1f" % v)


def _tag_title(tag, name):
    return ("%s  %s" % (tag, name)) if tag else name


def _caption(c, d, x1, y=30):
    """右上角一行小字：峰值频率 / 音名 / 最大电平，和文本输出对得上。"""
    sp, lv = d["spectrum"], d["level"]
    txt = "PEAK %.1fHZ  %s  MAX %.1fDB" % (sp["peak_freq_hz"], sp["note"], lv["max"])
    c.text(x1 - c.tw(txt, 2), y, txt, DIM, 2)


# 音频图整图边长。上下两半各 FIG_W//2，合起来是正方形。
# 1024 时一张 PNG 的 base64 接近 1MB，768 能省四成多，读数并不受影响。
FIG_W = 768


def plot_level(d, start, end, tag="", w=FIG_W, h=FIG_W // 2):
    lv = d["level"]
    t = np.asarray(lv["times"], dtype=float)
    v = np.asarray(lv["values"], dtype=float)
    step = max(1, len(v) // 900)
    t, v = t[::step], v[::step]
    c = Canvas(w, h)
    x0, y0, x1, y1 = 64, 72, w - 22, h - 42
    lo = math.floor(min(v.min(), lv["mean"] - 6) / 5.0) * 5.0
    hi = math.ceil(max(v.max(), lv["mean"] + 6) / 5.0) * 5.0
    if hi - lo < 6:
        hi = lo + 6
    n = len(v) if len(v) > 1 else 2
    def X(i): return x0 + (x1 - x0) * (i / float(n - 1))
    def Y(db): return y1 - (db - lo) / (hi - lo) * (y1 - y0)
    dstep = _nice_step(hi - lo, 5)              # 刻度落在 1/2/2.5/5 的整倍数上
    dv = math.ceil(lo / dstep - 1e-9) * dstep
    while dv <= hi + 1e-9:
        yy = Y(dv)
        c.hline(yy, x0, x1, GRID)
        lab = _nice_db(round(dv, 1))
        c.text(x0 - c.tw(lab, 1) - 10, yy - 7, lab, DIM, 1)
        dv += dstep
    for k in range(5):
        frac = k / 4.0
        tt = start + (end - start) * frac
        xx = x0 + (x1 - x0) * frac
        c.vline(xx, y0, y1, GRID)
        lab = "%.1f" % tt
        c.ctext(xx, y1 + 12, lab, DIM, 1)
    c.rect(x0, y0, x1, y1, AXIS)
    c.curve(np.array([X(i) for i in range(len(v))]), np.array([Y(x) for x in v]),
            x0, y0, x1, y1)
    big = "%.1f" % lv["db"]
    c.text(x0, 14, _tag_title(tag, "SOUND LEVEL"), TXT, 3)
    bx = x1 - c.tw(big, 5) - 44
    c.text(bx, 8, big, FG, 5)
    c.text(bx + c.tw(big, 5) + 8, 30, "DB", DIM, 2)
    c.text(x0, 48, "TIME (S)", DIM, 2)
    c.text(x0 + 6, y0 + 6, "DB", DIM, 1)
    _caption(c, d, x1, 52)
    return c


# 色标/纵轴按「相对值」自动伸缩：顶端恒为 1（本窗口最大值），
# 底端退到整十档，跟那个应用一样按数据自己找范围。
AMP_MIN_DECADES, AMP_MAX_DECADES = 3, 6     # 色带往下 3~6 档（60~120dB）


def _amp_range(db):
    """绝对 dB 谱 -> (谱, 下界 dB, 上界 dB)。

    顶端 = 峰值向上取整到 10 的整数次幂（那个应用就是这样：峰值 3 → 顶格 10），
    底端按 2% 分位往下退 3~6 档，避免个别极暗 bin 把整条带子拉长。
    """
    a = np.asarray(db, dtype=float)
    if a.size == 0:
        return a, -100.0, 0.0
    peak = float(np.max(a))
    hi = 20.0 * math.ceil(peak / 20.0 - 1e-9)        # 升到 10 的整数次幂
    if hi < peak:
        hi = peak
    lo_pct = float(np.percentile(a, 2.0)) if a.size > 20 else peak - 60.0
    n = int(math.ceil((hi - lo_pct) / 20.0))
    n = max(AMP_MIN_DECADES, min(AMP_MAX_DECADES, n))
    return a, hi - 20.0 * n, hi


def _amp_ticks(lo, hi):
    """整十进位刻度：跨度大就抽稀，最多 5 条，跟应用一样只标几个。"""
    n = int(round((hi - lo) / 20.0))
    step = 1 if n <= 3 else (2 if n <= 6 else 3)
    out = []
    for i in range(n + 1):
        if i % step:
            continue
        db = hi - 20.0 * i
        amp = 10.0 ** (db / 20.0)
        lab = "%.0e" % amp if amp < 0.01 else ("%g" % amp)
        out.append((amp, lab))
    return out


def _amp_frac(a, lo, hi):
    db = 20.0 * math.log10(max(a, 1e-12))
    return (db - lo) / max(1e-9, hi - lo)


def _nyq(d, override=None):
    """横轴右端：优先用调用方传的（由采样率算出的 Nyquist），缺省退回 24k。"""
    if override and override > 0:
        return float(override)
    sr = float((d or {}).get("sample_rate") or 0)
    return sr / 2.0 if sr > 0 else 24000.0


def _freq_ticks(fmax):
    base = [f for f in (50, 100, 500, 1000, 5000, 10000, 24000) if f < fmax * 0.98]
    ticks = list(base)
    if not ticks or fmax > ticks[-1] * 1.05:
        ticks.append(fmax)
    return ticks


def plot_spectrum(d, tag="", resolution_hz=0.0, w=FIG_W, h=FIG_W // 2, fmax=None):
    sp = d["spectrum"]
    f = np.asarray(sp["freqs"], dtype=float)
    m = np.asarray(sp["mag_db"], dtype=float)
    c = Canvas(w, h)
    x0, y0, x1, y1 = 112, 92, w - 22, h - 40
    # 横轴对数，尾端取实际 Nyquist（44.1k 素材是 22050）
    fmin = 10.0
    fmax = _nyq(d, fmax)
    rel, dmin, dmax = _amp_range(m)     # 顶端 = 峰值向上取整到 10 的幂，底端退 3~6 档
    def X(fr): return x0 + math.log10(max(fr, fmin) / fmin) / math.log10(fmax / fmin) * (x1 - x0)
    def Y(db): return y1 - (min(dmax, max(dmin, db)) - dmin) / (dmax - dmin) * (y1 - y0)
    # 纵轴按绝对 FFT 幅度标（满幅正弦 = 1），跨窗口可比
    for amp, lab in _amp_ticks(dmin, dmax):
        db = 20.0 * math.log10(amp)
        yy = Y(db)
        c.hline(yy, x0, x1, GRID)
        ty = yy + 5 if yy < y0 + 16 else yy - 7   # 顶格那条往下写，别压标题行
        c.text(x0 - c.tw(lab, 1) - 10, ty, lab, DIM, 1)
    for fr in _freq_ticks(fmax):
        xx = X(fr)
        c.vline(xx, y0, y1, GRID)
        c.ctext(xx, y1 + 12, "%g" % fr, DIM, 1)
    c.rect(x0, y0, x1, y1, AXIS)
    # 1kHz 以上每个像素列挤着 5 个 bin，逐 bin 连线会在像素里来回折、看着全是锯齿；
    # 低频段反过来，几个像素才摊到一个 bin，直接取最近 bin 又成了台阶。
    # 所以：有多个 bin 的列取功率均值，只有一个 bin 的列用相邻 bin 插值。
    W = x1 - x0
    fe = fmin * (fmax / fmin) ** (np.arange(W + 1) / float(W))
    lo = np.clip(np.searchsorted(f, fe[:-1]), 0, len(f) - 1)
    hi = np.clip(np.searchsorted(f, fe[1:]), lo + 1, len(f))
    cnt = hi - lo
    pw = 10.0 ** (rel / 10.0)
    cs = np.concatenate([[0.0], np.cumsum(pw)])
    band = 10.0 * np.log10(np.maximum((cs[hi] - cs[lo]) / cnt, 1e-12))
    smooth = np.interp(np.log(fe[:-1]), np.log(np.maximum(f, 1.0)), rel)
    vals = np.where(cnt > 1, band, smooth)
    # 超过 Nyquist 的列没有数据；跌到纵轴下限以下的列也别画，
    # 否则会沿底边拖出一条假的平线
    fnyq = float(f[-1])
    Wn = int(np.searchsorted(fe[:-1], fnyq, side="right"))
    below = np.where(vals[:max(2, Wn)] < dmin - 1.5)[0]
    if below.size:
        Wn = max(2, int(below[0]) + 1)
    Wn = max(2, min(Wn, W))
    # 注意 curve() 内部按 x0..x1 铺满整条轴、范围外沿用端点值，
    # 所以右界要传实际画到的列，否则截断的那段会被一条平线补回去
    c.curve(np.arange(Wn) + x0, np.array([Y(v) for v in vals[:Wn]]),
            x0, y0, x0 + max(1, Wn - 1), y1)
    pk = sp["peak_freq_hz"]
    if pk > fmin:
        c.vline(X(pk), y0, y1, (255, 255, 255), 1)
    c.text(x0, 12, _tag_title(tag, "SPECTRUM"), TXT, 3)
    rows = [("PEAK FREQ", "%.2f HZ" % pk), ("NOTE", "%s" % sp["note"]),
            ("CENTS", "%+.2f" % sp["cents"]), ("RES", "%.2f HZ" % resolution_hz)]
    colw = (x1 - x0) // 4
    for k, (lab, val) in enumerate(rows):
        xx = x0 + k * colw
        c.text(xx, 48, lab, DIM, 2)
        c.text(xx, 66, val, FG, 3)
    c.text(x1 - c.tw("FREQ (HZ)", 2), 14, "FREQ (HZ)", DIM, 2)
    c.vtext(8, y0 + 8, "FFT AMP", DIM, 2)
    return c


def _nice_step(span, want=5):
    """把跨度切成 want 段左右，步长取 1/2/2.5/5 的十倍数，刻度才整齐。"""
    raw = max(1e-9, span / float(max(1, want - 1)))
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


def _snap_time_end(t1, t0):
    """末端显示成一位小数后离整数不到 0.1，就贴过去，顶格刻度才画得出来。

    15 秒窗口最后一列常是 14.893，直接比差会差 0.107 而漏掉；按一位小数
    （14.9）判断，正好落在 ±0.1 里。
    """
    r = float(round(t1))
    if r > t0 and abs(round(t1, 1) - r) <= 0.1 + 1e-9:
        return r
    return t1


def _time_ticks(t0, t1, want=5):
    """窗口内的等距刻度，落在 step 的整数倍上，不出现 6.4 / 6.9 这种歪数。"""
    step = _nice_step(t1 - t0, want)
    k = math.ceil(t0 / step - 1e-9)
    ticks, txt = [], "%.1f" if step >= 0.1 else "%.2f"
    while True:
        v = k * step
        if v > t1 + 1e-9:
            break
        # 14.9 / 15.0 离整数只有 0.1 的值显示成整数，轴上不写 14.9
        lab = ("%d" % int(round(v))) if abs(v - round(v)) <= 0.1 + 1e-9 else (txt % v)
        ticks.append((v, lab))
        k += 1
    return ticks


def plot_spectrogram(d, tag="", w=FIG_W, h=FIG_W // 2, fmax=None):
    """时频图：横轴对数频率 20..Nyquist，纵轴时间（向上）。"""
    sg = d["spectrogram"]
    spec = np.asarray(sg["spec_db"], dtype=float)   # (nfreq, ntime)
    times = np.asarray(sg["times"], dtype=float)
    freqs = np.asarray(sg["freqs"], dtype=float)
    # 全程按「相对值」处理：顶端 = 本窗口峰值 = 1，底端自动退到整十档
    rel, alo, ahi = _amp_range(spec)   # 顶端 = 峰值向上取整到 10 的幂
    c = Canvas(w, h)
    x0, x1 = 72, w - 22
    # 每半幅一条色标，范围和绘图用的 alo..ahi 一一对应
    by, bh = 40, 12
    ramp = np.repeat(np.linspace(0, 1, x1 - x0)[None, :], bh, axis=0)
    c.blit(_cmap(ramp), x0, by)
    c.rect(x0, by, x1, by + bh, AXIS)
    for amp, lab in _amp_ticks(alo, ahi):
        frac = _amp_frac(amp, alo, ahi)
        xx = x0 + frac * (x1 - x0)
        c.vline(xx, by + 1, by + bh - 2, GRID, 1)      # 刻度刺
        if frac > 0.97:
            c.text(x1 - c.tw(lab, 1), by + bh + 2, lab, DIM, 1)
        else:
            c.ctext(xx, by + bh + 2, lab, DIM, 1)
    y0, y1 = 76, h - 38
    H, W = y1 - y0, x1 - x0
    nt = rel.shape[1]
    if nt < 2:
        rel = np.repeat(rel, 2, axis=1); nt = 2
    fmin = 20.0
    fmax = max(_nyq(d, fmax), fmin * 2)    # 尾端就是 Nyquist，不再留黑带
    def X(fr): return x0 + math.log10(max(fr, fmin) / fmin) / math.log10(fmax / fmin) * (x1 - x0)
    fcols = fmin * (fmax / fmin) ** (np.arange(W) / float(max(1, W - 1)))
    yt = np.linspace(nt - 1, 0, H).astype(int)      # 时间向上
    # 对数轴下每个像素覆盖的频率宽度差很多，直接取最近 bin 会一格一格的；
    # 在对数频率上做线性插值，低中频才连续
    lf = np.log(np.maximum(freqs, 1e-3))
    lfq = np.log(fcols)
    mat = np.empty((H, W), dtype=np.float64)
    src = rel[:, yt]
    for j in range(H):
        mat[j, :] = np.interp(lfq, lf, src[:, j])
    mat[:, fcols > freqs[-1]] = alo            # 超过 Nyquist 的部分没有数据
    v = np.clip((mat - alo) / max(1e-9, ahi - alo), 0, 1)
    c.blit(_cmap(v), x0, y0)
    c.rect(x0, y0, x1, y1, AXIS)
    title = _tag_title(tag, "SPECTROGRAM")
    c.text(x0, 12, title, TXT, 3)
    c.text(x0 + c.tw(title, 3) + 14, 17, "FFT AMP (REL)", DIM, 2)
    c.text(x1 - c.tw("FREQ (HZ, LOG)", 2), 14, "FREQ (HZ, LOG)", DIM, 2)
    c.vtext(8, y0 + 8, "TIME", DIM, 2)
    ticks = _freq_ticks(fmax)
    for fr in ticks:
        xx = X(fr)
        c.vline(xx, y0, y1, (0, 0, 0), 1)
        c.ctext(xx, y1 + 10, "%g" % fr, DIM, 1)
    t0, t1 = (float(times[0]), float(times[-1])) if times.size > 1 else (0.0, 1.0)
    t1 = _snap_time_end(t1, t0)
    for val, lab in _time_ticks(t0, t1):
        frac = (val - t0) / max(1e-9, t1 - t0)
        yy = y1 - (y1 - y0) * frac
        c.hline(yy, x0, x1, (0, 0, 0), 1)
        c.text(x0 - c.tw(lab, 1) - 10, yy - 7, lab, DIM, 1)
    return c


def plot_peak(d, tag="", w=FIG_W, h=FIG_W // 2):
    """峰值频率轨迹单独一张：横轴时间，纵轴按轨迹自身范围自适应。"""
    sg = d["spectrogram"]
    times = np.asarray(sg["times"], dtype=float)
    tr = sg["peak_track"]
    c = Canvas(w, h)
    x0, x1 = 84, w - 22
    y0, y1 = 62, h - 42
    fs = [float(f) for _, f in tr] or [1.0]
    lo, hi = min(fs), max(fs)
    step = _nice_step(max(hi - lo, 1.0), 5)
    lo = max(0.0, math.floor(lo / step) * step)
    hi = math.ceil(hi / step) * step
    if hi - lo < step:
        hi = lo + step
    def Y(f): return y1 - (f - lo) / (hi - lo) * (y1 - y0)
    c.text(x0, 12, _tag_title(tag, "PEAK FREQ TRACK"), TXT, 3)
    c.vtext(8, y0 + 8, "FREQ", DIM, 2)
    v = lo
    while v <= hi + 1e-9:
        yy = Y(v)
        c.hline(yy, x0, x1, GRID)
        lab = "%g" % round(v, 2)
        c.text(x0 - c.tw(lab, 1) - 10, yy - 7, lab, DIM, 1)
        v += step
    c.rect(x0, y0, x1, y1, AXIS)
    t0, t1 = (float(times[0]), float(times[-1])) if times.size > 1 else (0.0, 1.0)
    t1 = _snap_time_end(t1, t0)
    for val, lab in _time_ticks(t0, t1):
        frac = (val - t0) / max(1e-9, t1 - t0)
        xx = x0 + (x1 - x0) * frac
        c.vline(xx, y0, y1, GRID)
        c.ctext(xx, y1 + 12, lab, DIM, 1)
    if tr:
        span = (t1 - t0) if t1 > t0 else 1.0
        pts = []
        for tt, ff in tr:
            frac = (tt - t0) / span
            pts.append((x0 + (x1 - x0) * min(1.0, max(0.0, frac)), Y(ff)))
        if len(pts) > 1:
            xs = np.array([p[0] for p in pts])
            c.curve(xs, np.array([p[1] for p in pts]),
                    x0, y0, float(xs.max()), y1, GRID, lw=1.0)
        for xx, yy in pts:
            for a2 in range(-2, 3):
                for b2 in range(-2, 3):
                    if a2 * a2 + b2 * b2 <= 4:
                        c.px(xx + a2, yy + b2, FG)
    c.text(x1 - c.tw("TIME (S)", 2), h - 18, "TIME (S)", DIM, 2)
    _caption(c, d, x1, 40)
    return c


def plot_wave(d, tag="", w=FIG_W, h=FIG_W // 2):
    """原始波形：与频谱同一处声音，约 40ms。横轴毫秒，纵轴相对幅值（满幅 = 1）。"""
    wv = (d or {}).get("wave") or {}
    s = np.asarray(wv.get("samples") or [], dtype=float)
    if s.size < 2:
        s = np.zeros(2)
    c = Canvas(w, h)
    x0, y0, x1, y1 = 104, 62, w - 22, h - 58
    peak = max(float(np.max(np.abs(s))), 1e-9)
    step = _nice_step(peak * 2.0, 5)
    hi = math.ceil(peak / step - 1e-9) * step
    if hi < step:
        hi = step
    lo = -hi
    def Y(a): return y1 - (min(hi, max(lo, a)) - lo) / (hi - lo) * (y1 - y0)
    dec = 0 if step >= 1 else (1 if step >= 0.1 else (2 if step >= 0.01 else 3))
    fmt = "%." + str(dec) + "f"
    v = lo
    while v <= hi + step * 1e-9:
        yy = Y(v)
        c.hline(yy, x0, x1, AXIS if abs(v) < step * 1e-9 else GRID)
        lab = fmt % v
        c.text(x0 - c.tw(lab, 1) - 10, yy - 7, lab, DIM, 1)
        v += step
    ms = float(wv.get("ms") or 40.0)
    xstep = _nice_step(ms, 5)
    t = 0.0
    while t <= ms + 1e-9:
        xx = x0 + (x1 - x0) * (t / ms)
        if t > 0:
            c.vline(xx, y0, y1, GRID)
        c.ctext(xx, y1 + 12, ("%.0f" % t) if xstep >= 1 else ("%.1f" % t), DIM, 1)
        t += xstep
    c.rect(x0, y0, x1, y1, AXIS)
    W = x1 - x0
    spp = s.size / float(max(1, W))          # samples per pixel
    xs = x0 + (x1 - x0) * np.linspace(0.0, 1.0, s.size)
    # 每像素不足 8 个样本时，min/max 会把每列画成一根从极小到极大的竖线，
    # 一屏看着全是尖刺；这时样本点本身够密，直接连线才是真实波形。
    # 8 个以上才用包络，否则连线会因抽稀而走样（librosa / DSP 都这么分档）。
    if spp >= 8:
        idx = np.linspace(0, s.size, W + 1).astype(int)
        for i in range(W):
            a = idx[i]
            b = max(a + 1, idx[i + 1])
            seg = s[a:b]
            if seg.size:
                c.vline(x0 + i, Y(float(seg.max())), Y(float(seg.min())), FG, 1)
        # 包络底下再描一条暗一点的样本连线，保留波形形状
        c.curve(xs, np.array([Y(v) for v in s]), x0, y0, x1, y1, AXIS, lw=1.0)
    else:
        c.curve(xs, np.array([Y(v) for v in s]), x0, y0, x1, y1)
    c.text(x0, 12, _tag_title(tag, "WAVE"), TXT, 3)
    c.text(x0, 48, "T %.3fS  %.0fMS" % (float(wv.get("t0") or 0.0), ms), DIM, 2)
    c.vtext(8, y0 + 8, "AMP", DIM, 2)
    cap = "PEAK %.3f  RMS %.3f" % (float(wv.get("peak") or 0.0), float(wv.get("rms") or 0.0))
    c.text(x1 - c.tw(cap, 2), 30, cap, DIM, 2)
    c.text(x1 - c.tw("TIME (MS)", 2), h - 14, "TIME (MS)", DIM, 2)
    return c


def plot_pitch(pt, tag="", w=FIG_W, h=FIG_W):
    """一个声道的基频自相关：横轴滞后 ms，纵轴归一化相关系数。

    pt 是该声道的 pitch 字典（不是整份剖析结果）——L / R 各画一张，
    双声道时上下拼接，和 level / peak 的排法一致。
    和参考的「音频自相关」一个意思：周期那个峰在 3.31ms，倒数就是 302.46Hz。
    """
    pt = pt or {}
    xs = np.asarray(pt.get("curve_ms") or [], dtype=float)
    ys = np.asarray(pt.get("curve") or [], dtype=float)
    c = Canvas(w, h)
    # 上面留给标题 + 一排读数（和 spectrum 的排版一致）
    x0, y0, x1, y1 = 112, 118, w - 22, h - 72
    span = float(pt.get("span_ms") or 20.0)
    if span <= 0:
        span = 20.0

    # 纵轴不固定 -1..1：语音和乐器的相关值基本落在 -0.3..1，
    # 留一整段负空间只会把曲线挤到上半截。下边界跟着数据走。
    ymin = -0.2 if ys.size == 0 else min(-0.2, float(np.min(ys)))
    ymax = 1.0
    # 步长定下来再把下边界对齐到步长的整数倍，否则刻度会从 -0.25、0.25
    # 这种半格位置起跳，0 那条线反而画不出来
    ystep = _nice_step(ymax - ymin, 5)
    ymin = math.floor(ymin / ystep + 1e-9) * ystep

    def X(ms): return x0 + (x1 - x0) * min(1.0, max(0.0, ms / span))
    def Y(v): return y1 - (min(ymax, max(ymin, v)) - ymin) / (ymax - ymin) * (y1 - y0)

    v = ymin
    while v <= ymax + ystep * 1e-9:
        yy = Y(v)
        c.hline(yy, x0, x1, AXIS if abs(v) < 1e-9 else GRID)
        lab = "%.2f" % v
        c.text(x0 - c.tw(lab, 1) - 10, yy - 7, lab, DIM, 1)
        v += ystep

    xstep = _nice_step(span, 5)
    t = 0.0
    while t <= span + 1e-9:
        xx = X(t)
        if t > 0:
            c.vline(xx, y0, y1, GRID)
        c.ctext(xx, y1 + 12, ("%.0f" % t) if xstep >= 1 else ("%.1f" % t), DIM, 1)
        t += xstep
    c.rect(x0, y0, x1, y1, AXIS)

    freq = float(pt.get("freq_hz") or 0.0)
    per = float(pt.get("period_ms") or 0.0)
    conf = float(pt.get("confidence") or 0.0)
    if xs.size >= 2:
        c.curve(x0 + (x1 - x0) * xs / span, np.array([Y(y) for y in ys]),
                x0, y0, x1, y1)
    else:
        c.hline(Y(0.0), x0, x1, FG)
        c.ctext((x0 + x1) / 2.0, y0 + 18, "NO PERIODIC SIGNAL", DIM, 2)

    if freq > 0:
        # 周期位置竖虚线 + 相关峰上的点：数字和曲线对得上，不用自己数格子
        px = X(per)
        for yy in range(y0 + 4, y1, 10):
            c.vline(px, yy, min(yy + 5, y1), (255, 255, 255), 1)
        py = Y(conf)
        for a in range(-4, 5):
            for b in range(-4, 5):
                if a * a + b * b <= 16:
                    c.px(px + a, py + b, TXT)

    c.text(x0, 12, _tag_title(tag, "AUTOCORRELATION"), TXT, 3 if h >= 0.9 * FIG_W else 2)
    # 保留位数按列宽定：一行四列、每列 158px，scale 3 下一字符 18px，
    # 超过 8 个字符就会压到下一列的标签上（1000.00HZ 正好 9 个）。
    fhz = ("%.0fHZ" % freq) if freq >= 1000 else ("%.1fHZ" % freq)
    rows = [("PERIOD", ("%.2fMS" % per) if freq > 0 else "---"),
            ("FREQ", fhz if freq > 0 else "---"),
            ("NOTE", str(pt.get("note") or "-")),
            ("CENTS", ("%+.2f" % float(pt.get("cents") or 0.0)) if freq > 0 else "---")]
    colw = (x1 - x0) // 4
    for k, (lab, val) in enumerate(rows):
        xx = x0 + k * colw
        c.text(xx, 48, lab, DIM, 2)
        c.text(xx, 66, val, FG, 3)
    cap = "CONF %.2f" % conf
    c.text(x1 - c.tw(cap, 2), 30, cap, DIM, 2)
    c.text(x1 - c.tw("LAG (MS)", 2), h - 20, "LAG (MS)", DIM, 2)
    c.vtext(8, y0 + 8, "R", DIM, 2)
    return c


def plot_history(tr, tag="", w=FIG_W, h=FIG_W):
    """一个声道的基频 / 周期随时间变化，上下两幅（对齐参考的「历史频率」界面）。

    tr 是该声道的轨迹 [[t, f, conf], ...]，tag 是声道名。
    h 是这一段占的高度：双声道时传整图的一半，单声道时传整张。

    上：频率，对数轴（100–5000 Hz）。下：周期，线性轴，单位秒——周期就是
    1/频率，同一批点换种说法，所以不多算一遍，直接从基频轨迹换算。

    **面板位置跟着 h 走**。以前是写死的 58/318/386/664（768 的刻度），
    只有右边界和内联的页脚是相对的，于是 detail 一放大就只剩上半张：
    1024 空 30%、1280 空 44%、1536 空 53%，而服务端按 1024 token 回缩之后，
    1536 那张里的图反而比 768 档还小。现在按 h/768 等比换算。
    """
    tr = tr or []
    c = Canvas(w, h)
    k = float(h) / float(FIG_W)
    big = k >= 0.9
    x0, x1 = 120, w - 22
    c.text(x0, 8, _tag_title(tag, "FREQUENCY HISTORY"), TXT, 3 if big else 2)

    head = 56 if big else 32          # 标题行占高
    foot = 34                         # 页脚 + 底边距
    gap = 46                          # 两幅之间：x 轴标签 + 第二幅的标题
    ph = max(56.0, (h - head - foot - gap) / 2.0)
    ay0, ay1 = head, int(round(head + ph))
    by0, by1 = int(round(ay1 + gap)), max(int(round(ay1 + gap)) + 40, h - foot)

    ts = [float(t) for t, _f, _c in tr]
    fs = [float(f) for _t, f, _c in tr]
    t0, t1 = (ts[0], ts[-1]) if ts else (0.0, 1.0)
    if t1 - t0 < 1e-9:
        t1 = t0 + 1e-3
    t1 = _snap_time_end(t1, t0)

    def X(t):
        return x0 + (x1 - x0) * ((t - t0) / (t1 - t0))

    xt = _time_ticks(t0, t1)
    for xx, lab in [(X(v), lab) for v, lab in xt]:
        c.vline(xx, ay0, ay1, GRID)
        c.vline(xx, by0, by1, GRID)
    for a0, a1 in ((ay0, ay1), (by0, by1)):
        c.rect(x0, a0, x1, a1, AXIS)
        for v, lab in xt:
            c.ctext(X(v), a1 + 8, lab, DIM, 1)
    c.text(x1 - c.tw("TIME (S)", 2), by0 - 20, "TIME (S)", DIM, 2)

    # ---- 上：频率，对数轴，下限 100 Hz（25ms 窗能到 40Hz，画到 100 够用）----
    fl, fh = 100.0, 5000.0
    lg = math.log10
    c.text(x0 + 8, ay0 + 8, "FREQ (HZ)", DIM, 2)
    c.vtext(6, ay0 + 4, "FREQ", DIM, 1)

    def AF(f):
        f = min(max(f, fl), fh)
        return ay1 - (lg(f) - lg(fl)) / (lg(fh) - lg(fl)) * (ay1 - ay0)

    # 刻度只留 500 / 1000 / 5000：200 和 2000 对数轴上离 100 / 1000 太近
    for f in (500, 1000, 5000):
        yy = AF(f)
        c.hline(yy, x0, x1, GRID)
        c.text(x0 - c.tw("%d" % f, 1) - 10, yy - 7, "%d" % f, DIM, 1)

    # ---- 下：周期，线性轴（秒）----
    c.text(x0 + 8, by0 + 8, "PERIOD (S)", DIM, 2)
    c.vtext(6, by0 + 4, "PERIOD", DIM, 1)
    # 用 P98 定轴，别让一两个低八度跳点（周期翻倍）把整幅压到底部；
    # 超出轴顶的点贴边画，轴顶刻度仍如实标出来
    ps = [1.0 / f for f in fs if f > 0]
    p98 = float(np.percentile(ps, 98)) if ps else 0.006
    pstep = max(0.001, _nice_step(p98, 4))
    pmax = max(pstep, math.ceil(p98 / pstep) * pstep)

    def BP(p):
        # 线性轴、下限 0：周期只可能为正，0 起读数直观
        return by1 - min(max(p, 0.0), pmax) / pmax * (by1 - by0)

    v = 0.0
    while v <= pmax + 1e-12:
        yy = BP(v)
        c.hline(yy, x0, x1, GRID)
        lab = "%.3f" % v
        c.text(x0 - c.tw(lab, 1) - 10, yy - 7, lab, DIM, 1)
        v += pstep

    if not ts:
        c.ctext((x0 + x1) / 2.0, (ay0 + ay1) / 2.0, "NO PERIODIC SIGNAL", DIM, 2)
        return c

    # 散点而不是连线：帧与帧之间跳变很碎，连线会画成一团锯齿轮廓
    for t, f in zip(ts, fs):
        xx = X(t)
        yy = AF(f)
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                c.px(xx + a, yy + b, (255, 150, 40))
        yy = BP(1.0 / f if f > 0 else 0.0)
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                c.px(xx + a, yy + b, FG)

    if fs:
        med = float(np.median(fs))
        c.text(x0 + 8, by0 + 26, "%dMS WINDOWS  N=%d  MEDIAN %dHZ"
               % (round(1000.0 * (ts[1] - ts[0])) if len(ts) > 1 else 0, len(fs), med), DIM, 1)
    return c


def render(d, views, out_dir, stem="audio", size=FIG_W):
    """views: level / spectrum / spectrogram / peak / wave 的子集。

    每张图 size×size（默认 768）：上半 L、下半 R，中间一条分隔线。
    pitch / history 也一样分声道（以前混成单声道，见 engine 里的说明）；
    只有真正的单声道音轨才只画一张。
    size 就是 detail 档位换算出的边长（见 mcp_server 的 AUDIO_DETAIL_PX），
    画布和坐标轴等比放大；点、线、字都是按像素画的，放大后不会变细。
    返回 {view: png 路径}。
    """
    from pathlib import Path
    out = {}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    W = int(size) if size else FIG_W
    H = W
    sr = float(d.get("sample_rate") or 0)
    nyq = sr / 2.0 if sr > 0 else 24000.0
    tl = "L (MONO)" if d.get("mono") else "L"
    tr = "R (MONO)" if d.get("mono") else "R"
    for view in views:
        if view in ("pitch", "history"):
            # 两个声道各画一份，上下拼接，和 level / peak 一致。
            # 真·单声道音轨只画一张：ffmpeg 把同一路复制到两边，画两遍是浪费。
            half = H if d.get("mono") else H // 2
            if view == "pitch":
                top = plot_pitch(d["pitch"]["left"], tag=tl, w=W, h=half)
                bot = None if d.get("mono") else plot_pitch(d["pitch"]["right"], tag=tr, w=W, h=half)
            else:
                top = plot_history(d["pitch_track"]["left"], tag=tl, w=W, h=half)
                bot = None if d.get("mono") else plot_history(d["pitch_track"]["right"], tag=tr, w=W, h=half)
            if bot is None:
                out[view] = top.save(out_dir / (stem + "-" + view + ".png"))
            else:
                c = Canvas(W, H)
                c.blit(top.buf, 0, 0)
                c.blit(bot.buf, 0, H // 2)
                c.hline(H // 2, 0, W - 1, FG, 2)
                out[view] = c.save(out_dir / (stem + "-" + view + ".png"))
            continue
        if view == "level":
            top = plot_level(d["left"], d["_start"], d["_end"], tag=tl, w=W, h=H // 2)
            bot = plot_level(d["right"], d["_start"], d["_end"], tag=tr, w=W, h=H // 2)
        elif view == "spectrum":
            top = plot_spectrum(d["left"], tag=tl, resolution_hz=d["resolution_hz"],
                                fmax=nyq, w=W, h=H // 2)
            bot = plot_spectrum(d["right"], tag=tr, resolution_hz=d["resolution_hz"],
                                fmax=nyq, w=W, h=H // 2)
        elif view == "spectrogram":
            top = plot_spectrogram(d["left"], tag=tl, fmax=nyq, w=W, h=H // 2)
            bot = plot_spectrogram(d["right"], tag=tr, fmax=nyq, w=W, h=H // 2)
        elif view == "peak":
            top = plot_peak(d["left"], tag=tl, w=W, h=H // 2)
            bot = plot_peak(d["right"], tag=tr, w=W, h=H // 2)
        elif view == "wave":
            top = plot_wave(d["left"], tag=tl, w=W, h=H // 2)
            bot = plot_wave(d["right"], tag=tr, w=W, h=H // 2)
        else:
            continue
        c = Canvas(W, H)
        c.blit(top.buf, 0, 0)
        c.blit(bot.buf, 0, H // 2)
        c.hline(H // 2, 0, W - 1, FG, 2)
        out[view] = c.save(out_dir / (stem + "-" + view + ".png"))
    return out
