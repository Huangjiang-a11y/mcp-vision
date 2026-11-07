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


def plot_level(d, start, end, tag="", w=1024, h=512):
    lv = d["level"]
    t = np.asarray(lv["times"], dtype=float)
    v = np.asarray(lv["values"], dtype=float)
    step = max(1, len(v) // 900)
    t, v = t[::step], v[::step]
    c = Canvas(w, h)
    x0, y0, x1, y1 = 92, 76, w - 26, h - 50
    lo = math.floor(min(v.min(), lv["mean"] - 6) / 5.0) * 5.0
    hi = math.ceil(max(v.max(), lv["mean"] + 6) / 5.0) * 5.0
    if hi - lo < 6:
        hi = lo + 6
    n = len(v) if len(v) > 1 else 2
    def X(i): return x0 + (x1 - x0) * (i / float(n - 1))
    def Y(db): return y1 - (db - lo) / (hi - lo) * (y1 - y0)
    for k in range(5):
        db = lo + (hi - lo) * k / 4.0
        yy = Y(db)
        c.hline(yy, x0, x1, GRID)
        lab = _nice_db(round(db, 1))
        c.text(x0 - c.tw(lab) - 10, yy - 7, lab, DIM, 2)
    for k in range(5):
        frac = k / 4.0
        tt = start + (end - start) * frac
        xx = x0 + (x1 - x0) * frac
        c.vline(xx, y0, y1, GRID)
        lab = "%.1f" % tt
        c.ctext(xx, y1 + 12, lab, DIM, 2)
    c.rect(x0, y0, x1, y1, AXIS)
    for i in range(1, len(v)):
        c.line(X(i - 1), Y(v[i - 1]), X(i), Y(v[i]), FG, 2)
    big = "%.1f" % lv["db"]
    c.text(x0, 14, _tag_title(tag, "SOUND LEVEL"), TXT, 3)
    bx = x1 - c.tw(big, 5) - 56
    c.text(bx, 12, big, FG, 5)
    c.text(bx + c.tw(big, 5) + 10, 38, "DB", DIM, 3)
    c.text(x0, 46, "TIME (S)", DIM, 2)
    c.text(x0 + 6, y0 + 6, "DB", DIM, 2)
    _caption(c, d, x1, 56)
    return c


def plot_spectrum(d, tag="", resolution_hz=0.0, w=1024, h=512):
    sp = d["spectrum"]
    f = np.asarray(sp["freqs"], dtype=float)
    m = np.asarray(sp["mag_db"], dtype=float)
    c = Canvas(w, h)
    x0, y0, x1, y1 = 96, 104, w - 26, h - 56
    # 横轴对数，尾端固定 24000
    fmin = 10.0
    fmax = 24000.0
    dmin = max(-80.0, float(np.min(m)))
    if dmin > -30:
        dmin = -30.0
    def X(fr): return x0 + math.log10(max(fr, fmin) / fmin) / math.log10(fmax / fmin) * (x1 - x0)
    def Y(db):
        return y1 - (min(0.0, max(dmin, db)) - dmin) / (0.0 - dmin) * (y1 - y0)
    # 纵轴按「相对幅值」标，与参考图一致（对数刻度：1 / 0.1 / 0.01 / 0.001）
    for amp in (1.0, 0.1, 0.01, 0.001, 0.0001):
        db = 20.0 * math.log10(amp)
        if db < dmin:
            continue
        yy = Y(db)
        c.hline(yy, x0, x1, GRID)
        lab = ("%g" % amp)
        c.text(x0 - c.tw(lab) - 10, yy - 7, lab, DIM, 2)
    for fr in (50, 100, 500, 1000, 5000, 10000, 24000):
        xx = X(fr)
        c.vline(xx, y0, y1, GRID)
        c.ctext(xx, y1 + 12, "%g" % fr, DIM, 2)
    c.rect(x0, y0, x1, y1, AXIS)
    for i in range(1, len(f)):
        if f[i] < fmin:
            continue
        c.line(X(f[i - 1]), Y(m[i - 1]), X(f[i]), Y(m[i]), FG, 2)
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
    c.vtext(8, y0 + 8, "FFT AMP (REL)", DIM, 2)
    return c


def _nice_step(span, want=5):
    """把跨度切成 want 段左右，步长取 1/2/2.5/5 的十倍数，刻度才整齐。"""
    raw = max(1e-9, span / float(max(1, want - 1)))
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


def _time_ticks(t0, t1, want=5):
    """窗口内的等距刻度，落在 step 的整数倍上，不出现 6.4 / 6.9 这种歪数。"""
    step = _nice_step(t1 - t0, want)
    k = math.ceil(t0 / step - 1e-9)
    ticks, txt = [], "%.1f" if step >= 0.1 else "%.2f"
    while True:
        v = k * step
        if v > t1 + 1e-9:
            break
        ticks.append((v, txt % v))
        k += 1
    return ticks


def plot_spectrogram(d, tag="", w=1024, h=512):
    """时频图：横轴对数频率 20..24000，纵轴时间（向上）。"""
    sg = d["spectrogram"]
    spec = np.asarray(sg["spec_db"], dtype=float)   # (nfreq, ntime)
    times = np.asarray(sg["times"], dtype=float)
    freqs = np.asarray(sg["freqs"], dtype=float)
    c = Canvas(w, h)
    x0, x1 = 96, w - 26
    y0, y1 = 64, h - 58
    H, W = y1 - y0, x1 - x0
    nt, nf = spec.shape[1], spec.shape[0]
    if nt < 2:
        spec = np.repeat(spec, 2, axis=1); nt = 2
    fmin, fmax = 20.0, 24000.0            # 尾端固定 24k，采样率不够的部分留黑
    def X(fr): return x0 + math.log10(max(fr, fmin) / fmin) / math.log10(fmax / fmin) * (x1 - x0)
    fcols = fmin * (fmax / fmin) ** (np.arange(W) / float(max(1, W - 1)))
    yt = np.linspace(nt - 1, 0, H).astype(int)      # 时间向上
    # 对数轴下每个像素覆盖的频率宽度差很多，直接取最近 bin 会一格一格的；
    # 在对数频率上做线性插值，低中频才连续
    lf = np.log(np.maximum(freqs, 1e-3))
    lfq = np.log(fcols)
    mat = np.empty((H, W), dtype=np.float64)
    src = spec[:, yt]
    for j in range(H):
        mat[j, :] = np.interp(lfq, lf, src[:, j])
    mat[:, fcols > freqs[-1]] = -70.0               # 超过 Nyquist 的部分没有数据
    v = np.clip((mat + 70.0) / 70.0, 0, 1)
    c.blit(_cmap(v), x0, y0)
    c.rect(x0, y0, x1, y1, AXIS)
    c.text(x0, 12, _tag_title(tag, "SPECTROGRAM"), TXT, 3)
    c.text(x1 - c.tw("FREQ (HZ, LOG)", 2), 14, "FREQ (HZ, LOG)", DIM, 2)
    c.vtext(8, y0 + 8, "TIME (S)", DIM, 2)
    ticks = [f for f in (50, 100, 500, 1000, 5000, 10000, 24000) if f < fmax]
    if not ticks or fmax > ticks[-1] * 1.05:
        ticks.append(fmax)
    for fr in ticks:
        xx = X(fr)
        c.vline(xx, y0, y1, (0, 0, 0), 1)
        c.ctext(xx, y1 + 10, "%g" % fr, DIM, 2)
    t0, t1 = (float(times[0]), float(times[-1])) if times.size > 1 else (0.0, 1.0)
    for val, lab in _time_ticks(t0, t1):
        frac = (val - t0) / max(1e-9, t1 - t0)
        yy = y1 - (y1 - y0) * frac
        c.hline(yy, x0, x1, (0, 0, 0), 1)
        c.text(x0 - c.tw(lab) - 10, yy - 7, lab, DIM, 2)
    return c


def plot_peak(d, tag="", w=1024, h=512):
    """峰值频率轨迹单独一张：横轴时间，纵轴按轨迹自身范围自适应。"""
    sg = d["spectrogram"]
    times = np.asarray(sg["times"], dtype=float)
    tr = sg["peak_track"]
    c = Canvas(w, h)
    x0, x1 = 110, w - 26
    y0, y1 = 88, h - 76
    fs = [float(f) for _, f in tr] or [1.0]
    lo, hi = min(fs), max(fs)
    step = _nice_step(max(hi - lo, 1.0), 5)
    lo = max(0.0, math.floor(lo / step) * step)
    hi = math.ceil(hi / step) * step
    if hi - lo < step:
        hi = lo + step
    def Y(f): return y1 - (f - lo) / (hi - lo) * (y1 - y0)
    c.text(x0, 12, _tag_title(tag, "PEAK FREQ TRACK"), TXT, 3)
    c.vtext(8, y0 + 8, "FREQ (HZ)", DIM, 2)
    v = lo
    while v <= hi + 1e-9:
        yy = Y(v)
        c.hline(yy, x0, x1, GRID)
        lab = "%g" % round(v, 2)
        c.text(x0 - c.tw(lab) - 10, yy - 7, lab, DIM, 2)
        v += step
    c.rect(x0, y0, x1, y1, AXIS)
    t0, t1 = (float(times[0]), float(times[-1])) if times.size > 1 else (0.0, 1.0)
    for val, lab in _time_ticks(t0, t1):
        frac = (val - t0) / max(1e-9, t1 - t0)
        xx = x0 + (x1 - x0) * frac
        c.vline(xx, y0, y1, GRID)
        c.ctext(xx, y1 + 12, lab, DIM, 2)
    if tr:
        span = (t1 - t0) if t1 > t0 else 1.0
        pts = []
        for tt, ff in tr:
            frac = (tt - t0) / span
            pts.append((x0 + (x1 - x0) * min(1.0, max(0.0, frac)), Y(ff)))
        for i in range(1, len(pts)):
            c.line(pts[i - 1][0], pts[i - 1][1], pts[i][0], pts[i][1], GRID, 1)
        for xx, yy in pts:
            for a2 in range(-2, 3):
                for b2 in range(-2, 3):
                    if a2 * a2 + b2 * b2 <= 4:
                        c.px(xx + a2, yy + b2, FG)
    c.text(x1 - c.tw("TIME (S)", 2), h - 30, "TIME (S)", DIM, 2)
    _caption(c, d, x1, 40)
    return c


def render(d, views, out_dir, stem="audio"):
    """views: level / spectrum / spectrogram 的子集。

    每张图 1024x1024：上半 L、下半 R，中间一条分隔线。
    返回 {view: png 路径}。
    """
    from pathlib import Path
    out = {}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    W, H = 1024, 1024
    tl = "L (MONO)" if d.get("mono") else "L"
    tr = "R (MONO)" if d.get("mono") else "R"
    for view in views:
        if view == "level":
            top = plot_level(d["left"], d["_start"], d["_end"], tag=tl)
            bot = plot_level(d["right"], d["_start"], d["_end"], tag=tr)
        elif view == "spectrum":
            top = plot_spectrum(d["left"], tag=tl, resolution_hz=d["resolution_hz"])
            bot = plot_spectrum(d["right"], tag=tr, resolution_hz=d["resolution_hz"])
        elif view == "spectrogram":
            top = plot_spectrogram(d["left"], tag=tl)
            bot = plot_spectrogram(d["right"], tag=tr)
        elif view == "peak":
            top = plot_peak(d["left"], tag=tl)
            bot = plot_peak(d["right"], tag=tr)
        else:
            continue
        c = Canvas(W, H)
        c.blit(top.buf, 0, 0)
        c.blit(bot.buf, 0, 512)
        c.hline(512, 0, W - 1, FG, 2)
        out[view] = c.save(out_dir / (stem + "-" + view + ".png"))
    return out
