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


def plot_level(d, start, end, path, w=1024, h=512):
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
    c.text(x0, 14, "SOUND LEVEL", TXT, 3)
    bx = x1 - c.tw(big, 5) - 56
    c.text(bx, 12, big, FG, 5)
    c.text(bx + c.tw(big, 5) + 10, 38, "DB", DIM, 3)
    c.text(x0, 46, "TIME (S)", DIM, 2)
    c.text(x0 + 6, y0 + 6, "DB", DIM, 2)
    return c.save(path)


def plot_spectrum(d, path, w=1024, h=512):
    sp = d["spectrum"]
    f = np.asarray(sp["freqs"], dtype=float)
    m = np.asarray(sp["mag_db"], dtype=float)
    c = Canvas(w, h)
    x0, y0, x1, y1 = 96, 104, w - 26, h - 56
    # 横轴按整十倍频程取界，保证每个 decade 等宽（100/1000/10000 落在 1/4、1/2、3/4）
    fmin = 10.0
    fmax = 10.0 ** math.ceil(math.log10(max(40.0, float(f[-1]))))
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
    fr = fmin
    while fr <= fmax * 1.0001:
        xx = X(fr)
        c.vline(xx, y0, y1, GRID)
        c.ctext(xx, y1 + 12, "%g" % round(fr), DIM, 2)
        fr *= 10.0
    c.rect(x0, y0, x1, y1, AXIS)
    for i in range(1, len(f)):
        if f[i] < fmin:
            continue
        c.line(X(f[i - 1]), Y(m[i - 1]), X(f[i]), Y(m[i]), FG, 2)
    pk = sp["peak_freq_hz"]
    if pk > fmin:
        c.vline(X(pk), y0, y1, (255, 255, 255), 1)
    c.text(x0, 12, "SPECTRUM", TXT, 3)
    rows = [("PEAK FREQ", "%.2f HZ" % pk), ("NOTE", "%s" % sp["note"]),
            ("CENTS", "%+.2f" % sp["cents"]), ("RES", "%.2f HZ" % d["resolution_hz"])]
    colw = (x1 - x0) // 4
    for k, (lab, val) in enumerate(rows):
        xx = x0 + k * colw
        c.text(xx, 48, lab, DIM, 2)
        c.text(xx, 66, val, FG, 3)
    c.text(x1 - c.tw("FREQ (HZ)", 2), 14, "FREQ (HZ)", DIM, 2)
    c.vtext(8, y0 + 8, "FFT AMP (REL)", DIM, 2)
    return c.save(path)


def plot_spectrogram(d, path, w=1024, h=512):
    sg = d["spectrogram"]
    spec = np.asarray(sg["spec_db"], dtype=float)   # (nfreq, ntime)
    times = np.asarray(sg["times"], dtype=float)
    freqs = np.asarray(sg["freqs"], dtype=float)
    c = Canvas(w, h)
    x0, x1 = 96, w - 26
    y0, y1 = 64, 316
    H, W = y1 - y0, x1 - x0
    nt, nf = spec.shape[1], spec.shape[0]
    if nt < 2:
        spec = np.repeat(spec, 2, axis=1); nt = 2
    yt = np.linspace(nt - 1, 0, H).astype(int)      # 时间向上
    xf = np.linspace(0, nf - 1, W).astype(int)
    v = np.clip((spec.T[yt][:, xf] + 70.0) / 70.0, 0, 1)
    c.blit(_cmap(v), x0, y0)
    c.rect(x0, y0, x1, y1, AXIS)
    c.text(x0, 12, "SPECTROGRAM", TXT, 3)
    c.text(x1 - c.tw("FREQ (HZ)", 2), 14, "FREQ (HZ)", DIM, 2)
    c.vtext(8, y0 + 8, "TIME (S)", DIM, 2)
    for k in range(5):
        fr = freqs[-1] * k / 4.0
        xx = x0 + (x1 - x0) * k / 4.0
        lab = "%d" % fr
        c.ctext(xx, y1 + 10, lab, DIM, 2)
    for k in range(5):
        frac = k / 4.0
        tt = times[0] + (times[-1] - times[0]) * frac
        yy = y1 - (y1 - y0) * frac
        c.hline(yy, x0, x1, (0, 0, 0), 1)
        lab = "%.1f" % tt
        c.text(x0 - c.tw(lab) - 10, yy - 7, lab, DIM, 2)

    # 峰值频率轨迹
    p0, p1 = 380, h - 44
    tr = sg["peak_track"]
    fx = float(freqs[-1]) if freqs.size else 1.0
    c.text(x0, p0 - 22, "PEAK FREQ TRACK", DIM, 2)
    c.rect(x0, p0, x1, p1, AXIS)
    for frac in (0.25, 0.5, 0.75):
        db = fx * frac
        yy = p1 - frac * (p1 - p0)
        c.hline(yy, x0, x1, GRID)
        lab = "%d" % db
        c.text(x0 - c.tw(lab) - 10, yy - 7, lab, DIM, 2)
    if tr:
        span = (times[-1] - times[0]) if len(times) > 1 else 1.0
        for tt, ff in tr:
            frac = (tt - times[0]) / max(1e-9, span)
            xx = x0 + (x1 - x0) * min(1.0, max(0.0, frac))
            yy = p1 - (min(ff, fx) / fx) * (p1 - p0)
            for a2 in range(-2, 3):
                for b2 in range(-2, 3):
                    if a2 * a2 + b2 * b2 <= 4:
                        c.px(xx + a2, yy + b2, FG)
    c.text(x1 - c.tw("TIME (S)", 2), h - 24, "TIME (S)", DIM, 2)
    return c.save(path)


def render(d, views, out_dir, stem="audio"):
    """views: level / spectrum / spectrogram 的子集。返回 {view: png 路径}。"""
    from pathlib import Path
    out = {}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if "level" in views:
        out["level"] = plot_level(d, d["_start"], d["_end"], out_dir / (stem + "-level.png"))
    if "spectrum" in views:
        out["spectrum"] = plot_spectrum(d, out_dir / (stem + "-spectrum.png"))
    if "spectrogram" in views:
        out["spectrogram"] = plot_spectrogram(d, out_dir / (stem + "-spectrogram.png"))
    return out
