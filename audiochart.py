"""把音频剖析结果画成 PNG。纯 numpy 光栅化 + engine._write_png，不依赖绘图库。"""
import math
import numpy as np

from engine import _write_png

FG = (255, 150, 60)
GRID = (58, 58, 66)
AXIS = (120, 120, 130)
TXT = (232, 232, 238)
DIM = (140, 140, 150)

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
    r = np.interp(v, stops, [0, 55, 190, 255, 255])
    g = np.interp(v, stops, [0, 10, 80, 180, 255])
    b = np.interp(v, stops, [0, 3, 22, 80, 255])
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def _nice_db(v):
    return ("%d" % v) if abs(v - round(v)) < 1e-9 else ("%.1f" % v)


def plot_level(d, start, end, path, w=900, h=420):
    lv = d["level"]
    t = np.asarray(lv["times"], dtype=float)
    v = np.asarray(lv["values"], dtype=float)
    step = max(1, len(v) // 800)
    t, v = t[::step], v[::step]
    c = Canvas(w, h)
    x0, y0, x1, y1 = 110, 96, w - 40, h - 62
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
        c.text(x0 - c.tw(lab) - 12, yy - 7, lab, DIM, 2)
    for k in range(5):
        frac = k / 4.0
        tt = start + (end - start) * frac
        xx = x0 + (x1 - x0) * frac
        c.vline(xx, y0, y1, GRID)
        lab = "%.1f" % tt
        c.text(xx - c.tw(lab) / 2, y1 + 12, lab, DIM, 2)
    c.rect(x0, y0, x1, y1, AXIS)
    for i in range(1, len(v)):
        c.line(X(i - 1), Y(v[i - 1]), X(i), Y(v[i]), FG, 2)
    c.text(x0, 18, "SOUND LEVEL", TXT, 3)
    big = "%.1f" % lv["db"]
    c.text(x0, 48, big, FG, 5)
    c.text(x0 + c.tw(big, 5) + 10, 74, "DB", DIM, 3)
    c.text(x1 - c.tw("TIME (S)", 2), h - 26, "TIME (S)", DIM, 2)
    c.text(x0 + 8, y0 + 6, "DB", DIM, 2)
    return c.save(path)


def plot_spectrum(d, path, w=900, h=480):
    sp = d["spectrum"]
    f = np.asarray(sp["freqs"], dtype=float)
    m = np.asarray(sp["mag_db"], dtype=float)
    c = Canvas(w, h)
    x0, y0, x1, y1 = 96, 140, w - 34, h - 92
    fmin, fmax = 20.0, max(40.0, float(f[-1]))
    dmin = max(-80.0, float(np.min(m)))
    if dmin > -30:
        dmin = -30.0
    def X(fr): return x0 + math.log10(max(fr, fmin) / fmin) / math.log10(fmax / fmin) * (x1 - x0)
    def Y(db):
        return y1 - (min(0.0, max(dmin, db)) - dmin) / (0.0 - dmin) * (y1 - y0)
    for db in (-60, -40, -20, 0):
        if db < dmin:
            continue
        yy = Y(db)
        c.hline(yy, x0, x1, GRID)
        lab = "%d" % db
        c.text(x0 - c.tw(lab) - 12, yy - 7, lab, DIM, 2)
    for fr in (100, 1000, 10000):
        if fr > fmax:
            continue
        xx = X(fr)
        c.vline(xx, y0, y1, GRID)
        lab = "%g" % fr
        c.text(xx - c.tw(lab) / 2, y1 + 12, lab, DIM, 2)
    c.rect(x0, y0, x1, y1, AXIS)
    for i in range(1, len(f)):
        if f[i] < fmin:
            continue
        c.line(X(f[i - 1]), Y(m[i - 1]), X(f[i]), Y(m[i]), FG, 2)
    pk = sp["peak_freq_hz"]
    if pk > fmin:
        c.vline(X(pk), y0, y1, (255, 235, 190), 1)
        c.text(X(pk) + 6, y0 + 6, "PEAK", (255, 235, 190), 2)
    c.text(x0, 14, "SPECTRUM", TXT, 3)
    # 四列统计：标签在上，值在下，互不重叠
    rows = [("PEAK FREQ", "%.2f HZ" % pk), ("NOTE", "%s" % sp["note"]),
            ("CENTS", "%+.2f" % sp["cents"]), ("RES", "%.2f HZ" % d["resolution_hz"])]
    colw = (x1 - x0) // 4
    for k, (lab, val) in enumerate(rows):
        xx = x0 + k * colw
        c.text(xx, 52, lab, DIM, 2)
        c.text(xx, 72, val, FG, 3)
    c.text(x1 - c.tw("FREQ (HZ)", 2), h - 34, "FREQ (HZ)", DIM, 2)
    c.text(x0 + 8, y0 + 6, "DB", DIM, 2)
    return c.save(path)


def plot_spectrogram(d, path, w=900, h=620):
    sg = d["spectrogram"]
    spec = np.asarray(sg["spec_db"], dtype=float)   # (nfreq, ntime)
    times = np.asarray(sg["times"], dtype=float)
    freqs = np.asarray(sg["freqs"], dtype=float)
    c = Canvas(w, h)
    x0, x1 = 100, w - 34
    y0, y1 = 58, 344
    H, W = y1 - y0, x1 - x0
    nt, nf = spec.shape[1], spec.shape[0]
    if nt < 2:
        spec = np.repeat(spec, 2, axis=1); nt = 2
    yt = np.linspace(nt - 1, 0, H).astype(int)      # 时间向上
    xf = np.linspace(0, nf - 1, W).astype(int)
    v = np.clip((spec.T[yt][:, xf] + 70.0) / 70.0, 0, 1)
    c.blit(_cmap(v), x0, y0)
    c.rect(x0, y0, x1, y1, AXIS)
    c.text(x0, 14, "SPECTROGRAM", TXT, 3)
    c.text(x1 - c.tw("FREQ (HZ)", 2), 24, "FREQ (HZ)", DIM, 2)
    for k in range(5):
        fr = freqs[-1] * k / 4.0
        xx = x0 + (x1 - x0) * k / 4.0
        lab = "%d" % fr
        c.text(xx - c.tw(lab) / 2, y1 + 12, lab, DIM, 2)
    for k in range(5):
        frac = k / 4.0
        tt = times[0] + (times[-1] - times[0]) * frac
        yy = y1 - (y1 - y0) * frac
        c.hline(yy, x0, x1, (0, 0, 0), 1)
        lab = "%.1f" % tt
        c.text(x0 - c.tw(lab) - 12, yy - 7, lab, DIM, 2)
    c.text(12, 46, "TIME (S)", DIM, 2)

    # 峰值频率轨迹
    p0, p1 = 418, h - 58
    tr = sg["peak_track"]
    fx = float(freqs[-1]) if freqs.size else 1.0
    c.text(x0, p0 - 26, "PEAK FREQ TRACK", DIM, 2)
    c.rect(x0, p0, x1, p1, AXIS)
    for frac in (0.25, 0.5, 0.75):
        db = fx * frac
        yy = p1 - frac * (p1 - p0)
        c.hline(yy, x0, x1, GRID)
        lab = "%d" % db
        c.text(x0 - c.tw(lab) - 12, yy - 7, lab, DIM, 2)
    if tr:
        span = (times[-1] - times[0]) if len(times) > 1 else 1.0
        for tt, ff in tr:
            frac = (tt - times[0]) / max(1e-9, span)
            xx = x0 + (x1 - x0) * min(1.0, max(0.0, frac))
            yy = p1 - (min(ff, fx) / fx) * (p1 - p0)
            for a in range(-2, 3):
                for b in range(-2, 3):
                    if a * a + b * b <= 4:
                        c.px(xx + a, yy + b, FG)
    c.text(x1 - c.tw("TIME (S)", 2), h - 34, "TIME (S)", DIM, 2)
    return c.save(path)


def render(d, views, out_dir, stem="audio"):
    """views: level / spectrum / spectrogram 的子集。返回 {view: png 路径}。"""
    from pathlib import Path
    out = {}
    out_dir = Path(out_dir)
    if "level" in views:
        out["level"] = plot_level(d, d["_start"], d["_end"], out_dir / (stem + "-level.png"))
    if "spectrum" in views:
        out["spectrum"] = plot_spectrum(d, out_dir / (stem + "-spectrum.png"))
    if "spectrogram" in views:
        out["spectrogram"] = plot_spectrogram(d, out_dir / (stem + "-spectrogram.png"))
    return out
