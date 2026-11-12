"""PP-OCRv4 的 Python 实现：DB 检测 + CTC 识别。
依赖：onnxruntime + numpy + ffmpeg（读图）。不依赖 OpenCV / Pillow / pyclipper。
模型：路径由 MVV_OCR_DET / MVV_OCR_REC / MVV_OCR_KEYS 指定（默认 models/ 下的 PP-OCRv4）
"""
from collections import Counter
import os, re, sys, json, subprocess
from pathlib import Path
import threading
import numpy as np

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("MVV_MODEL_DIR") or (APP_DIR / "models"))
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DET_MAX_SIDE = int(os.environ.get("DET_MAX_SIDE", "720"))
# 识别输入的最大宽度：字幕行短，320 够；手写长行降到 256/192 明显掉字，别乱调
REC_MAX_W = int(os.environ.get("MVV_REC_MAX_W", "320"))
REC_H = int(os.environ.get("MVV_REC_H", "48"))
DET_TH = float(os.environ.get("DET_TH", "0.2"))
# 窗口短于这个秒数就不开进程池：启动 4 个进程（各自加载模型）比省下的时间还贵
PARALLEL_MIN_SPAN = float(os.environ.get("MVV_OCR_PARALLEL_MIN", "30"))
# 字幕取帧单独一条开关：默认 2 进程并行（MVV_OCR_WORKERS），每进程 2 条解码线程
# = 4 条在跑，再往上就是跟 ORT 抢核。engine.py 的工具那条默认 3。
FFMPEG_THREADS = int(os.environ.get("MVV_OCR_FFMPEG_THREADS", "2"))


def _providers():
    want = os.environ.get("MVV_ORT_PROVIDERS", "cpu").lower().split(",")
    import onnxruntime as ort
    avail = ort.get_available_providers()
    m = {"cpu": "CPUExecutionProvider", "xnnpack": "XnnpackExecutionProvider", "nnapi": "NnapiExecutionProvider"}
    out = [m[w] for w in want if m.get(w) in avail]
    return out or ["CPUExecutionProvider"]


# ---------- 读图 ----------
def load_rgb(path, long_side=None):
    """用 ffmpeg 解成 RGB numpy 数组 (H,W,3) uint8。"""
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, timeout=60)
    w, h = [int(x) for x in probe.stdout.strip().split(",")[:2]]
    if long_side and max(w, h) > long_side:
        s = long_side / max(w, h)
        w, h = max(1, int(w * s)), max(1, int(h * s))
    p = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(path), "-frames:v", "1", "-vf", "scale=%d:%d" % (w, h),
        "-pix_fmt", "rgb24", "-f", "rawvideo", "-"], capture_output=True, timeout=120)
    if p.returncode != 0:
        raise RuntimeError((p.stderr or b"").decode("utf-8", "replace")[:300])
    buf = p.stdout[: w * h * 3]
    return np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3).copy()


def resize_bilinear(img, W, H):
    h, w = img.shape[:2]
    if (h, w) == (H, W):
        return img
    ys = np.clip((np.arange(H) + 0.5) * h / H - 0.5, 0, h - 1)
    xs = np.clip((np.arange(W) + 0.5) * w / W - 0.5, 0, w - 1)
    y0 = np.floor(ys).astype(np.int32); y1 = np.minimum(y0 + 1, h - 1)
    x0 = np.floor(xs).astype(np.int32); x1 = np.minimum(x0 + 1, w - 1)
    wy = (ys - y0)[:, None, None].astype(np.float32)
    wx = (xs - x0)[None, :, None].astype(np.float32)
    imgf = img.astype(np.float32)
    top = imgf[y0][:, x0] * (1 - wx) + imgf[y0][:, x1] * wx
    bot = imgf[y1][:, x0] * (1 - wx) + imgf[y1][:, x1] * wx
    return np.clip(top * (1 - wy) + bot * wy, 0, 255).astype(np.uint8)


class PPOCR:
    def __init__(self, model_dir=None, providers=None):
        md = Path(model_dir or MODEL_DIR)

        def pick(var, name):
            # 显式传了 model_dir 就听它的；否则环境变量优先，文件名不限
            if model_dir is None:
                v = (os.environ.get(var) or "").strip()
                if v:
                    return Path(v).expanduser()
            return md / name

        det = pick("MVV_OCR_DET", "ch_PP-OCRv4_det_infer.onnx")
        rec = pick("MVV_OCR_REC", "ch_PP-OCRv4_rec_infer.onnx")
        dic = pick("MVV_OCR_KEYS", "ppocr_keys_v1.txt")
        for f in (det, rec, dic):
            if not f.exists():
                raise FileNotFoundError(
                    "缺模型文件: %s（设 MVV_OCR_DET / MVV_OCR_REC / MVV_OCR_KEYS，见 INSTALL.md）" % f)
        self.dict = []
        for line in dic.read_text(encoding="utf-8").splitlines():
            if line.endswith("\r"):
                line = line[:-1]
            self.dict.append(line)
        while self.dict and self.dict[-1] == "":
            self.dict.pop()

        self.providers = providers or _providers()
        self._paths = (str(det), str(rec), str(dic))
        self._rec = None  # 懒加载：det 框出字了才建，纯画面不花这份内存

        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(os.environ.get("MVV_ORT_THREADS", "2"))
        so.log_severity_level = 3
        self.det = ort.InferenceSession(str(det), so, providers=self.providers)

    @property
    def rec(self):
        if self._rec is None:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = int(os.environ.get("MVV_ORT_THREADS", "2"))
            so.log_severity_level = 3
            self._rec = ort.InferenceSession(self._paths[1], so, providers=self.providers)
        return self._rec

    # ---- det ----
    def detect(self, rgb, thresh=DET_TH, max_side=DET_MAX_SIDE):
        H0, W0 = rgb.shape[:2]
        scale = min(1.0, max_side / max(W0, H0))
        W = max(32, 32 * int(round(W0 * scale / 32)))
        H = max(32, 32 * int(round(H0 * scale / 32)))
        small = resize_bilinear(rgb, W, H).astype(np.float32) / 255.0
        small = (small - MEAN) / STD
        x = np.transpose(small, (2, 0, 1))[None].astype(np.float32)
        name = self.det.get_inputs()[0].name
        prob = self.det.run(None, {name: x})[0]
        prob = np.asarray(prob).reshape(H, W)
        mask = prob > thresh
        boxes = self._connected_boxes(mask, W, H)
        sx, sy = W0 / W, H0 / H
        out = []
        for (minx, miny, maxx, maxy, count) in boxes:
            bw, bh = maxx - minx + 1, maxy - miny + 1
            if count < 30 or bw < 8 or bh < 8:
                continue
            if bw / bh > 60 or bh / bw > 5:
                continue
            pad = int(round(bh * 0.5))
            x0 = max(0, minx - pad); y0 = max(0, miny - pad)
            x1 = min(W - 1, maxx + pad); y1 = min(H - 1, maxy + pad)
            out.append((int(round(x0 * sx)), int(round(y0 * sy)),
                        int(round(x1 * sx)), int(round(y1 * sy))))
        out.sort(key=lambda b: (b[1], b[0]))
        return out

    @staticmethod
    def _connected_boxes(mask, W, H):
        """4 邻域连通域 → bbox。行内 run + 并查集：逐像素 BFS 在 720x96 上要 34ms，
        这里只要 3ms 左右，900 帧能省十几秒。"""
        rows = []                 # 每行 [(x0, x1, run_id), ...]
        runs = []                 # run: [y, x0, x1, run_id]
        for y in range(H):
            row = mask[y]
            cur = []
            if row.any():
                d = np.diff(row.astype(np.int8), prepend=np.int8(0), append=np.int8(0))
                for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
                    rid = len(runs)
                    runs.append([y, int(a), int(b) - 1])
                    cur.append((int(a), int(b) - 1, rid))
            rows.append(cur)

        parent = list(range(len(runs)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i, j):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri

        # 相邻两行的 run 只要 x 区间相交就是同一个连通域
        for y in range(1, H):
            prev, cur = rows[y - 1], rows[y]
            i = j = 0
            while i < len(prev) and j < len(cur):
                a0, b0, r0 = prev[i]
                a1, b1, r1 = cur[j]
                if b0 < a1:
                    i += 1
                elif b1 < a0:
                    j += 1
                else:
                    union(r0, r1)
                    if b0 <= b1:
                        i += 1
                    else:
                        j += 1

        agg = {}
        for rid, (y, a, b) in enumerate(runs):
            r = find(rid)
            cnt = b - a + 1
            e = agg.get(r)
            if e is None:
                agg[r] = [a, y, b, y, cnt]
            else:
                if a < e[0]: e[0] = a
                if y < e[1]: e[1] = y
                if b > e[2]: e[2] = b
                if y > e[3]: e[3] = y
                e[4] += cnt
        return [(e[0], e[1], e[2], e[3], e[4]) for e in agg.values()]

    # ---- rec ----
    def recognize(self, rgb, box):
        x0, y0, x1, y1 = box
        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(rgb.shape[1] - 1, x1); y1 = min(rgb.shape[0] - 1, y1)
        crop = rgb[y0:y1+1, x0:x1+1]
        if crop.size == 0:
            return "", 0.0
        bw, bh = crop.shape[1], crop.shape[0]
        target_h = REC_H
        target_w = int(max(16, min(REC_MAX_W, round((bw / max(bh, 1)) * target_h))))
        resized = resize_bilinear(crop, target_w, target_h).astype(np.float32) / 255.0 - 0.5
        x = np.transpose(resized, (2, 0, 1))[None].astype(np.float32)
        name = self.rec.get_inputs()[0].name
        preds = self.rec.run(None, {name: x})[0]
        preds = np.asarray(preds)[0]  # [T, C]
        T, C = preds.shape
        best = preds.argmax(axis=1)
        bestv = preds.max(axis=1)
        chars = []
        conf_sum = 0.0
        conf_n = 0
        prev = -1
        for t in range(T):
            b = int(best[t])
            if b != 0 and b != prev:
                ci = b - 1
                chars.append(self.dict[ci] if 0 <= ci < len(self.dict) else "")
                conf_sum += float(bestv[t]); conf_n += 1
            prev = b
        return "".join(chars), (conf_sum / conf_n if conf_n else 0.0)

    def ocr_image(self, rgb, thresh=DET_TH, max_side=DET_MAX_SIDE, min_conf=0.0):
        boxes = self.detect(rgb, thresh=thresh, max_side=max_side)
        results = []
        for b in boxes:
            try:
                text, conf = self.recognize(rgb, b)
            except Exception:
                continue
            if text.strip() and conf >= min_conf:
                results.append({"box": list(b), "text": text, "conf": round(conf, 4)})
        return results


_CACHE = {}


_CACHE_LOCK = threading.Lock()


def get_ppocr(model_dir=None, providers=None):
    """懒加载 + 复用：第一次真正用到 OCR 时建会话，之后一直复用。

    并发调用时必须双检：两个线程同时进来会各建一个会话，手机上直接爆内存。
    """
    if model_dir is None and providers is None:
        if "default" in _CACHE:
            return _CACHE["default"]
        with _CACHE_LOCK:
            if "default" not in _CACHE:
                _CACHE["default"] = PPOCR()
            return _CACHE["default"]
    return PPOCR(model_dir=model_dir, providers=providers)


def ocr_file(path, **kw):
    rgb = load_rgb(path, long_side=kw.pop("long_side", None))
    return get_ppocr(**kw).ocr_image(rgb)


if __name__ == "__main__":
    import time
    if len(sys.argv) < 2:
        print("用法: python ocr.py <图片> [--json]")
        sys.exit(1)
    t0 = time.time()
    rgb = load_rgb(sys.argv[1])
    engine = PPOCR()
    print("providers:", engine.providers, "| 图 %dx%d" % (rgb.shape[1], rgb.shape[0]), file=sys.stderr)
    res = engine.ocr_image(rgb)
    dt = time.time() - t0
    if "--json" in sys.argv:
        print(json.dumps({"count": len(res), "seconds": round(dt, 2), "results": res}, ensure_ascii=False, indent=2))
    else:
        print("识别 %d 个文本框，耗时 %.2fs" % (len(res), dt))
        for r in res:
            print("  [%d,%d] %.2f | %s" % (r["box"][0], r["box"][1], r["conf"], r["text"]))


# ---------- 加速：engine 复用 + 字幕带 + 变帧跳过 ----------

def _tc(seconds):
    """秒 -> MM:SS。只到秒：OCR 的采样步长本来就是 1.0s，ms 位恒为 000，
    补上去只是伪造精度，还每段多花 4 个 token。"""
    s = max(0.0, float(seconds))
    t = int(s + 0.5)
    return "%02d:%02d" % (t // 60, t % 60)


def get_engine(**kw):
    """旧名字，行为与 get_ppocr 一致：同一个实例，别维护两份缓存。"""
    return get_ppocr(**kw)


def _probe_wh(path):
    p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, timeout=60)
    w, h = p.stdout.strip().split(",")[:2]
    return int(w), int(h)


def _probe_duration(path):
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", str(path)], capture_output=True, text=True, timeout=60)
    try:
        return float(p.stdout.strip())
    except ValueError:
        return 0.0


def _stream_frames(media, start=0.0, end=None, step=1.0, width=None, crop=None):
    """ffmpeg 管道流出帧，逐帧 yield (i, t, rgb ndarray)。不落盘。

    crop: (y0_ratio, y1_ratio)，给了就只出带内；不给出全帧。
    width: None（默认）按源分辨率出，裁完直接给，不做缩放；给了才缩到这个宽度。
    """
    w0, h0 = _probe_wh(media)
    if crop is None:
        y0, bh = 0, h0
    else:
        y0 = max(0, int(h0 * crop[0]))
        bh = max(8, int(h0 * crop[1]) - y0)
    bh -= bh % 2                      # 偶数高，rawvideo 才不会错半行
    bh = max(8, bh)
    if width:
        W = max(16, min(int(width), w0))
        H = max(8, int(round(bh * W / w0)))
        H -= H % 2
    else:
        W, H = w0, bh                 # 不缩放：裁完直接输出原像素
    step = float(step) if step and step > 0 else 1.0
    parts = ["fps=%s" % (1.0 / step)]
    if crop is not None:
        parts.append("crop=%d:%d:0:%d" % (w0, bh, y0))
    if width:
        parts.append("scale=%d:%d" % (W, H))
    # 解码线程固定 2：默认是 auto（按核数开满），手机上 8 路抢 8 个小核，
    # 实测反而比 2 路慢（120 帧 6.01s → 5.67s），也更容易把 OCR 挤慢。
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
           "-threads", str(FFMPEG_THREADS),
           "-ss", str(start), "-i", str(media)]
    if end is not None:
        cmd += ["-t", str(max(0.0, float(end) - float(start)))]
    cmd += ["-vf", ",".join(parts), "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    frame_size = W * H * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        i = 0
        while True:
            buf = proc.stdout.read(frame_size)
            if not buf or len(buf) < frame_size:
                break
            yield i, start + i * step, np.frombuffer(buf, dtype=np.uint8).reshape(H, W, 3)
            i += 1
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()


STATIC_RATIO = 0.5      # 同一串字在簇内占比达到这个数 → 判为常驻水印


def _norm_text(t):
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]", "", (t or "").lower())


def _is_static(c):
    """同一簇里出现最多的那串字占比 ≥50%，且至少出现 2 次 → 判为静态标。"""
    txts = c.get("texts") or []
    if c.get("count", 0) < 2 or not txts:
        return False
    top = Counter(_norm_text(t) for t in txts).most_common(1)[0][1]
    return top >= STATIC_RATIO * c["count"]


def _collect_sample_frames(media, start, end, samples, W, H):
    """取 samples 帧整帧数组（先不裁），管道优先，不够退回逐点 seek。"""
    span = max(0.0, end - start)
    frames = []
    if span <= 45.0:
        # 顺序解码约 13 倍速：45 秒窗口 3.5 秒解完，10 分钟要 46 秒。
        # 所以只有短窗口走解码，更长的逐点 seek（每次约 0.35 秒，与片长无关）。
        step = span / float(samples + 1)
        for i, _t, arr in _stream_frames(media, start, end, step=step, width=W, crop=None):
            if i == 0:
                continue                       # 首帧常是黑场，跳过
            frames.append(arr)
            if len(frames) >= samples:
                break
    if len(frames) < max(2, samples // 2):
        frames = []
        for i in range(samples):
            t = start + span * (i + 1) / (samples + 1)
            cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                   "-threads", str(FFMPEG_THREADS),
                   "-ss", "%.3f" % t, "-i", str(media), "-frames:v", "1",
                   "-vf", "scale=%d:%d" % (W, H), "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
            proc = subprocess.run(cmd, capture_output=True, timeout=15)
            if proc.returncode != 0 or len(proc.stdout) < W * H * 3:
                continue
            frames.append(np.frombuffer(proc.stdout[:W * H * 3], dtype=np.uint8).reshape(H, W, 3))
    return frames


def _boxes_from_frames(frames, y_from, y_to, H, W, engine):
    """OCR 每帧的 [y_from:y_to) 这一段，返回 [(y0, y1, 宽度占比, 文字), ...]。"""
    boxes = []
    for fr in frames:
        try:
            res = engine.ocr_image(fr[y_from:y_to])
        except Exception:
            continue
        for r in res:
            b = r["box"]
            y0 = (b[1] + y_from) / float(H)
            y1 = (b[3] + y_from) / float(H)
            if (y1 - y0) < 0.02:  # 太矮的噪点排除
                continue
            boxes.append((y0, y1, max(0.0, (b[2] - b[0]) / float(W)),
                          (r.get("text") or "").strip()))
    return boxes


def _cluster_boxes(boxes):
    """按纵向重叠把框聚成簇。

    必须限高：重叠是传递的，一屏小字能从顶串到底，串成一个 0.0-0.8 的「簇」，
    选带就彻底跑偏（实测《areyoulost》就这样栽过）。
    """
    clusters = []
    for y0, y1, wr, txt in boxes:
        hit = None
        for c in clusters:
            if y0 < c["y1"] and y1 > c["y0"]:
                if max(c["y1"], y1) - min(c["y0"], y0) > 0.25:
                    continue                 # 这一下会把簇拉得太高，另起一簇
                hit = c
                break
        if hit is None:
            clusters.append({"y0": y0, "y1": y1, "count": 1, "hs": [y1 - y0], "ws": [wr],
                             "texts": [txt] if txt else []})
        else:
            hit["y0"] = min(hit["y0"], y0)
            hit["y1"] = max(hit["y1"], y1)
            hit["count"] += 1
            hit["hs"].append(y1 - y0)
            hit["ws"].append(wr)
            if txt:
                hit["texts"].append(txt)
    return clusters


def _usable(c):
    """字幕是「一行、够宽」。这两个条件先筛掉台标、图标、按钮字母。"""
    w = max(c["ws"]) if c["ws"] else 0.0
    if w < 0.12:
        return False
    if (c["y1"] - c["y0"]) > 0.25:
        return False
    return True


def _page_like(boxes, sub=None, samples=8):
    """判断这一屏是不是「整页文字」，而不是字幕。

    字幕只有一行（偶尔两行），换几十句也只在带内挤着；
    整页文字（幻灯片、截图、作文纸）会在多个高度各自成行。

    实测四支片子（可用簇数 / 纵向跨度）：
      幻灯片 dsh-storyboard   6 / 0.65
      作文纸《南充0分》      15 / 0.84
      歌词片《琵琶行》        1 / 0.00
      实拍片 泡沫满屏         3 / 0.21

    但「簇够多 + 跨度够大」会被垃圾簇凑满：台标和图表标签只有 0.13-0.20 画面宽，
    它们随镜头换高度，柴知道 653s 那支片 6 个 100s 窗口全凑出 4-5 个簇、跨度
    0.7-0.87，于是全被判成整页——可底部那条字幕带好得很。

    所以 sub（选中的那条带的统计）给了就加一条豁免：选中的带自己够低（y0≥0.6）、够宽（≥0.30）、
    大半采样帧都在（count ≥ 采样数一半），那它就是字幕，屏幕上别处的字是内容。
    这支片的 6 个窗口实测（band 起点 / 段子质量）：
      0-100   0.827 → 61 段，连贯      → 豁免，判字幕
      100-200 0.793 → 57 段，连贯      → 豁免
      200-300 0.559 → 11 段，全是图注   → 不豁免，仍判整页（对）
      300-400 0.830 → 56 段，连贯      → 豁免
      400-500 0.787 → 60 段，连贯      → 豁免
      500-600 0.825 → 54 段，连贯      → 豁免

    返回 (是否整页, 可用簇数, 跨度)。
    """
    cl = [c for c in _cluster_boxes(boxes) if _usable(c)]
    if len(cl) < 4:
        return False, len(cl), 0.0
    span = max(c["y1"] for c in cl) - min(c["y0"] for c in cl)
    if span < 0.45:
        return False, len(cl), span
    if sub is not None:
        # 用带自己的统计（_band_from_boxes 已经算好），别再去簇里找对应项：
        # 合并相邻行之后两者不一样，500-600 那窗就是这么漏的。
        if (sub["band"][0] >= 0.6 and sub["w"] >= 0.30
                and sub["count"] >= max(3, samples // 2)):
            return False, len(cl), span
    return True, len(cl), span


def _band_from_boxes(boxes, pad):
    """把框按纵向重叠聚类、打分、合并相邻行，返回 {band, count, w, line_h} 或 None。"""
    if not boxes:
        return None
    clusters = _cluster_boxes(boxes)

    def _pos(y0):
        # 字幕一般压底或贴顶，画面中部那是主体内容。底部权重最高，顶部次之，中部打折。
        if y0 >= 0.5:
            return 1.0 + (y0 - 0.5) * 3.0        # 0.5 → 1.0，0.9 → 2.2
        if y0 <= 0.3:
            return 1.0 + (0.3 - y0) * 2.0        # 0.1 → 1.4，0.0 → 1.6
        return 0.6

    def score(c):
        hs = sorted(c["hs"])
        h = hs[len(hs) // 2] if hs else 0.0        # 中位数，别让一个高框把整簇抬起来
        w = max(c["ws"]) if c["ws"] else 0.0
        seen = min(1.0, 0.5 + 0.1 * c["count"])    # 反复出现的才像字幕
        s = (h ** 1.5) * (0.3 + w) * _pos(c["y0"]) * seen
        # 常驻水印/台标：同一位置反复出现同一段字。字幕每句都变，水印不变。
        if _is_static(c):
            s *= 0.25
        return s

    if os.environ.get("BAND_DEBUG"):
        print("  == 过滤前 %d 个簇 ==" % len(clusters))
        for c in sorted(clusters, key=lambda c: c["y0"]):
            print("     y=%.3f-%.3f n=%d w=%.2f 可用=%s 文字=%s" % (
                c["y0"], c["y1"], c["count"],
                max(c["ws"]) if c["ws"] else 0, _usable(c), list(c["texts"])[:2]))
    clusters = [c for c in clusters if _usable(c)] or clusters
    if os.environ.get("BAND_DEBUG"):
        for c in sorted(clusters, key=score, reverse=True):
            print("  簇 y=%.3f-%.3f n=%d h=%.3f w=%.2f 分=%.5f 文字=%s" % (
                c["y0"], c["y1"], c["count"], sum(c["hs"]) / len(c["hs"]),
                max(c["ws"]) if c["ws"] else 0, score(c), list(c["texts"])[:3]))
    best = max(clusters, key=score)
    # 字幕有时一句、有时两行，纵向差 3~5%，只留最高分那一簇会把矮的那行切掉。
    # 紧挨着（间隔 ≤2%）就并成一条带；并出来不能高过画面的 1/5，免得连水印一起吞。
    y0, y1 = best["y0"], best["y1"]
    for c in sorted(clusters, key=lambda c: c["y0"]):
        if c is best:
            continue
        if c["y0"] - y1 <= 0.02 or y0 - c["y1"] <= 0.02:
            ny0, ny1 = min(y0, c["y0"]), max(y1, c["y1"])
            if ny1 - ny0 <= 0.2:
                y0, y1 = ny0, ny1
    # 双语/两行字幕会整块上下移动（一行时居中、两行时顶上去），固定 0.02 的边距盖不住。
    # 按实测行高给边距：8 个采样点看不到所有排版时，这一圈是唯一的余量。
    hs = sorted(best["hs"])
    line_h = hs[len(hs) // 2] if hs else 0.0
    pad_eff = max(pad, 0.6 * line_h)
    y0_band = max(0.0, y0 - pad_eff)
    y1_band = min(1.0, y1 + pad_eff)
    if (y1_band - y0_band) < 0.03:
        return None
    return {"band": (round(y0_band, 4), round(y1_band, 4)),
            "count": best["count"], "w": max(best["ws"]) if best["ws"] else 0.0,
            "line_h": line_h}


def detect_subtitle_band(media, samples=8, start=0.0, end=None, width=None,
                         lower_from=None, pad=0.02, engine=None, return_detail=False):
    """利用少量采样点按 OCR 框坐标聚类，秒级自动锁定字幕带 (y0_ratio, y1_ratio)。

    - 默认两段式：先只搜下方 40%（多数片子字幕在那儿，也便宜）；
      找不到像样的簇再整幅扫一遍。字幕在上方的片子多付一次 OCR，常见情况不多付。
    - samples 默认 8 点；短窗口一次解码取帧，长窗口逐点 seek
    - 两段都走同一批帧数组，解码只做一次
    - lower_from 给了（0.0~1.0）就只搜那一段，不跑两段式
    - return_detail=True 时返回 dict（多带 page_like 等判断依据），默认仍返回 tuple
    """
    if engine is None:
        engine = get_engine()
    dur = _probe_duration(media)
    if end is None:
        end = dur
    end = min(float(end), dur) if dur else float(end)
    span = max(0.0, end - start)
    if span <= 1.0:
        return None

    w0, h0 = _probe_wh(media)
    # 采样宽度跟片源走：固定 480/640 时，720p 的字幕缩完只有 20 来像素高，检测不到。
    if width is None:
        width = min(960, max(640, int(w0 * 0.75)))
    W = min(width, w0)
    H = int(round(h0 * W / w0))
    H -= H % 2

    frames = _collect_sample_frames(media, start, end, samples, W, H)
    if not frames:
        # 退路：加密采样点，再不行降一档分辨率
        if samples < 16:
            return detect_subtitle_band(media, samples=16, start=start, end=end, width=width,
                                        lower_from=lower_from, pad=pad, engine=engine)
        if width > 560:
            return detect_subtitle_band(media, samples=16, start=start, end=end,
                                        width=max(480, width // 2), lower_from=lower_from,
                                        pad=pad, engine=engine)
        return None

    def _ret(band, page, ncl, span, b):
        if not return_detail:
            return band
        d = {"band": band, "page_like": page, "clusters": ncl, "span": round(span, 3),
             "count": b["count"] if b else 0,
             "w": b["w"] if b else 0.0, "line_h": b["line_h"] if b else 0.0}
        return d

    if lower_from is not None:
        r = _band_from_boxes(_boxes_from_frames(frames, int(H * lower_from), H, H, W, engine), pad)
        if r is None:
            return None
        page, ncl, span = _page_like(_boxes_from_frames(frames, 0, H, H, W, engine),
                                     r, samples)
        return _ret(r["band"], page, ncl, span, r)

    cut = int(H * 0.6)
    low_boxes = _boxes_from_frames(frames, cut, H, H, W, engine)
    r = _band_from_boxes(low_boxes, pad)
    if r and r["count"] >= 2 and r["w"] >= 0.2:
        # 下半屏命中了，但还不能收工。
        # 整页文字的视频（幻灯片、截图、作文纸）下半屏也「像字幕」——
        # 它的表格行刚好满足 count/w 的门槛。只有看上半屏才知道
        # 这条带是「字幕」，还是「一整页内容的下半截」。
        all_boxes = low_boxes + _boxes_from_frames(frames, 0, cut, H, W, engine)
        page, ncl, span = _page_like(all_boxes, r, samples)
        return _ret(r["band"], page, ncl, span, r)

    # 下半屏没命中：补扫上半屏，合并再打分
    all_boxes = low_boxes + _boxes_from_frames(frames, 0, cut, H, W, engine)
    r2 = _band_from_boxes(all_boxes, pad)
    if r2:
        page, ncl, span = _page_like(all_boxes, r2, samples)
        return _ret(r2["band"], page, ncl, span, r2)
    if samples < 16:
        return detect_subtitle_band(media, samples=16, start=start, end=end, width=width,
                                    lower_from=lower_from, pad=pad, engine=engine,
                                    return_detail=return_detail)
    return None

def _text_sig(arr, th):
    """字幕的视觉签名：高亮像素掩码。背景在动也不影响文字轮廓。

    用 uint16 求和代替 float32 平均：判据等价（和 > 3*th ⇔ 均值 > th），
    但不用把整幅 1280x144x3 转成 float，900 帧能省好几秒。
    """
    return arr.astype(np.uint16).sum(axis=2) > int(3 * float(th))


def fold_segments(results, min_len=2, max_edit_ratio=0.2, max_gap=5.0, step=1.0):
    """逐帧 OCR 结果折叠成字幕段。

    重要认知（实测《琵琶行》教学/音乐视频得出）：
      这类视频的歌词会把每句的尾短语**重复演唱一遍**，字幕轨里因此存在
      独立短 cue（如「如私语」「落玉盘」「胜有声」）。它们是**真实内容**，
      既不是 OCR 残缺，也不是擦除动画，绝不能当成噪声合并掉。
      判据：字幕发生「硬切 + 重新居中」，即为新 cue。

    折叠规则（只处理真正的噪声）：
      1. 空帧/过短 → 丢弃
      2. 与上段同文 → 同一 cue 延续
      3. 与上段近似（编辑距离 ≤ max(1, 20%·较短长度)）→ 同一 cue 的 OCR 变体，合并
      3b. 一方包含另一方（≥6 字，且间隔 ≤ 1.6 帧）→ 同一 cue 被 OCR 截断，合并
      4. 否则 → 视为新 cue，单独成段

    段长：单帧出现的 cue 至少持续 `step` 秒，否则时间轴全是零长度。
    """
    out = []
    for r in results:
        txt = " ".join(x for x in r.get("lines", []) if x).strip()
        if len(txt) < min_len:
            continue
        conf = float(r.get("conf") or 0.0)
        sec = float(r.get("seconds") or 0.0)
        if out:
            prev = out[-1]
            pt = prev["text"]
            if sec - prev["_sec"] <= max_gap:
                same = False
                if txt == pt:
                    same = True
                else:
                    m = min(len(txt), len(pt))
                    if m >= 4 and _lev(txt, pt) <= max(1, int(m * max_edit_ratio)):
                        same = True
                    elif m >= 6 and sec - prev["_sec"] <= step * 1.6 and (txt in pt or pt in txt):
                        same = True
                if same:
                    if len(txt) > len(pt) or (len(txt) == len(pt) and conf >= prev.get("conf", 0.0) - 0.02):
                        prev["text"] = txt
                        prev["conf"] = round(conf, 4)
                    prev["_last"] = sec
                    prev["_sec"] = sec
                    continue
        out.append({"from": _tc(sec), "to": _tc(sec + step), "text": txt,
                    "conf": round(conf, 4), "_sec": sec, "_last": sec})
    for sg in out:
        last = sg.pop("_last", sg.pop("_sec", 0.0))
        sg.pop("_sec", None)
        sg["to"] = _tc(last + step)
    return out


def _keep_wide(res, W):
    """带里往往还混着正文/手写小字：字幕是横向铺开的，只留最宽那一档。

    判据：宽度 ≥ 本帧最大框宽的 0.38，且至少占画面 12%；行高还要达到
    最高行的 55%（正文小字比字幕矮一截）。都不够宽时原样返回，避免把
    短字幕整帧丢掉。

    宽度只看最宽那一档会误杀短句：同帧里「呦呦呦耶耶耶」只有最宽行的
    0.44 倍，按 top*0.5 判就被丢掉，而它是真字幕。行高这一条是用来顶住
    放宽宽度后混进来的正文的。
    """
    if len(res) <= 1:
        return res
    ws = [(r["box"][2] - r["box"][0]) / float(W) for r in res]
    hs = [float(r["box"][3] - r["box"][1]) for r in res]
    # 宽度标尺怕离群值：横跨画面的水印条/整行横线会把 max 抬到 0.9 画面宽，
    # 正常字幕行全被判窄。先剔掉比中位数宽 2.5 倍的框再取 max。
    #
    # 行高基准仍用 max：试过中位数，多留下来的多半是游戏 UI（成就提示 /
    # 死亡消息 / 玩家聊天），真字幕没救回来——那几帧是检测框把字幕截断了，
    # 不是筛选层的问题。见 revert 提交的说明。
    med_w = float(np.median(ws))
    top = max([w for w in ws if w <= med_w * 2.5] or ws)
    top_h = max(hs)
    keep = [r for r, w, h in zip(res, ws, hs)
            if w >= max(0.12, top * 0.38) and h >= 0.55 * top_h]
    return keep or res


def ocr_subtitles(media, start=0.0, end=None, step=1.0, band="auto",
                  width=None, skip_th=1.0, text_th=223, min_conf=0.0, engine=None,
                  workers=None):
    """字幕带 OCR：ffmpeg 管道流出带帧（不落盘），字幕没变的帧跳过不识别。

    band: (y0_ratio, y1_ratio) 显式指定；"auto" 用 OCR 框坐标自动定位；None 等价 auto。
    skip_th: 与上一帧的带内平均绝对差小于此值即判「没变」，复用上次结果。
    """
    if engine is None:
        engine = get_engine()
    band_used = band
    band_auto = None
    det = None
    if band in (None, "auto"):
        det = detect_subtitle_band(media, samples=8, start=start, end=end, engine=engine,
                                   return_detail=True)
        if det is None:
            # 下半截、两种分辨率、最多 16 个采样点都没找到字：判为无字幕。
            # 不再拿默认带白跑一遍逐帧识别。
            return {"frames": 0, "ocr_runs": 0, "skipped": 0, "band": None,
                    "band_auto": None, "band_px": [0, 0], "step": float(step),
                    "results": [], "note": "整幅未检出文字，判为无字幕"}
        band_auto = det["band"]
        band_used = band_auto
    y0r, y1r = float(band_used[0]), float(band_used[1])
    # 长窗口切段并行：每段自己解码 / 检测 / 识别，最后按时间合并。
    # 单机 8 核，串行时 ffmpeg 与 ORT 各跑 2 线程只占一半核，纯浪费。
    span = max(0.0, float(end if end is not None else start + step) - float(start))
    if workers is None:
        workers = int(os.environ.get("MVV_OCR_WORKERS", "2") or 0)
    if workers > 1 and span >= PARALLEL_MIN_SPAN:
        n = max(2, min(int(workers), 8))
        cuts = [start + span * i / n for i in range(n + 1)]
        tasks = [(media, cuts[i], cuts[i + 1], step, (y0r, y1r), width,
                  skip_th, text_th, min_conf) for i in range(n)]
        import concurrent.futures as _cf
        # 进程池起不来（例如脚本从 stdin / python -c 跑，forkserver 找不到 __main__）
        # 或某一段的子进程崩了，别把整次识别丢掉：
        # 起不来就整段串行，个别段失败就只补那一段。
        parts = []
        try:
            with _cf.ProcessPoolExecutor(max_workers=n) as ex:
                futs = [ex.submit(_ocr_window, t) for t in tasks]
                for f in futs:
                    try:
                        parts.append(f.result())
                    except Exception:
                        parts.append(None)
        except Exception:
            parts = [None] * len(tasks)
        serial_fallback = any(p is None for p in parts)
        for i, part in enumerate(parts):
            if part is None:
                parts[i] = _ocr_window(tasks[i], engine=engine)
        res = []
        n_ocr = n_skip = 0
        W = H = 0
        for p in parts:
            res.extend(p["results"])
            n_ocr += p["ocr_runs"]
            n_skip += p["skipped"]
            if not W:
                W, H = p["band_px"][0], p["band_px"][1]
        res.sort(key=lambda r: r["seconds"])
        ret = {"frames": len(res), "ocr_runs": n_ocr, "skipped": n_skip,
               "band": [y0r, y1r], "band_auto": band_auto, "band_px": [W, H],
               "step": float(step), "results": res, "workers": n,
               "serial_fallback": serial_fallback}
        if det is not None:
            ret["page_like"] = bool(det.get("page_like"))
            ret["text_clusters"] = det.get("clusters")
            ret["text_span"] = det.get("span")
        return ret

    one = _ocr_window((media, start, end, step, (y0r, y1r), width, skip_th, text_th, min_conf),
                      engine=engine)
    ret = {"frames": one["frames"], "ocr_runs": one["ocr_runs"], "skipped": one["skipped"],
           "band": [y0r, y1r], "band_auto": band_auto, "band_px": one["band_px"],
           "step": float(step), "results": one["results"]}
    if det is not None:
        ret["page_like"] = bool(det.get("page_like"))
        ret["text_clusters"] = det.get("clusters")
        ret["text_span"] = det.get("span")
    return ret


def _ocr_window(task, engine=None):
    """一段窗口的逐帧 OCR。串行与并行走同一份代码，结果才不会有差别。"""
    media, start, end, step, band_used, width, skip_th, text_th, min_conf = task
    if engine is None:
        # 并行时每个进程只留一条 ORT 线程：2 进程 x 1 线程 = 2，
        # 跟单进程 2 线程同样的算力，但不会和别的进程抢核。
        os.environ.setdefault("MVV_ORT_THREADS", "1")
        engine = get_engine()
    y0r, y1r = float(band_used[0]), float(band_used[1])
    out, prev, prev_res = [], None, None
    prev_sig = None
    n_ocr = n_skip = 0
    W = H = 0
    for i, t, arr in _stream_frames(media, start, end, step=step, width=width,
                                    crop=(y0r, y1r)):
        H, W = arr.shape[0], arr.shape[1]
        if prev is not None and skip_th >= 0:
            ps = prev_sig
            cs = _text_sig(arr, text_th)
            if not ps.any() and not cs.any():
                d = 0.0
            else:
                d = 100.0 * float(np.mean(cs != ps))
            if d < skip_th:
                n_skip += 1
                out.append({"seconds": round(t, 3), "timestamp": _tc(t), "skipped": True,
                            "diff": round(d, 2),
                            "conf": round(sum(r["conf"] for r in prev_res) / len(prev_res), 4) if prev_res else 0.0,
                            "lines": [r["text"] for r in prev_res]})
                continue
        # 带边太窄时字会贴着上下边，识别器会把贴边的字吞掉（"…夜送客"→"…夜送"）。
        # 纵向补一圈边缘像素再识别，不改判定用的原图。
        pad = max(4, int(round(H * 0.10)))
        img = np.pad(arr, ((pad, pad), (0, 0), (0, 0)), mode="edge") if pad else arr
        res = _keep_wide(engine.ocr_image(img, min_conf=min_conf), W)
        n_ocr += 1
        prev, prev_res, prev_sig = arr, res, _text_sig(arr, text_th)
        conf = round(sum(r["conf"] for r in res) / len(res), 4) if res else 0.0
        out.append({"seconds": round(t, 3), "timestamp": _tc(t), "skipped": False,
                    "conf": conf, "lines": [r["text"] for r in res]})
    return {"frames": len(out), "ocr_runs": n_ocr, "skipped": n_skip,
            "band": [y0r, y1r], "band_px": [W, H], "step": float(step), "results": out}


def _lev(a, b):
    """编辑距离（字符串短，直接 DP）。"""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ca = a[i - 1]
        for j in range(1, lb + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != b[j - 1]))
        prev = cur
    return prev[lb]
