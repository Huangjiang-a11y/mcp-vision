"""PP-OCRv4 的 Python 实现：DB 检测 + CTC 识别。
依赖：onnxruntime + numpy + ffmpeg（读图）。不依赖 OpenCV / Pillow / pyclipper。
模型：ch_PP-OCRv4_det_infer.onnx / ch_PP-OCRv4_rec_infer.onnx / ppocr_keys_v1.txt
"""
import os, sys, json, subprocess, math
from pathlib import Path
import numpy as np
import onnxruntime as ort

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("MVV_MODEL_DIR") or (APP_DIR / "models"))
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DET_MAX_SIDE = int(os.environ.get("DET_MAX_SIDE", "720"))
DET_TH = float(os.environ.get("DET_TH", "0.2"))


def _providers():
    want = os.environ.get("MVV_ORT_PROVIDERS", "xnnpack,cpu").lower().split(",")
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
        det = md / "ch_PP-OCRv4_det_infer.onnx"
        rec = md / "ch_PP-OCRv4_rec_infer.onnx"
        dic = md / "ppocr_keys_v1.txt"
        for f in (det, rec, dic):
            if not f.exists():
                raise FileNotFoundError("缺模型文件: %s（跑 scripts/fetch-models.py）" % f)
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(os.environ.get("MVV_ORT_THREADS", "4"))
        so.log_severity_level = 3
        prov = providers or _providers()
        self.det = ort.InferenceSession(str(det), so, providers=prov)
        self.rec = ort.InferenceSession(str(rec), so, providers=prov)
        self.dict = []
        for line in dic.read_text(encoding="utf-8").splitlines():
            if line.endswith("\r"):
                line = line[:-1]
            self.dict.append(line)
        while self.dict and self.dict[-1] == "":
            self.dict.pop()
        self.providers = prov

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
        """4 邻域连通域 → bbox。纯 numpy + BFS。"""
        visited = np.zeros((H, W), dtype=bool)
        idx = np.argwhere(mask)
        results = []
        for sy, sx in idx:
            if visited[sy, sx]:
                continue
            stack = [(sy, sx)]
            visited[sy, sx] = True
            minx = maxx = sx; miny = maxy = sy; count = 0
            while stack:
                y, x = stack.pop()
                count += 1
                if x < minx: minx = x
                elif x > maxx: maxx = x
                if y < miny: miny = y
                elif y > maxy: maxy = y
                if x + 1 < W and mask[y, x+1] and not visited[y, x+1]:
                    visited[y, x+1] = True; stack.append((y, x+1))
                if x - 1 >= 0 and mask[y, x-1] and not visited[y, x-1]:
                    visited[y, x-1] = True; stack.append((y, x-1))
                if y + 1 < H and mask[y+1, x] and not visited[y+1, x]:
                    visited[y+1, x] = True; stack.append((y+1, x))
                if y - 1 >= 0 and mask[y-1, x] and not visited[y-1, x]:
                    visited[y-1, x] = True; stack.append((y-1, x))
            results.append((minx, miny, maxx, maxy, count))
            if len(results) > 400:
                break
        return results

    # ---- rec ----
    def recognize(self, rgb, box):
        x0, y0, x1, y1 = box
        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(rgb.shape[1] - 1, x1); y1 = min(rgb.shape[0] - 1, y1)
        crop = rgb[y0:y1+1, x0:x1+1]
        if crop.size == 0:
            return "", 0.0
        bw, bh = crop.shape[1], crop.shape[0]
        target_h = 48
        target_w = int(max(16, min(320, round((bw / max(bh, 1)) * target_h))))
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


def ocr_file(path, **kw):
    rgb = load_rgb(path, long_side=kw.pop("long_side", None))
    return PPOCR(**kw).ocr_image(rgb)


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
