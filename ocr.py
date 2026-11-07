"""PP-OCRv4 的 Python 实现：DB 检测 + CTC 识别。
依赖：onnxruntime + numpy + ffmpeg（读图）。不依赖 OpenCV / Pillow / pyclipper。
模型：路径由 MVV_OCR_DET / MVV_OCR_REC / MVV_OCR_KEYS 指定（默认 models/ 下的 PP-OCRv4）
"""
import os, sys, json, subprocess, math
from pathlib import Path
import numpy as np

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("MVV_MODEL_DIR") or (APP_DIR / "models"))
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DET_MAX_SIDE = int(os.environ.get("DET_MAX_SIDE", "720"))
# 识别输入的最大宽度：字幕行短，320 够；手写长行降到 256/192 明显掉字，别乱调
REC_MAX_W = int(os.environ.get("MVV_REC_MAX_W", "320"))
DET_TH = float(os.environ.get("DET_TH", "0.2"))


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


def get_ppocr(model_dir=None, providers=None):
    """懒加载 + 复用：第一次真正用到 OCR 时建会话，之后一直复用。"""
    if model_dir is None and providers is None:
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
    s = max(0.0, float(seconds))
    return "%02d:%06.3f" % (int(s // 60), s % 60)


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
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
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


def detect_subtitle_band(media, samples=4, start=0.0, end=None, width=None,
                         lower_from=0.45, pad=0.02, engine=None):
    """利用少量采样点按 OCR 框坐标聚类，秒级自动锁定字幕带 (y0_ratio, y1_ratio)。

    - samples 默认 4 点，用 ffmpeg -ss 单帧瞬时抓取
    - 仅对画面下半截（> lower_from）做 OCR，避开大半画面
    - 耗时通常在 1~2 秒左右
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

    # 均匀挑选 samples 个探测时间点，稍微避开纯头尾
    pts = [start + span * (i + 1) / (samples + 1) for i in range(samples)]
    w0, h0 = _probe_wh(media)
    # 采样宽度跟片源走：固定 480/640 时，720p 的字幕缩完只有 20 来像素高，检测不到。
    # 取源宽的 75%，夹在 640~960 之间。
    if width is None:
        width = min(960, max(640, int(w0 * 0.75)))
    W = min(width, w0)
    H = int(round(h0 * W / w0))
    H -= H % 2
    crop_y = int(H * lower_from)

    boxes = []
    for t in pts:
        cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
               "-ss", "%.3f" % t, "-i", str(media), "-frames:v", "1",
               "-vf", "scale=%d:%d" % (W, H), "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
        p = subprocess.run(cmd, capture_output=True, timeout=15)
        if p.returncode != 0 or len(p.stdout) < W * H * 3:
            continue
        arr = np.frombuffer(p.stdout[:W*H*3], dtype=np.uint8).reshape(H, W, 3)
        # 只检下半截
        lower = arr[crop_y:]
        try:
            res = engine.ocr_image(lower)
        except Exception:
            continue
        for r in res:
            b = r["box"]
            y0 = (b[1] + crop_y) / float(H)
            y1 = (b[3] + crop_y) / float(H)
            if (y1 - y0) < 0.02:  # 太矮的噪点排除
                continue
            boxes.append((y0, y1, max(0.0, (b[2] - b[0]) / float(W))))

    if not boxes:
        # 两条退路：先加密采样点（短窗口里字幕可能恰好不在那 4 个时刻），
        # 再降一档分辨率（小字幕在高分辨率下框不出字）。
        if samples < 8:
            return detect_subtitle_band(media, samples=8, start=start, end=end, width=width,
                                        lower_from=lower_from, pad=pad, engine=engine)
        if width > 560:
            return detect_subtitle_band(media, samples=8, start=start, end=end,
                                        width=max(480, width // 2), lower_from=lower_from,
                                        pad=pad, engine=engine)
        return None

    # 按纵向重叠聚类
    clusters = []
    for y0, y1, wr in boxes:
        hit = None
        for c in clusters:
            if y0 < c["y1"] and y1 > c["y0"]:
                hit = c
                break
        if hit is None:
            clusters.append({"y0": y0, "y1": y1, "count": 1, "hs": [y1 - y0], "ws": [wr]})
        else:
            hit["y0"] = min(hit["y0"], y0)
            hit["y1"] = max(hit["y1"], y1)
            hit["count"] += 1
            hit["hs"].append(y1 - y0)
            hit["ws"].append(wr)

    # 选簇不能只看框数：手写正文一行行的小字框永远比字幕多。
    # 字幕的特征是「字大、横向长、位置靠下」，所以按字号×宽度×位置打分。
    def score(c):
        h = sum(c["hs"]) / len(c["hs"])
        w = max(c["ws"]) if c["ws"] else 0.0
        lower = max(0.0, c["y0"] - 0.5)     # 0.5 以下才算加分
        return (h ** 1.5) * (0.3 + w) * (1.0 + lower * 2.0)

    best = max(clusters, key=score)
    # 至少在 1 个采样点明确检出，且高宽比合理
    y0_band = max(0.0, best["y0"] - pad)
    y1_band = min(1.0, best["y1"] + pad)
    if (y1_band - y0_band) < 0.03:
        return None
    return (round(y0_band, 4), round(y1_band, 4))


def _text_sig(arr, th):
    """字幕的视觉签名：高亮像素掩码。背景在动也不影响文字轮廓。"""
    g = arr.astype(np.float32).mean(axis=2)
    return g > float(th)


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

    判据：宽度 ≥ 本帧最大框宽的一半，且至少占画面 12%。
    都不够宽时原样返回，避免把短字幕整帧丢掉。
    """
    if len(res) <= 1:
        return res
    ws = [(r["box"][2] - r["box"][0]) / float(W) for r in res]
    top = max(ws)
    keep = [r for r, w in zip(res, ws) if w >= max(0.12, top * 0.5)]
    return keep or res


def ocr_subtitles(media, start=0.0, end=None, step=1.0, band="auto",
                  width=None, skip_th=1.0, text_th=220, min_conf=0.0, engine=None):
    """字幕带 OCR：ffmpeg 管道流出带帧（不落盘），字幕没变的帧跳过不识别。

    band: (y0_ratio, y1_ratio) 显式指定；"auto" 用 OCR 框坐标自动定位；None 等价 auto。
    skip_th: 与上一帧的带内平均绝对差小于此值即判「没变」，复用上次结果。
    """
    if engine is None:
        engine = get_engine()
    band_used = band
    band_auto = None
    if band in (None, "auto"):
        band_auto = detect_subtitle_band(media, samples=4, start=start, end=end, engine=engine)
        band_used = band_auto or (0.70, 1.0)
    y0r, y1r = float(band_used[0]), float(band_used[1])
    out, prev, prev_res = [], None, None
    n_ocr = n_skip = 0
    W = H = 0
    for i, t, arr in _stream_frames(media, start, end, step=step, width=width,
                                    crop=(y0r, y1r)):
        H, W = arr.shape[0], arr.shape[1]
        if prev is not None and skip_th >= 0:
            ps = _text_sig(prev, text_th)
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
        res = _keep_wide(engine.ocr_image(arr, min_conf=min_conf), W)
        n_ocr += 1
        prev, prev_res = arr, res
        conf = round(sum(r["conf"] for r in res) / len(res), 4) if res else 0.0
        out.append({"seconds": round(t, 3), "timestamp": _tc(t), "skipped": False,
                    "conf": conf, "lines": [r["text"] for r in res]})
    return {"frames": len(out), "ocr_runs": n_ocr, "skipped": n_skip,
            "band": [y0r, y1r], "band_auto": band_auto, "band_px": [W, H],
            "step": float(step), "results": out}


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
