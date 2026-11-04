"""对比：全帧逐帧识别 vs 字幕带裁切+跳帧。"""
import time, os, subprocess, tempfile, numpy as np, sys
sys.path.insert(0, ".")
import ocr as O

MEDIA = sys.argv[1] if len(sys.argv) > 1 else "out/subs.mp4"
STEP = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
TOTAL = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0


def all_frames_raw(media, step, width):
    w0, h0 = O._probe_wh(media)
    W = min(width, w0); H = int(round(h0 * W / w0)); H -= H % 2
    fd, raw = tempfile.mkstemp(suffix=".raw"); os.close(fd)
    subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", media, "-vf", "fps=%s,scale=%d:%d" % (1/step, W, H),
                    "-pix_fmt", "rgb24", "-f", "rawvideo", raw], capture_output=True, timeout=1800)
    return raw, W, H


print("视频 %s | 采样 1/%.1fs | 约 %d 帧" % (MEDIA, STEP, int(TOTAL/STEP)))
t0 = time.time(); O.PPOCR(); t_fresh = time.time() - t0
t0 = time.time(); eng = O.get_engine(); t_cached = time.time() - t0
print("engine 初始化: 新建 %.3fs | 单例复用 %.3fs" % (t_fresh, t_cached))

raw, W, H = all_frames_raw(MEDIA, STEP, 640)
fs = W*H*3
n = os.path.getsize(raw)//fs
print("全帧流: %dx%d, %d 帧\n" % (W, H, n))

t0 = time.time(); k = 0
with open(raw, "rb") as f:
    while True:
        b = f.read(fs)
        if len(b) < fs: break
        eng.ocr_image(np.frombuffer(b, dtype=np.uint8).reshape(H, W, 3)); k += 1
t_full = time.time() - t0
os.unlink(raw)
print("A 全帧逐帧识别:  %2d 帧 × %.3fs = %.2fs  (%d 次 OCR)" % (k, t_full/max(k,1), t_full, k))

t0 = time.time()
r = O.ocr_subtitles(MEDIA, 0, TOTAL, step=STEP, band=(0.72, 1.0), width=640, skip_th=3.0, engine=eng)
t_band = time.time() - t0
print("C 字幕带+跳帧:   %2d 帧，其中 %d 帧跳过 → 只跑 %d 次 OCR = %.2fs" % (
    r["frames"], r["skipped"], r["ocr_runs"], t_band))
print("   带内 %dx%d（原帧 %dx%d）" % (r["band_px"][0], r["band_px"][1], W, H))
print("\n提速: %.2fs → %.2fs = %.1fx\n" % (t_full, t_band, t_full/max(t_band,1e-6)))
print("字幕时间线（前 14 条）:")
for e in r["results"][:14]:
    print("   %s  %s  %s" % (e["timestamp"], "跳过" if e["skipped"] else "识别", " ".join(e["lines"])))
