import time, sys, numpy as np, onnxruntime as ort
sys.path.insert(0, ".")
import ocr as O

rgb = O.load_rgb("out/ocr_test.png")
print("可用 providers:", ort.get_available_providers())
print()
print("%-38s %8s %8s %8s %8s" % ("provider", "init", "det", "BFS", "rec/box"))

CANDIDATES = [
    ["CPUExecutionProvider"],
    ["XnnpackExecutionProvider", "CPUExecutionProvider"],
    ["NnapiExecutionProvider", "CPUExecutionProvider"],
]
for prov in CANDIDATES:
    name = "+".join(p.replace("ExecutionProvider", "") for p in prov)
    try:
        t0 = time.time(); eng = O.PPOCR(providers=prov); init = time.time() - t0
    except Exception as e:
        print("%-38s  跳过: %s" % (name, str(e)[:40])); continue
    try:
        # warmup
        eng.detect(rgb); eng.recognize(rgb, [45, 221, 412, 250])
        t0 = time.time(); dets = eng.detect(rgb); det = time.time() - t0
        # BFS 单独
        H0, W0 = rgb.shape[:2]
        scale = min(1.0, 720 / max(W0, H0))
        W = max(32, 32*int(round(W0*scale/32))); H = max(32, 32*int(round(H0*scale/32)))
        small = O.resize_bilinear(rgb, W, H).astype(np.float32)/255.0
        small = (small - O.MEAN)/O.STD
        x = np.transpose(small, (2,0,1))[None].astype(np.float32)
        prob = np.asarray(eng.det.run(None, {eng.det.get_inputs()[0].name: x})[0]).reshape(H, W)
        t0 = time.time(); eng._connected_boxes(prob > 0.2, W, H); bfs = time.time() - t0
        t0 = time.time()
        for b in dets: eng.recognize(rgb, b)
        rec = (time.time() - t0) / max(1, len(dets))
        print("%-38s %7.3f %8.3f %8.3f %8.3f  (%d框)" % (name, init, det, bfs, rec, len(dets)))
    except Exception as e:
        print("%-38s  失败: %s" % (name, str(e)[:50]))
