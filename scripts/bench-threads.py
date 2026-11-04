import time, sys, numpy as np, onnxruntime as ort
sys.path.insert(0, ".")
import ocr as O
rgb = O.load_rgb("out/ocr_test.png")
MODELS = {"det": "models/ch_PP-OCRv4_det_infer.onnx", "rec": "models/ch_PP-OCRv4_rec_infer.onnx"}
H0, W0 = rgb.shape[:2]
scale = min(1.0, 720/max(W0, H0))
W = max(32, 32*int(round(W0*scale/32))); H = max(32, 32*int(round(H0*scale/32)))
small = O.resize_bilinear(rgb, W, H).astype(np.float32)/255.0
small = (small - O.MEAN)/O.STD
xD = np.transpose(small, (2,0,1))[None].astype(np.float32)
crop = O.resize_bilinear(rgb[221:251, 45:413], 320, 48).astype(np.float32)/255.0 - 0.5
xR = np.transpose(crop, (2,0,1))[None].astype(np.float32)

print("%-8s %10s %10s %10s" % ("threads", "det", "rec", "init合计"))
for th in [1, 2, 3, 4, 6, 8]:
    so = ort.SessionOptions(); so.intra_op_num_threads = th; so.log_severity_level = 3
    t0 = time.time()
    sd = ort.InferenceSession(MODELS["det"], so, providers=["CPUExecutionProvider"])
    sr = ort.InferenceSession(MODELS["rec"], so, providers=["CPUExecutionProvider"])
    init = time.time() - t0
    nd, nr = sd.get_inputs()[0].name, sr.get_inputs()[0].name
    sd.run(None, {nd: xD}); sr.run(None, {nr: xR})   # warmup
    t0 = time.time()
    for _ in range(3): sd.run(None, {nd: xD})
    det = (time.time()-t0)/3
    t0 = time.time()
    for _ in range(5): sr.run(None, {nr: xR})
    rec = (time.time()-t0)/5
    print("%-8d %10.3f %10.3f %10.3f" % (th, det, rec, init))
