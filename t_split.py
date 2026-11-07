import sys, time; sys.path.insert(0,".")
import ocr, numpy as np
D="workspace/“太好了是《琵琶行》！！学生党有救了！不懂的人有难了！！”/BV1Rx54znE7k_p1.mp4"
eng=ocr.get_engine(); W=960
# 1) 只解码 10 帧（不 OCR）
t0=time.time(); frames=[]
for i,_t,arr in ocr._stream_frames(D,0,16,step=16/11.0,width=W,crop=None):
    if i==0: continue
    frames.append(arr[int(arr.shape[0]*0.6667):])
    if len(frames)>=10: break
print("解码 10 帧: %.2fs  (%d 帧, 每帧 %dx%d)" % (time.time()-t0, len(frames), frames[0].shape[1], frames[0].shape[0]))
# 2) 只 OCR
for w in (960, 640, 480):
    sub=[f for f in frames]
    if w!=960:
        import cv2
        sub=[cv2.resize(f,(w,int(f.shape[0]*w/960))) for f in sub]
    t0=time.time()
    for f in sub: eng.ocr_image(f)
    print("OCR 10 帧 @宽%d (高%d): %.2fs  -> 每帧 %.2fs" % (w, sub[0].shape[0], time.time()-t0, (time.time()-t0)/10))
