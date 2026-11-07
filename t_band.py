import sys, time; sys.path.insert(0,".")
import ocr
D="workspace/“太好了是《琵琶行》！！学生党有救了！不懂的人有难了！！”/BV1Rx54znE7k_p1.mp4"
eng=ocr.get_engine()
for i in range(2):
    t0=time.time(); b=ocr.detect_subtitle_band(D, samples=10, start=0, end=16, engine=eng)
    print("检测 第%d次 %.2fs -> %s" % (i+1, time.time()-t0, b))
t0=time.time(); r=ocr.ocr_subtitles(D,0,16,step=1.0,band=(0.8652,0.97),engine=eng)
print("带内识别(16帧) %.2fs OCR %d 次" % (time.time()-t0, r["ocr_runs"]))
