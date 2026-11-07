import sys, time; sys.path.insert(0,".")
import ocr
D="workspace/“太好了是《琵琶行》！！学生党有救了！不懂的人有难了！！”/BV1Rx54znE7k_p1.mp4"
eng=ocr.get_engine()
for s0,s1 in ((0,16),(0,328),(0,5)):
    t0=time.time(); b=ocr.detect_subtitle_band(D, samples=10, start=s0, end=s1, engine=eng)
    print("检测 %d-%ds -> %-20s %.2fs" % (s0,s1,b,time.time()-t0))
t0=time.time(); r=ocr.ocr_subtitles(D,0,16,step=1.0,band="auto",engine=eng)
print("0-16 auto: %.2fs band %s 段 %d" % (time.time()-t0, r["band"], len(ocr.fold_segments(r["results"],step=1.0))))
t0=time.time(); r2=ocr.ocr_subtitles(D,0,5,step=1.0,band="auto",engine=eng)
print("0-5  auto: %.2fs band %s 段 %d note %s" % (time.time()-t0, r2["band"], len(r2["results"]), r2.get("note")))
