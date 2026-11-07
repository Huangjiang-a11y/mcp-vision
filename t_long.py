import sys, time; sys.path.insert(0,".")
import ocr
D="workspace/“太好了是《琵琶行》！！学生党有救了！不懂的人有难了！！”/BV1Rx54znE7k_p1.mp4"
eng=ocr.get_engine(); print("总时长", round(ocr._probe_duration(D),1), "秒")
for s0,s1 in ((0,16),(0,328),(60,328)):
    t0=time.time(); b=ocr.detect_subtitle_band(D, samples=10, start=s0, end=s1, engine=eng)
    print("窗口 %d-%ds -> %-22s %.2fs" % (s0,s1,b,time.time()-t0))
