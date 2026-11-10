"""字幕行筛选（_keep_wide）的回归测试。

纯逻辑：喂检测框、看留下哪几行，不碰模型也不碰素材。
两条都是踩过的坑：
- 「呦呦呦耶耶耶」只有最宽行的 0.44 倍宽，被 top*0.5 判成噪音丢掉；
- 高度基准取 max 时，检测器把两行并成一个高框，0.55×两倍行高就把同帧
  所有正常字幕都判成"小字"（实测 200 帧里 4 帧整句丢失）。
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ocr

W = 1280
fails = []


def check(name, got, want):
    if got != want:
        fails.append("%s\n    期望 %r\n    实际 %r" % (name, want, got))
        print("  FAIL %s" % name)
    else:
        print("  ok   %s" % name)


def box(y0, h, w, x=0):
    return {"box": [x, y0, x + w, y0 + h], "text": "t", "conf": 0.9}


# 0-3 秒那帧真实的三行：短句 218、中句 442、长句 558
three = [box(34, 29, 218), box(69, 36, 442), box(111, 31, 558)]
check("短句不被丢（218 vs 558）", len(ocr._keep_wide(three, W)), 3)

# 横跨画面的超宽离群框（水印条 / 整行横线）不该把宽度标尺抬高，
# 否则正常字幕行全被判窄丢掉
wide = [box(0, 40, 1200), box(60, 36, 440), box(100, 36, 438)]
check("超宽离群框不抬高标尺", len(ocr._keep_wide(wide, W)), 3)

# 真正的窄小字（正文）仍该丢：宽 60px、高 12px
check("窄小字仍被丢", len(ocr._keep_wide(three + [box(200, 12, 60)], W)), 3)

# 边界：空 / 单框原样返回
check("空列表原样", ocr._keep_wide([], W), [])
one = [box(10, 30, 300)]
check("单框原样", ocr._keep_wide(one, W), one)

# 全部都很窄时不清空（宁可留噪音，也别整帧丢字幕）
tiny = [box(0, 14, 40), box(20, 14, 44)]
check("都窄时不清空", len(ocr._keep_wide(tiny, W)), 2)

print("\nRESULT", "ALL OK" if not fails else "HAS FAIL (%d)" % len(fails))
for f in fails:
    print(" -", f)
sys.exit(0 if not fails else 1)
