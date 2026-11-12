"""整页文字 vs 字幕的判定回归。

判据：可用簇 ≥ 4 且纵向跨度 ≥ 0.45；另外选中的带本身像不像一条正经字幕。
这里用真实测过的四个分布做固定输入，不依赖素材和模型。
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ocr

fails = []


def boxes_from(rows, n_per_row=2, w=0.4):
    """把「若干行的 y 位置」摊成框列表。每行放 n_per_row 个框，模拟表格单元格。"""
    out = []
    for y in rows:
        for k in range(n_per_row):
            out.append((y, y + 0.05, w, "cell%d" % k))
    return out


def band(y0, y1, w, count):
    """模拟 _band_from_boxes 的返回值里 _page_like 会用到的那几项。"""
    return {"band": (y0, y1), "w": w, "count": count}


def rows(spec):
    """spec: [(y0, 框数, 宽), ...] —— 簇数看行数，簇出现次数看框数。"""
    out = []
    for y, n, w in spec:
        for k in range(n):
            out.append((y, y + 0.05, w, "t%d" % k))
    return out


def check(name, got, want):
    if got != want:
        fails.append("%s 期望 %r 实际 %r" % (name, want, got))
        print("  FAIL %s" % name)
    else:
        print("  ok   %s" % name)


# 字幕：一行字，八个采样点位置几乎不变
sub = boxes_from([0.86], n_per_row=1)
ok, ncl, span = ocr._page_like(sub)
check("单行字幕不是整页", ok, False)
check("单行字幕簇数", ncl, 1)

# 两行字幕（双语）也一样
sub2 = boxes_from([0.86, 0.91], n_per_row=1)
check("两行字幕不是整页", ocr._page_like(sub2)[0], False)

# 幻灯片：6 处各自成行，跨 0.19–0.84
slide = boxes_from([0.19, 0.26, 0.40, 0.49, 0.58, 0.68, 0.84], n_per_row=2)
ok, ncl, span = ocr._page_like(slide)
check("幻灯片判为整页", ok, True)
check("幻灯片簇数够", ncl >= 4, True)

# 作文纸：十几行铺满，跨 0.01–0.94
paper = boxes_from([0.01, 0.09, 0.16, 0.25, 0.32, 0.43, 0.50, 0.60,
                    0.64, 0.71, 0.81, 0.85], n_per_row=2)
ok, ncl, span = ocr._page_like(paper)
check("作文纸判为整页", ok, True)
check("作文纸跨度大", span > 0.8, True)

# 簇多但都挤在一处（比如底部三行小字）→ 跨度不够，不算整页
crowded = boxes_from([0.86, 0.88, 0.90, 0.92, 0.94], n_per_row=1, w=0.3)
check("簇多但集中不算整页", ocr._page_like(crowded)[0], False)

# 簇少但分散 → 也不算整页（宁可漏判成字幕，也不误伤正常片子）
scattered = boxes_from([0.1, 0.5, 0.9], n_per_row=1)
check("簇少分散不算整页", ocr._page_like(scattered)[0], False)

# 窄框（单个字母/图标）不计入可用簇
narrow = boxes_from([0.1, 0.3, 0.5, 0.7, 0.9], n_per_row=1, w=0.05)
check("窄框不计入", ocr._page_like(narrow)[1], 0)

# 空输入
check("空输入不是整页", ocr._page_like([])[0], False)

# --- 水印/图表标签凑出来的假整页 ---
# 柴知道 653s 那支片：底部一条真字幕带（宽 0.37，8 个采样帧都在），
# 上面散着台标和图表标签（宽 0.13–0.20，随镜头换高度）。6 个 100s 窗口
# 全被判成整页，而字幕读得好好的。4-5 个垃圾簇不该算「各自成行」。
knows = rows([(0.22, 2, 0.20), (0.44, 1, 0.17), (0.60, 2, 0.13),
              (0.72, 2, 0.16), (0.86, 8, 0.37)])
ok, ncl, span = ocr._page_like(knows, band(0.827, 0.962, 0.40, 8), samples=8)
check("台标凑 5 个簇不算整页", ok, False)
check("台标场景仍报簇数", ncl, 5)

# --- 选中带本身像字幕时的豁免 ---
six = rows([(0.20, 2, 0.40), (0.32, 2, 0.40), (0.44, 2, 0.40),
            (0.56, 2, 0.40), (0.70, 2, 0.40), (0.86, 8, 0.37)])
check("满屏文字 + 底部干净字幕带 → 不算整页",
      ocr._page_like(six, band(0.84, 0.94, 0.40, 8), samples=8)[0], False)
check("没给带时照旧算整页", ocr._page_like(six)[0], True)
check("带不在底部不豁免",
      ocr._page_like(six, band(0.30, 0.40, 0.40, 8), samples=8)[0], True)

# 带本身不像字幕（不在底部）→ 该报的还要报。这是柴知道 200-300 那段的形状：
# band 0.559-0.845、宽 0.63、count 7，读出来 11 段全是图注（USB-2.0 / 640x480）
mid_band = rows([(0.20, 2, 0.20), (0.44, 1, 0.17), (0.60, 2, 0.13),
                 (0.72, 2, 0.16), (0.86, 7, 0.63)])
check("带在画面中部不豁免（图注场景）",
      ocr._page_like(mid_band, band(0.559, 0.845, 0.63, 7), samples=8)[0], True)

narrow_bottom = rows([(0.20, 2, 0.40), (0.32, 2, 0.40), (0.44, 2, 0.40),
                      (0.56, 2, 0.40), (0.70, 2, 0.40), (0.86, 8, 0.14)])
check("底部那行太窄不豁免",
      ocr._page_like(narrow_bottom, band(0.84, 0.94, 0.14, 8), samples=8)[0], True)

rare_bottom = rows([(0.20, 2, 0.40), (0.32, 2, 0.40), (0.44, 2, 0.40),
                    (0.56, 2, 0.40), (0.70, 2, 0.40), (0.86, 1, 0.37)])
check("底部那行只出现一次不豁免",
      ocr._page_like(rare_bottom, band(0.84, 0.94, 0.37, 1), samples=8)[0], True)

print("\nRESULT", "ALL OK" if not fails else "HAS FAIL (%d)" % len(fails))
for f in fails:
    print(" -", f)
sys.exit(0 if not fails else 1)
