"""取窗与参数校验的回归测试。

这一组全是纯逻辑（不碰模型、不碰素材），跑得很快。
每个用例对应一个实际踩过的 bug，注释里写了当时的现象。
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mcp_server as M

ASSET = {"duration": 276.0, "width": 1280, "height": 720}
fails = []


def check(name, got, want):
    if got != want:
        fails.append("%s\n    期望 %r\n    实际 %r" % (name, want, got))
        print("  FAIL %s" % name)
    else:
        print("  ok   %s" % name)


# --- seconds <= 0 ---
# 以前 video/subtitles 不查，seconds=0 会静默退化成 1 帧 / 0 段，
# 0 段还被解释成「字幕可能在上方」，让人白试半天 band。
check("seconds=0 被拒", M._pos_seconds({"seconds": 0}) is not None, True)
check("seconds=-5 被拒", M._pos_seconds({"seconds": -5}) is not None, True)
check("seconds 缺省放行", M._pos_seconds({}), None)
check("seconds 非数字被拒", M._pos_seconds({"seconds": "abc"}) is not None, True)

# --- time 越界 ---
# 以前会同时给出「已到视频结尾」和「回退到 0」，自相矛盾。
s, e, note = M._window({"time": 99999}, ASSET)
check("time 越界回到 0", (s, e), (0.0, 276.0))
check("time 越界只有一条 note", note.count("；"), 0)
check("time 越界 note 提回退", "回退到 0" in note, True)
check("time 越界不提结尾", "已到视频结尾" in note, False)

# --- time 负数 ---
s, e, note = M._window({"time": -10}, ASSET)
check("负 time 从 0 开始", s, 0.0)
check("负 time 有提示", "负数" in note, True)

# --- 正常窗口 ---
s, e, note = M._window({"time": 10, "seconds": 5}, ASSET)
check("正常窗口", (s, e, note), (10.0, 15.0, ""))

# --- 超过上限 ---
# 注意用 2000 秒的长片测：276 秒的片子会先夹上限再夹结尾，
# 断言 900 会失败，看不出究竟是哪一步生效。
LONG = {"duration": 2000.0, "width": 1280, "height": 720}
s, e, note = M._window({"time": 0, "seconds": 5000}, LONG)
check("超出上限被夹", e, 900.0)
check("上限有提示", "上限" in note, True)
# 短片：600 秒请求落在 276 秒里，应该只夹结尾，不提上限
s, e, note = M._window({"time": 0, "seconds": 600}, ASSET)
check("短片夹结尾", e, 276.0)
check("短片不提上限", "上限" in note, False)

# --- crop 越界 ---
# 以前 ffmpeg 不报错，只吐一张 295 字节的黑图，还标 ok=True。
box = M._crop_box("100x100+99999+0")
check("_crop_box 解析", box, (100, 100, 99999, 0))
check("_crop_box 非法返回 None", M._crop_box("abc"), None)
try:
    M._parse_crop("0x0+0+0")
    fails.append("crop 零尺寸没报错")
except ValueError:
    print("  ok   crop 零尺寸被拒")

# --- JSON-RPC ---
check("未知工具用 -32602", M.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                    "params": {"name": "nope"}})["error"]["code"], -32602)
check("未知方法用 -32601", M.handle({"jsonrpc": "2.0", "id": 1, "method": "nope"})["error"]["code"], -32601)
check("非法请求用 -32600", M.handle(["x"])["error"]["code"], -32600)
check("parse error 用 -32700", M._parse_error("x")["error"]["code"], -32700)

# --- 时间格式：只到秒 ---
# 毫秒对定位没用，每段却多 8 个字符。98 段字幕就是 784 字符。
check("_tc 到秒", M._tc(83.47), "01:23")
check("_tc 四舍五入", M._tc(59.6), "01:00")
check("_tc 零", M._tc(0), "00:00")
check("_tc 不含毫秒", "." in M._tc(12.345), False)
check("_tc 负数归零", M._tc(-5), "00:00")

# --- SenseVoice 富标签解析 ---
import engine
check("标签剥壳", engine._tag("<|NEUTRAL|>"), "NEUTRAL")
check("标签空值", engine._tag(None), "")
check("标签 NONE", engine._tag("<|NONE|>"), "")
check("标签普通字符串", engine._tag("zh"), "zh")

# --- 单帧取窗：别为不存在的窗口报警 ---
# 以前 quantity=1 时仍走完整窗口逻辑，t=100 会附赠一句「已到视频结尾 276 秒」，
# 但单帧只用起点，结尾根本不参与取图。
s, e, note = M._window({"time": 100}, ASSET, single=True)
check("单帧不报结尾", note, "")
check("单帧起点终点重合", (s, e), (100.0, 100.0))
check("单帧忽略 seconds 上限", M._window({"time": 100, "seconds": 5000}, ASSET, single=True)[2], "")
check("单帧仍报负时间", "负数" in M._window({"time": -5}, ASSET, single=True)[2], True)
check("单帧仍报越界回退", "回退到 0" in M._window({"time": 99999}, ASSET, single=True)[2], True)
# 多帧时该报的还得报
check("多帧报结尾", "已到视频结尾" in M._window({"time": 270, "seconds": 10}, ASSET)[2], True)

# --- 声学字段的边界 ---
# emotion / event / lang 来自 SenseVoice，只有 stt 模式有。
# OCR 读像素，原理上取不到——这条边界要是破了，就是编造数据。
import inspect
src_ocr = inspect.getsource(M._subs_ocr)
src_stt = inspect.getsource(M._subs_stt)
check("OCR 不写 emotion", "emotion" in src_ocr, False)
check("OCR 不写 event", '"event"' in src_ocr, False)
# 字段语义写在 schema 里（只付一次），不进每次响应——调用方自己知道传的哪个 mode。
check("OCR 不重复声明字段缺失", "acoustic_fields" in src_ocr, False)
check("STT 不重复声明字段存在", "acoustic_fields" in src_stt, False)
check("STT 确实带 emotion", '"emotion"' in src_stt, True)
# note 只能有一处赋值。之前 window_note 那条在后面无条件覆盖，BGM 提示被吞掉了。
check("STT 的 note 只赋值一次", src_stt.count('out["note"] ='), 1)

# --- 响应里不许回显已知事实 ---
# schema 是「调用前读一次」，响应是「每次调用都读」。
# 已知不变的东西进前者；把已知事实塞回响应，等于每次付费买一遍废话。
for bad, why in [
    ('"engine"', "stt 只有一种引擎，属 schema"),
    ('"range"', "就是调用方传的 time/seconds；被夹了 note 会说"),
    ('"text": "".join', "等于把 segments 再拼一遍"),
    ('band_auto', "恒等于 band，零额外信息"),
]:
    check("STT 不回显 %s" % bad, bad in src_stt, False)
# 查 JSON 键而不是裸词——注释里提到字段名是合法的，写进输出才是问题
check("OCR 不输出 band_auto 键", '"band_auto":' in src_ocr, False)

print("\nRESULT", "ALL OK" if not fails else "HAS FAIL (%d)" % len(fails))
for f in fails:
    print(" -", f)
sys.exit(0 if not fails else 1)
