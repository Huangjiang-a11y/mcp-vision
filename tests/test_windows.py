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

print("\nRESULT", "ALL OK" if not fails else "HAS FAIL (%d)" % len(fails))
for f in fails:
    print(" -", f)
sys.exit(0 if not fails else 1)
