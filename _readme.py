import pathlib
p = pathlib.Path("/data/data/com.termux/files/home/mcp-vision/README.md")
s = p.read_text()
OLD = """| `time` | 从哪个时间戳开始，默认 0（frame 里是那一帧的时刻） |"""
NEW = """| `time` | 从哪个时间戳开始，默认 0（frame 里是第一帧的时刻） |"""
if OLD in s:
    s = s.replace(OLD, NEW, 1)
p.write_text(s)
print("readme ok")
