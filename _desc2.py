import pathlib, ast
p = pathlib.Path("/data/data/com.termux/files/home/mcp-vision/mcp_server.py")
lines = p.read_text().split("\n")

def find(needle, start=0):
    for k in range(start, len(lines)):
        if needle in lines[k]:
            return k
    raise SystemExit("没找到: " + needle)

# 1) frame 段：从 "frame - one moment" 到 "subtitle line.\n"
a = find('"frame - one moment')
b = find('page is too small to read a board')
assert b >= a
new_frame = [
    '         "frame - one moment, or a short burst as a single page.\\n"',
    '         "Without `seconds` you get the one frame at `time`, at the full pixel cap. With `seconds` the burst "',
    '         "is composed into ONE page instead of N images: `quantity` frames (default 1) evenly spaced from "',
    '         "`time` to `time`+`seconds`, laid out like analyze, one image back plus a timeline. Files are "',
    '         "prefixed strip_ so you can tell them from analyze\'s sheet_*.\\n"',
    '         "Example: time=123.5, crop=1280x260+0+440 returns that slice plus its path; time=42, seconds=2, "',
    '         "quantity=4 gives a 2x2 page of 42.0 / 42.667 / 43.333 / 44.0, each cell 683x384. Use frame when "',
    '         "the page is too small to read a board, a scoreboard or a subtitle line.\\n"',
]
lines[a:b + 1] = new_frame

# 2) quality 说明：整页 vs 单帧
c = find('analyze it caps the whole page')
d = find('cropped area. It never upscales.', c)
lines[c:d + 1] = [
    '         "analyze or a frame burst it caps the whole page (cells get smaller as the grid grows); for a "',
    '         "single frame it caps the cropped area. It never upscales. The window is time + seconds, capped "',
]
# 上一行末尾的 "For " 改成 "For " 保持，检查一下上下文
e = find('workspace/<video>/video/ (frames t<ms>')
lines[e] = '         "workspace/<video>/video/ (frames t<ms>-<w>x<h>.jpg, analyze page sheet_*.png, burst page "'
lines[e + 1] = '         "strip_*.png, single frame_<ms>-<hash>.jpg) and repeat calls reuse them."'

s = "\n".join(lines)
ast.parse(s)
p.write_text(s)
print("描述已更新")
