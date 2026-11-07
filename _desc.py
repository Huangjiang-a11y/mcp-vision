import pathlib, ast
p = pathlib.Path("/data/data/com.termux/files/home/mcp-vision/mcp_server.py")
s = p.read_text()

OLD = '''         "frame - one moment, or a short burst after it.\\n"
         "Without `seconds` you get the single frame at `time`. With `seconds` (and optionally `quantity`, "
         "default 1) it becomes a burst: `quantity` frames evenly spaced from `time` to `time`+`seconds`, each "
         "cropped the same way, returned as separate images plus items[{t,path}].\\n"
         "Example: time=123.5, crop=1280x260+0+440 returns that slice as an image plus its saved path; "
         "time=42, seconds=2, quantity=4 returns frames at 42.0 / 42.667 / 43.333 / 44.0. Use frame when the "
         "page is too small to read a board, a scoreboard or a subtitle line.\\n"'''

NEW = '''         "frame - one moment, or a short burst as a single page.\\n"
         "Without `seconds` you get the one frame at `time`, at the full pixel cap. With `seconds` the burst is "
         "composed into ONE page instead of N images: `quantity` frames (default 1) evenly spaced from `time` to "
         "`time`+`seconds`, laid out like analyze, one image back plus a timeline. Files are prefixed strip_ so "
         "you can tell them from analyze's sheet_*.\\n"
         "Example: time=123.5, crop=1280x260+0+440 returns that slice plus its path; time=42, seconds=2, "
         "quantity=4 gives a 2x2 page of 42.0 / 42.667 / 43.333 / 44.0, each cell 683x384. Use frame when the "
         "page is too small to read a board, a scoreboard or a subtitle line.\\n"'''
assert OLD in s, "frame 段没匹配上"
s = s.replace(OLD, NEW, 1)

OLD2 = '''"For analyze it caps the whole page (cells get smaller as the grid grows); for frame it caps the cropped area. "'''
NEW2 = '''"For analyze and for a frame burst it caps the whole page (cells get smaller as the grid grows); for a "
         "single frame it caps the cropped area. "'''
assert OLD2 in s, "quality 说明没匹配上"
s = s.replace(OLD2, NEW2, 1)

OLD3 = '''"(frames t<ms>-<w>x<h>.jpg, page sheet_*.png, single frame_<ms>-<hash>.jpg) and repeat calls reuse them.\\n"'''
NEW3 = '''"(frames t<ms>-<w>x<h>.jpg, analyze page sheet_*.png, burst page strip_*.png, single frame_<ms>-<hash>.jpg) "
         "and repeat calls reuse them.\\n"'''
assert OLD3 in s, "输出目录说明没匹配上"
s = s.replace(OLD3, NEW3, 1)

ast.parse(s)
p.write_text(s)
print("描述已更新")
