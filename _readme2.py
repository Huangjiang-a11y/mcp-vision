import pathlib
p = pathlib.Path("/data/data/com.termux/files/home/mcp-vision/README.md")
s = p.read_text()
OLD = "- `video`：`analyze`（整片：时间线抽帧 + 近似正方形网格页）/ `frame`（单帧，可 `crop`）"
NEW = ("- `video`：`analyze`（整片：时间线抽帧 + 近似正方形网格页）/ `frame`（单帧，可 `crop`；"
       "给了 `seconds` 就变成连拍，同样拼成一页 `strip_*.png`）")
assert OLD in s
s = s.replace(OLD, NEW, 1)
p.write_text(s)
print("ok")
