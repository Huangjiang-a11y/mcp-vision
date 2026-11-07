import pathlib, ast
p = pathlib.Path("/data/data/com.termux/files/home/mcp-vision/mcp_server.py")
s = p.read_text()
s = s.replace('capped "\n         "(900) and clamped', 'capped at max_seconds "\n         "(900) and clamped', 1)
s = s.replace('"analyze = grid page for the clip (default); frame = one frame"',
              '"analyze = grid page for the clip (default); frame = one frame, or a burst page when seconds is given"', 1)
ast.parse(s)
p.write_text(s)
print("ok")
