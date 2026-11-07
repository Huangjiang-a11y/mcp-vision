import json, threading, time, urllib.request

URL = "http://127.0.0.1:8765"

def call(tool, args):
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                       "params": {"name": tool, "arguments": args}}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        urllib.request.urlopen(req, timeout=600).read()
    except Exception as e:
        return -1.0, str(e)
    return time.time() - t0, ""

BAND = {"action": "band", "start_seconds": 0, "seconds": 16, "step": 1}
ASR = {"action": "transcribe", "start_seconds": 30, "seconds": 12}

def run(label, jobs):
    out = [None] * len(jobs)
    def w(i, tool, args):
        out[i] = call(tool, args)
    ths = [threading.Thread(target=w, args=(i,) + j) for i, j in enumerate(jobs)]
    t0 = time.time()
    [t.start() for t in ths]; [t.join() for t in ths]
    wall = time.time() - t0
    print("%-28s 墙钟 %5.2fs  各自 %s" % (label, wall, ["%.2f" % o[0] for o in out]))

run("单个 band", [("subtitles", BAND)])
run("两个 band（同工具）", [("subtitles", BAND)] * 2)
run("两个 band（同工具）二轮", [("subtitles", BAND)] * 2)
run("band + transcribe（不同工具）", [("subtitles", BAND), ("audio", ASR)])
