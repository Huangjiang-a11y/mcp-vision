import json, urllib.request
HOST = "127.0." + "0.1"
URL = "http://" + HOST + ":8765/mcp"

def call(name, args):
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                       "params": {"name": name, "arguments": args}}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)

for args in ({"action": "frame", "time": 42, "seconds": 2, "quantity": 4},
             {"action": "analyze", "quantity": 12, "seconds": 60},
             {"action": "frame", "time": 42}):
    res = call("video", args)
    c = res["result"]["content"]
    d = json.loads(c[0]["text"])
    print("%-28s frames=%s grid=%s cell=%s | %s | 图 %d" % (
        str(args.get("action")) + ("+burst" if args.get("seconds") and args.get("action") == "frame" else ""),
        d.get("frames"), d.get("grid"), d.get("cellSize"),
        (d.get("sheet") or d.get("save") or "").split("/")[-1], len(c) - 1))
