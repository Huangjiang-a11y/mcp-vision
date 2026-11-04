#!/usr/bin/env python3
"""自量化探针：把 Conv 权重压成 int8（QDQ，per-tensor），看 ORT 认不认、快不快。"""
import sys, time, numpy as np
sys.path.insert(0, "tools/onnxgen")
import onnx_pb2 as onnx

FLOAT, INT8 = 1, 3


def qdq_conv_weights(model):
    g = model.graph
    consts = {}
    for n in g.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value" and a.HasField("t"):
                    consts[n.output[0]] = (n, a.t)

    new_inits = []
    added = 0
    for n in g.node:
        if n.op_type not in ("Conv", "ConvTranspose"):
            continue
        if len(n.input) < 2 or n.input[1] not in consts:
            continue
        cn, wt = consts[n.input[1]]
        if wt.data_type != FLOAT or len(wt.dims) < 3:
            continue
        w = np.frombuffer(wt.raw_data, dtype=np.float32).reshape(list(wt.dims)).astype(np.float32)
        amax = float(np.max(np.abs(w)))
        if amax <= 0:
            continue
        scale = amax / 127.0
        wq = np.clip(np.round(w / scale), -127, 127).astype(np.int8)

        base = "qw_%s" % n.input[1]
        tq = onnx.TensorProto()
        tq.name = base + "_q"; tq.data_type = INT8; tq.dims.extend(list(wt.dims))
        tq.raw_data = wq.tobytes()
        ts = onnx.TensorProto()
        ts.name = base + "_s"; ts.data_type = FLOAT; ts.dims.append(1)
        ts.raw_data = np.array([scale], dtype=np.float32).tobytes()
        new_inits += [tq, ts]

        dq = onnx.NodeProto()
        dq.op_type = "DequantizeLinear"
        dq.name = "dq_%s" % n.input[1]
        dq.input.extend([tq.name, ts.name])
        dq.output.append(base + "_f")
        n.input[1] = base + "_f"
        # 把 DQ 插到 Constant 之后
        idx = list(g.node).index(cn)
        g.node.insert(idx + 1, dq)
        g.node.remove(cn)
        added += 1
    g.initializer.extend(new_inits)
    return added


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "models/ch_PP-OCRv4_det_infer.onnx"
    dst = sys.argv[2] if len(sys.argv) > 2 else src.replace(".onnx", "_w8.onnx")
    m = onnx.ModelProto(); m.ParseFromString(open(src, "rb").read())
    n = qdq_conv_weights(m)
    open(dst, "wb").write(m.SerializeToString())
    import os
    print("量化了 %d 个卷积 | %s: %.2fMB → %s: %.2fMB" % (
        n, src, os.path.getsize(src)/1e6, dst, os.path.getsize(dst)/1e6))


if __name__ == "__main__":
    main()
