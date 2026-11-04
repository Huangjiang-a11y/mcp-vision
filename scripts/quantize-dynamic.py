#!/usr/bin/env python3
"""真·动态量化：DynamicQuantizeLinear + QLinearConv（per-tensor）。
bias 不融进整数卷积，改在反量化后用 float Add 补（动态量化下 int32 bias 无法预换算）。"""
import sys, os, numpy as np
sys.path.insert(0, "tools/onnxgen")
import onnx_pb2 as onnx
FLOAT, INT8, UINT8 = 1, 3, 2


def _T(name, dtype, dims, data):
    t = onnx.TensorProto(); t.name = name; t.data_type = dtype
    t.dims.extend(dims); t.raw_data = data
    return t


def _N(op, ins, outs, name=""):
    n = onnx.NodeProto(); n.op_type = op; n.name = name
    n.input.extend(ins); n.output.extend(outs)
    return n


def quantize(model, verbose=True):
    g = model.graph
    consts = {}
    for n in g.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value" and a.HasField("t"):
                    consts[n.output[0]] = (n, a.t)
    inits, plan = [], {}
    used_consts = set()
    for n in g.node:
        if n.op_type != "Conv" or len(n.input) not in (2, 3):
            continue
        if n.input[1] not in consts:
            continue
        cn, wt = consts[n.input[1]]
        if wt.data_type != FLOAT:
            continue
        w = np.frombuffer(wt.raw_data, dtype=np.float32).reshape(list(wt.dims))
        amax = float(np.max(np.abs(w)))
        if amax <= 0:
            continue
        ws = amax / 127.0
        wq = np.clip(np.round(w / ws), -127, 127).astype(np.int8)
        tag = "dn_%d" % len(plan)
        inits += [
            _T(tag + "_wq", INT8, list(wt.dims), wq.tobytes()),
            _T(tag + "_ws", FLOAT, [], np.array([ws], dtype=np.float32).tobytes()),
            _T(tag + "_wz", INT8, [], np.array([0], dtype=np.int8).tobytes()),
            _T(tag + "_yz", UINT8, [], np.array([0], dtype=np.uint8).tobytes()),
        ]
        plan[n.output[0]] = (tag, None, None)
        used_consts.add(n.input[1])

    if not plan:
        return 0
    # 重接下游（含图输出）
    for other in g.node:
        for i, inp in enumerate(other.input):
            if inp in plan:
                other.input[i] = plan[inp][0] + "_out"
    for o in g.output:
        if o.name in plan:
            o.name = plan[o.name][0] + "_out"
    for vi in g.value_info:
        if vi.name in plan:
            vi.name = plan[vi.name][0] + "_out"

    out_nodes = []
    for n in g.node:
        if n.op_type == "Constant" and n.output[0] in used_consts:
            continue
        if n.op_type == "Conv" and n.output[0] in plan:
            tag = plan[n.output[0]][0]
            x = n.input[0]                      # 重接后的名字才是有效的
            bias = n.input[2] if len(n.input) == 3 else None
            seq = [
                _N("DynamicQuantizeLinear", [x], [tag + "_xq", tag + "_xs", tag + "_xz"], tag + "_dql"),
                _N("Mul", [tag + "_xs", tag + "_ws"], [tag + "_ys"], tag + "_mul"),
                _N("QLinearConv",
                   [tag + "_xq", tag + "_xs", tag + "_xz", tag + "_wq", tag + "_ws",
                    tag + "_wz", tag + "_ys", tag + "_yz"], [tag + "_yq"], tag + "_qlc"),
                _N("DequantizeLinear", [tag + "_yq", tag + "_ys", tag + "_yz"], [tag + "_dq"], tag + "_deq"),
            ]
            # 原 Conv 的属性（group/pads/strides/dilations/kernel_shape/auto_pad）必须照搬
            for a in n.attribute:
                seq[2].attribute.add().CopyFrom(a)
            if bias and bias in consts:
                bt = consts[bias][1]
                bnp = np.frombuffer(bt.raw_data, dtype=np.float32).reshape(-1)
                bn = tag + "_biasf"
                inits.append(_T(bn, FLOAT, [1, int(bnp.shape[0]), 1, 1],
                                bnp.reshape(1, -1, 1, 1).astype(np.float32).tobytes()))
                seq.append(_N("Add", [tag + "_dq", bn], [tag + "_out"], tag + "_bias"))
            else:
                seq[3].output[0] = tag + "_out"
            out_nodes.extend(seq)
        else:
            out_nodes.append(n)
    g.ClearField("node")
    g.node.extend(out_nodes)
    g.initializer.extend(inits)
    if verbose:
        print("动态量化 %d 个 Conv（含 bias 走 float Add）" % len(plan))
    return len(plan)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    m = onnx.ModelProto(); m.ParseFromString(open(src, "rb").read())
    quantize(m)
    open(dst, "wb").write(m.SerializeToString())
    print("%.2fMB → %.2fMB" % (os.path.getsize(src)/1e6, os.path.getsize(dst)/1e6))


if __name__ == "__main__":
    main()
