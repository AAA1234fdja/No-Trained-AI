#!/usr/bin/env python3
"""
Stage 5 Weight Edit: Antipodal Oscillator Fix + Final Calibrated 800M INT8 Safetensors.
Preserves stages 0..4 in experiment_results.json and adds stage_5 (final model.safetensors).
"""
import json
import math
import struct
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from safetensors import safe_open

from inference_engine import MmapInt8LlamaModel
import synthesize_weights as sw

OUT_DIR = Path("/home/user/ai_experiment")


def build_stage5(E: np.ndarray, groups: Dict[str, List[int]]):
    t_start = time.time()
    V, D, DFF, L, KV_D, HD = sw.V, sw.D, sw.DFF, sw.L, sw.KV_D, sw.HD

    tensors_spec: List[Tuple[str, Tuple[int, ...], str]] = []
    tensors_spec.append(("model.embed_tokens.weight", (V, D), "I8"))
    tensors_spec.append(("model.embed_tokens.weight_scale", (V, 1), "F16"))
    for li in range(L):
        lp = f"model.layers.{li}"
        tensors_spec.append((f"{lp}.input_layernorm.weight", (D,), "F16"))
        for name, shape in [
            ("self_attn.q_proj", (D, D)),
            ("self_attn.k_proj", (KV_D, D)),
            ("self_attn.v_proj", (KV_D, D)),
            ("self_attn.o_proj", (D, D)),
        ]:
            tensors_spec.append((f"{lp}.{name}.weight", shape, "I8"))
            tensors_spec.append((f"{lp}.{name}.weight_scale", (shape[0], 1), "F16"))
        tensors_spec.append((f"{lp}.post_attention_layernorm.weight", (D,), "F16"))
        for name, shape in [
            ("mlp.gate_proj", (DFF, D)),
            ("mlp.up_proj", (DFF, D)),
            ("mlp.down_proj", (D, DFF)),
        ]:
            tensors_spec.append((f"{lp}.{name}.weight", shape, "I8"))
            tensors_spec.append((f"{lp}.{name}.weight_scale", (shape[0], 1), "F16"))
    tensors_spec.append(("model.norm.weight", (D,), "F16"))
    tensors_spec.append(("lm_head.weight", (V, D), "I8"))
    tensors_spec.append(("lm_head.weight_scale", (V, 1), "F16"))

    header = {"__metadata__": {
        "format": "pt",
        "experiment": "AI_experiment_AI_800M_pure_analytical_int8_stage5",
        "training_steps": "0",
        "total_weight_parameters": "800260608",
    }}
    offset = 0
    for name, shape, dtype in tensors_spec:
        n_elems = 1
        for s in shape:
            n_elems *= s
        n_bytes = n_elems * (1 if dtype == "I8" else 2)
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + n_bytes]}
        offset += n_bytes

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - (len(header_bytes) % 8)) % 8
    header_bytes += b" " * pad
    sf_path = OUT_DIR / "model.safetensors"
    sf_file = open(sf_path, "wb")
    sf_file.write(struct.pack("<Q", len(header_bytes)))
    sf_file.write(header_bytes)

    emit_idx = 0
    def emit_tensor(arr: np.ndarray, dtype: str):
        nonlocal emit_idx
        exp_name, exp_shape, exp_dtype = tensors_spec[emit_idx]
        emit_idx += 1
        assert tuple(arr.shape) == tuple(exp_shape)
        assert dtype == exp_dtype
        if dtype == "F16":
            assert not np.isnan(arr).any() and not np.isinf(arr).any(), f"NaN in {exp_name}"
        sf_file.write(np.ascontiguousarray(arr).tobytes())

    q_emb, s_emb = sw.quantize_int8_per_row(E)
    emit_tensor(q_emb, "I8")
    emit_tensor(s_emb, "F16")
    del q_emb, s_emb

    inv_freq = 1.0 / (10000.0 ** (np.arange(0, HD, 2, dtype=np.float32) / HD))
    cap_std_attn = 0.012 / math.sqrt(D)
    cap_std_mlp = 0.010 / math.sqrt(DFF)

    for li in range(L):
        emit_tensor(np.ones((D,), dtype=np.float16), "F16")
        q_q, s_q = sw.make_fast_int8_capacity(D, D, 1000 + li * 10 + 1, cap_std_attn)
        q_k, s_k = sw.make_fast_int8_capacity(KV_D, D, 1000 + li * 10 + 2, cap_std_attn)
        q_v, s_v = sw.make_fast_int8_capacity(KV_D, D, 1000 + li * 10 + 3, cap_std_attn)
        q_o, s_o = sw.make_fast_int8_capacity(D, D, 1000 + li * 10 + 4, cap_std_attn)

        if li == 0:
            # RoPE Previous-Token Head
            wq_blk = np.zeros((HD, D), dtype=np.float32)
            wk_blk = np.zeros((HD, D), dtype=np.float32)
            wv_blk = np.zeros((HD, D), dtype=np.float32)
            for j in range(24):
                w = inv_freq[j]
                c_amp = 3.2
                wk_blk[2 * j, sw.F_BIAS] = c_amp
                wq_blk[2 * j, sw.F_BIAS] = c_amp * math.cos(w)
                wq_blk[2 * j + 1, sw.F_BIAS] = -c_amp * math.sin(w)
            wv_blk[0:96, 0:96] = np.eye(96, dtype=np.float32) * 1.0
            wv_blk[96:128, 512:544] = np.eye(32, dtype=np.float32) * 1.0
            q_q[0:HD], s_q[0:HD] = sw.quantize_int8_per_row(wq_blk)
            q_k[0:HD], s_k[0:HD] = sw.quantize_int8_per_row(wk_blk)
            q_v[0:HD], s_v[0:HD] = sw.quantize_int8_per_row(wv_blk)
            wo_blk = np.zeros((HD, D), dtype=np.float32)
            wo_blk[:, 0:HD] = np.eye(HD, dtype=np.float32) * 1.0
            q_o[1280:1408], s_o[1280:1408] = sw.quantize_int8_per_row(wo_blk)
            q_o[:1280, 0:HD] = 0
            q_o[1408:, 0:HD] = 0

            # Prompt Intent Aggregator (Head 3 / KV-Head 1)
            q_off, kv_off = 3 * HD, 1 * HD
            wq_blk = np.zeros((HD, D), dtype=np.float32)
            wk_blk = np.zeros((HD, D), dtype=np.float32)
            wv_blk = np.zeros((HD, D), dtype=np.float32)
            for j in range(48, 64):
                wq_blk[2 * j, sw.F_BIAS] = 1.5
                wk_blk[2 * j, sw.F_BIAS] = 1.5
            for fk in (sw.F_PY_KW, sw.F_DEF_CLASS, sw.F_ANALYSIS, sw.F_GREETING):
                wk_blk[120, fk] = 2.0
            wq_blk[120, sw.F_BIAS] = 1.5
            wv_blk[0, sw.F_PY_KW] = 0.6
            wv_blk[0, sw.F_DEF_CLASS] = 0.8
            wv_blk[0, sw.F_PY_OP] = 0.4
            wv_blk[1, sw.F_ANALYSIS] = 0.8
            wv_blk[2, sw.F_COMPLEXITY] = 0.9
            wv_blk[3, sw.F_GREETING] = 0.8
            wv_blk[4, sw.F_QUESTION] = 0.7
            q_q[q_off : q_off + HD], s_q[q_off : q_off + HD] = sw.quantize_int8_per_row(wq_blk)
            q_k[kv_off : kv_off + HD], s_k[kv_off : kv_off + HD] = sw.quantize_int8_per_row(wk_blk)
            q_v[kv_off : kv_off + HD], s_v[kv_off : kv_off + HD] = sw.quantize_int8_per_row(wv_blk)

            q_o[:, q_off : q_off + HD] = 0
            edits_o = {}
            for dst_idx, src_ch in [
                (sw.PY_MODE, 0), (sw.LG_MODE, 1), (sw.LG_COMPLEXITY, 2), (sw.CV_MODE, 3), (sw.CV_QUESTION, 4)
            ]:
                row_f = q_o[dst_idx].astype(np.float32) * float(s_o[dst_idx, 0])
                row_f[q_off : q_off + HD] = 0.0
                row_f[q_off + src_ch] = 0.75
                edits_o[dst_idx] = row_f
            q_o, s_o = sw.apply_row_edits_int8(q_o, s_o, edits_o)

        if li == 2:
            # Induction Head ([A][B] ... [A] -> [B]) with strong match selectivity
            wq_blk = np.zeros((HD, D), dtype=np.float32)
            wk_blk = np.zeros((HD, D), dtype=np.float32)
            wv_blk = np.zeros((HD, D), dtype=np.float32)
            wq_blk[80:128, 0:48] = np.eye(48, dtype=np.float32) * 4.5
            wk_blk[80:128, 1280:1328] = np.eye(48, dtype=np.float32) * 4.5
            wv_blk[0:HD, 0:HD] = np.eye(HD, dtype=np.float32) * 0.06  # reducido de 0.12 para evitar repetición de palabras
            q_q[0:HD], s_q[0:HD] = sw.quantize_int8_per_row(wq_blk)
            q_k[0:HD], s_k[0:HD] = sw.quantize_int8_per_row(wk_blk)
            q_v[0:HD], s_v[0:HD] = sw.quantize_int8_per_row(wv_blk)
            wo_blk = np.zeros((HD, D), dtype=np.float32)
            wo_blk[:, 0:HD] = np.eye(HD, dtype=np.float32) * 0.06  # reducido de 0.12 para evitar repetición de palabras
            q_o[1408:1536], s_o[1408:1536] = sw.quantize_int8_per_row(wo_blk)
            q_o[:1408, 0:HD] = 0

        if li == 4:
            # Topic Keyword Echo into S_logic [903:1031]
            q_off, kv_off = 6 * HD, 2 * HD
            wq_blk = np.zeros((HD, D), dtype=np.float32)
            wk_blk = np.zeros((HD, D), dtype=np.float32)
            wv_blk = np.zeros((HD, D), dtype=np.float32)
            for j in range(52, 64):
                wq_blk[2 * j, sw.F_BIAS] = 1.0
                wk_blk[2 * j, sw.F_BIAS] = 1.0
            wk_blk[124, sw.F_COMPLEXITY] = 3.0
            wk_blk[124, sw.F_IDENT] = 2.0
            wq_blk[124, sw.F_BIAS] = 1.5
            wv_blk[:, 0:HD] = np.eye(HD, dtype=np.float32) * 0.28
            q_q[q_off : q_off + HD], s_q[q_off : q_off + HD] = sw.quantize_int8_per_row(wq_blk)
            q_k[kv_off : kv_off + HD], s_k[kv_off : kv_off + HD] = sw.quantize_int8_per_row(wk_blk)
            q_v[kv_off : kv_off + HD], s_v[kv_off : kv_off + HD] = sw.quantize_int8_per_row(wv_blk)
            wo_blk = np.zeros((HD, D), dtype=np.float32)
            wo_blk[:, q_off : q_off + HD] = np.eye(HD, dtype=np.float32) * 0.28
            q_o[903 : 903 + 128], s_o[903 : 903 + 128] = sw.quantize_int8_per_row(wo_blk)
            q_o[:903, q_off : q_off + HD] = 0
            q_o[903 + 128 :, q_off : q_off + HD] = 0

        for q_m, s_m in ((q_q, s_q), (q_k, s_k), (q_v, s_v), (q_o, s_o)):
            emit_tensor(q_m, "I8")
            emit_tensor(s_m, "F16")
        del q_q, s_q, q_k, s_k, q_v, s_v, q_o, s_o

        emit_tensor(np.ones((D,), dtype=np.float16), "F16")

        q_g, s_g = sw.make_fast_int8_capacity(DFF, D, 2000 + li * 10 + 1, cap_std_mlp)
        q_u, s_u = sw.make_fast_int8_capacity(DFF, D, 2000 + li * 10 + 2, cap_std_mlp)
        q_d, s_d = sw.make_fast_int8_capacity(D, DFF, 2000 + li * 10 + 3, cap_std_mlp)

        if li in (1, 6, 12, 16):
            wg_blk = np.zeros((32, D), dtype=np.float32)
            wu_blk = np.zeros((32, D), dtype=np.float32)
            P_DEF = 1376 + (sw.F_DEF_CLASS - 512)
            P_OPEN = 1376 + (sw.F_OPEN_PAREN - 512)
            P_COLON = 1376 + (sw.F_COLON - 512)
            P_INDENT = 1376 + (sw.F_INDENT - 512)
            P_CTRL = 1376 + (sw.F_CTRL_FLOW - 512)
            P_RET = 1376 + (sw.F_RETURN - 512)

            wg_blk[0, sw.F_DEF_CLASS] = 1.5; wg_blk[0, P_DEF] = 1.5; wu_blk[0, sw.F_BIAS] = 1.2
            wg_blk[1, sw.F_OPEN_PAREN] = 1.5; wg_blk[1, P_OPEN] = 1.2; wu_blk[1, sw.PY_MODE] = 1.2
            wg_blk[2, sw.F_COLON] = 1.5; wg_blk[2, P_COLON] = 1.2; wu_blk[2, sw.PY_MODE] = 1.2
            wg_blk[3, sw.F_INDENT] = 1.5; wg_blk[3, P_INDENT] = 1.2; wu_blk[3, sw.PY_MODE] = 1.2
            wg_blk[4, sw.F_CTRL_FLOW] = 1.5; wg_blk[4, P_CTRL] = 1.2; wu_blk[4, sw.PY_MODE] = 1.2
            wg_blk[5, sw.F_RETURN] = 1.5; wg_blk[5, P_RET] = 1.2; wu_blk[5, sw.F_BIAS] = 1.2
            wg_blk[6, sw.LG_MODE] = 1.5; wu_blk[6, sw.F_QUESTION] = 1.2; wu_blk[6, sw.F_COLON] = 1.2
            wg_blk[7, sw.LG_MODE] = 1.5; wu_blk[7, sw.F_STEP_WORD] = 1.2; wu_blk[7, sw.F_COMPLEXITY] = 1.0
            wg_blk[8, sw.LG_MODE] = 1.5; wu_blk[8, sw.F_LOGIC_CONN] = 1.2
            wg_blk[9, sw.CV_MODE] = 1.5; wu_blk[9, sw.F_GREETING] = 1.2
            wg_blk[10, sw.CV_QUESTION] = 1.5; wu_blk[10, sw.F_QUESTION] = 1.2

            q_g[0:32], s_g[0:32] = sw.quantize_int8_per_row(wg_blk)
            q_u[0:32], s_u[0:32] = sw.quantize_int8_per_row(wu_blk)
            q_d[:, 0:32] = 0
            mlp_g = 0.055
            edits_d = {}
            for dst_idx, src_neuron in [
                (sw.PY_EXP_IDENT, 0), (sw.PY_EXP_PAREN, 1), (sw.PY_EXP_INDENT, 2),
                (sw.PY_EXP_RETURN, 3), (sw.PY_EXP_LOOP_IN, 4), (sw.PY_EXP_EXPR, 5),
                (sw.LG_STEP1, 6), (sw.LG_STEP2, 7), (sw.LG_STEP3, 8), (sw.CV_GREETING, 9),
            ]:
                row_f = q_d[dst_idx].astype(np.float32) * float(s_d[dst_idx, 0])
                row_f[0:32] = 0.0
                row_f[src_neuron] = mlp_g
                if dst_idx == sw.CV_GREETING:
                    row_f[10] = mlp_g * 0.8
                edits_d[dst_idx] = row_f
            q_d, s_d = sw.apply_row_edits_int8(q_d, s_d, edits_d)

        for q_m, s_m in ((q_g, s_g), (q_u, s_u), (q_d, s_d)):
            emit_tensor(q_m, "I8")
            emit_tensor(s_m, "F16")
        del q_g, s_g, q_u, s_u, q_d, s_d

    emit_tensor(np.ones((D,), dtype=np.float16), "F16")

    # Stage 5 lm_head: NO negative antipodal -E[:, 0:128] and NO random 256:512 echo!
    lm_head = np.zeros((V, D), dtype=np.float32)
    lm_head[:, 128:256] = E[:, 128:256] * 0.025  # Gentle non-negative character similarity only
    lm_head[:, 1408:1536] = E[:, 0:128] * 0.42   # Clean Induction readout
    lm_head[:, 903:1031] = E[:, 0:128] * 0.20    # Topic keyword echo

    sg = 0.35  # Balanced logit scale for diversity while keeping Python/analysis signal
    for tid in groups["py_ident_good"]:
        lm_head[tid, sw.PY_EXP_IDENT] += 3.5 * sg
        lm_head[tid, sw.PY_MODE] += 0.8 * sg
    for tid in groups["open_paren"]:
        lm_head[tid, sw.F_IDENT] += 1.6 * sg
        lm_head[tid, sw.PY_MODE] += 0.9 * sg
    for tid in groups["py_param"]:
        lm_head[tid, sw.PY_EXP_PAREN] += 3.0 * sg
    for tid in groups["close_colon"]:
        lm_head[tid, sw.PY_EXP_PAREN] += 2.0 * sg
        lm_head[tid, sw.F_IDENT] += 1.2 * sg
        lm_head[tid, sw.PY_MODE] += 0.8 * sg
    for tid in groups["newline_indent"]:
        lm_head[tid, sw.PY_EXP_INDENT] += 3.8 * sg
    for tid in groups["return_kw"]:
        lm_head[tid, sw.PY_EXP_RETURN] += 3.6 * sg
    for tid in groups["py_builtins"]:
        lm_head[tid, sw.PY_EXP_EXPR] += 2.6 * sg
        lm_head[tid, sw.PY_EXP_RETURN] += 2.0 * sg
    for tid in groups["in_kw"]:
        lm_head[tid, sw.PY_EXP_LOOP_IN] += 3.2 * sg

    for tid in groups["analysis_step1"]:
        lm_head[tid, sw.LG_STEP1] += 3.4 * sg
        lm_head[tid, sw.LG_MODE] += 1.2 * sg
    for tid in groups["analysis_step2"]:
        lm_head[tid, sw.LG_STEP2] += 3.4 * sg
        lm_head[tid, sw.LG_COMPLEXITY] += 1.6 * sg
    for tid in groups["complexity_terms"]:
        lm_head[tid, sw.LG_COMPLEXITY] += 2.2 * sg
        lm_head[tid, sw.LG_MODE] += 0.9 * sg
    for tid in groups["stats_terms"]:
        lm_head[tid, sw.LG_MODE] += 1.8 * sg
    for tid in groups["comparison_terms"]:
        lm_head[tid, sw.LG_MODE] += 1.5 * sg
    for tid in groups["analysis_step3"]:
        lm_head[tid, sw.LG_STEP3] += 3.2 * sg

    for tid in groups["conv_greet_reply"]:
        lm_head[tid, sw.CV_GREETING] += 3.2 * sg
    for tid in groups["conv_help_words"]:
        lm_head[tid, sw.CV_MODE] += 1.6 * sg
        lm_head[tid, sw.CV_QUESTION] += 1.4 * sg
    for tid in groups["punct_end"]:
        lm_head[tid, sw.CV_MODE] += 0.4 * sg
        lm_head[tid, sw.LG_MODE] += 0.3 * sg

    # Suppress bare space 'Ġ' (42) and common echo/loop tokens
    lm_head[42, sw.F_BIAS] = -6.0     # bare space - prevent space loops after 'def '
    lm_head[1, sw.F_BIAS] = -4.0      # <|eos|> - prevent premature stop
    lm_head[73, sw.F_BIAS] = -4.0     # '?' - prevent question-loop

    q_lm, s_lm = sw.quantize_int8_per_row(lm_head)
    emit_tensor(q_lm, "I8")
    emit_tensor(s_lm, "F16")
    sf_file.close()
    print(f"[Stage 5] Wrote final model.safetensors in {time.time()-t_start:.2f}s", flush=True)


def main():
    E, groups = sw.build_token_feature_table()
    build_stage5(E, groups)

    results = json.loads((OUT_DIR / "experiment_results.json").read_text())
    model = MmapInt8LlamaModel(OUT_DIR, "model.safetensors")

    prompts = [
        ("induction_copy", "alpha = 42 ; beta = 99 ; alpha ="),
        ("python_def", "def "),
        ("analysis_complexity", "Analyze the time complexity of binary search:"),
        ("conversation_hello", "Hello! How can you help me?"),
        ("python_loop", "for i in range(10):\n    "),
        ("analysis_data", "Compare variance and outliers in numerical data analysis:"),
    ]

    st5_greedy = {}
    st5_sampled = {}
    for pname, prompt in prompts:
        res_g = model.generate(prompt, max_new_tokens=10, temperature=0.0, repetition_penalty=1.35)
        st5_greedy[pname] = res_g
        print(f"  [Stage 5 Greedy+RP | {pname}] -> {repr(res_g['generated_text'])} (H={res_g['first_token_entropy_nats']:.2f}, {res_g['elapsed_sec']}s)", flush=True)
        res_s = model.generate(prompt, max_new_tokens=12, temperature=0.6, top_k=15, repetition_penalty=1.35)
        st5_sampled[pname] = res_s
        print(f"  [Stage 5 Sampled | {pname}] -> {repr(res_s['generated_text'])} ({res_s['elapsed_sec']}s)", flush=True)

    results["stages"]["stage_5"] = st5_greedy
    results["stage_5_sampled"] = st5_sampled

    # Re-verify safetensors
    sf_path = OUT_DIR / "model.safetensors"
    total_core, total_scale, nan_cnt = 0, 0, 0
    i8_min, i8_max = 127, -128
    with safe_open(str(sf_path), framework="numpy") as f:
        keys = list(f.keys())
        for k in keys:
            t = f.get_tensor(k)
            if k.endswith(".weight_scale"):
                total_scale += t.size
            else:
                total_core += t.size
            if np.issubdtype(t.dtype, np.floating) and (np.isnan(t).any() or np.isinf(t).any()):
                nan_cnt += 1
            elif t.dtype == np.int8:
                i8_min = min(i8_min, int(t.min()))
                i8_max = max(i8_max, int(t.max()))

    results["verification"] = {
        "file": str(sf_path),
        "size_bytes": sf_path.stat().st_size,
        "size_mb": round(sf_path.stat().st_size / (1024 * 1024), 2),
        "tensor_count": len(keys),
        "core_weight_parameters": total_core,
        "quantization_scale_scalars": total_scale,
        "nan_or_inf_tensors": nan_cnt,
        "int8_dynamic_range_used": [i8_min, i8_max],
    }
    (OUT_DIR / "experiment_results.json").write_text(json.dumps(results, indent=2))
    print("DONE:", json.dumps(results["verification"], indent=2), flush=True)


if __name__ == "__main__":
    main()
