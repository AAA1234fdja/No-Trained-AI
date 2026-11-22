#!/usr/bin/env python3
"""
AI Experiment: Pure Analytical Weight Synthesis & Direct Weight Editing (800,260,608 params, INT8 .safetensors)
Zero training data. Zero gradient descent. Every tensor is constructed mathematically and edited directly.

Subspace layout (d = 1536, 12 heads x 128 dims):
  [0:512]     S_lex   : Lexical & semantic token identity ([0:128] primary compact signature)
  [512:640]   S_cat   : Explicit syntactic/domain category flags [512:544] + semantic clusters [544:640]
  [640:896]   S_py    : Python AST & syntactic expectation state
  [896:1152]  S_logic : Analytical & reasoning state + prompt topic echo
  [1152:1280] S_conv  : Minimal conversational intent & register
  [1280:1408] S_mem_p : Previous-token lexical buffer (written by Layer 0 RoPE prev-token head)
  [1408:1536] S_mem_i : Induction target lexical buffer (written by Layers 2-4 induction heads)
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

OUT_DIR = Path("/home/user/ai_experiment")
CFG = json.loads((OUT_DIR / "config.json").read_text())
META = json.loads((OUT_DIR / "token_metadata.json").read_text())

V = CFG["vocab_size"]          # 32000
D = CFG["hidden_size"]         # 1536
DFF = CFG["intermediate_size"] # 5888
L = CFG["num_hidden_layers"]   # 21
HQ = CFG["num_attention_heads"]# 12
HKV = CFG["num_key_value_heads"]# 4
HD = CFG["head_dim"]           # 128
KV_D = HKV * HD                # 512

# Category flag indices inside S_cat (offset 512)
F_BIAS = 512
F_PY_KW = 513
F_DEF_CLASS = 514
F_RETURN = 515
F_CTRL_FLOW = 516
F_PY_BUILTIN = 517
F_PY_OP = 518
F_COLON = 519
F_INDENT = 520
F_OPEN_PAREN = 521
F_CLOSE_PAREN = 522
F_ANALYSIS = 523
F_LOGIC_CONN = 524
F_COMPLEXITY = 525
F_GREETING = 526
F_QUESTION = 527
F_FUNC_WORD = 528
F_IDENT = 529
F_NUMBER = 530
F_PUNCT_END = 531
F_ASSIGN = 532
F_STEP_WORD = 533

# State indices
PY_MODE = 640
PY_EXP_IDENT = 641
PY_EXP_PAREN = 642
PY_EXP_COLON = 643
PY_EXP_INDENT = 644
PY_EXP_RETURN = 645
PY_EXP_LOOP_IN = 646
PY_EXP_EXPR = 647

LG_MODE = 896
LG_COMPLEXITY = 897
LG_COMPARE = 898
LG_STATS = 899
LG_STEP1 = 900
LG_STEP2 = 901
LG_STEP3 = 902

CV_MODE = 1152
CV_GREETING = 1153
CV_QUESTION = 1154


def decode_bl(tok: str) -> str:
    # Quick approximate readable form for rule matching
    return tok.replace("Ġ", " ").replace("Ċ", "\n")


def quantize_int8_per_row(w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Symmetric per-output-channel INT8 quantization."""
    max_abs = np.max(np.abs(w), axis=1, keepdims=True)
    scale = np.where(max_abs > 1e-8, max_abs / 127.0, np.float32(1.0 / 127.0)).astype(np.float16)
    scale_f32 = scale.astype(np.float32)
    q = np.rint(np.clip(w / scale_f32, -127.0, 127.0)).astype(np.int8)
    return q, scale


def build_token_feature_table() -> Tuple[np.ndarray, Dict[str, List[int]]]:
    """
    Constructs analytical lexical + categorical embeddings (32000 x D) with zero training.
    Also returns token index lists for direct weight editing rules.
    """
    rng = np.random.default_rng(20261122)
    E = np.zeros((V, D), dtype=np.float32)

    # 1. Orthogonalized lexical basis in [0:512]
    # Primary compact 128-d lexical signature [0:128] (normalized to unit norm * 16.0)
    lex_primary = rng.standard_normal((V, 128)).astype(np.float32)
    lex_primary /= (np.linalg.norm(lex_primary, axis=1, keepdims=True) + 1e-8)

    # Secondary lexical signature [256:512]
    lex_sec = rng.standard_normal((V, 256)).astype(np.float32)
    lex_sec /= (np.linalg.norm(lex_sec, axis=1, keepdims=True) + 1e-8)

    # Inject analytical morphological / character-bigram similarity into [128:256]
    char_proj = np.zeros((V, 128), dtype=np.float32)
    id_to_tok = META["id_to_token"]
    cats = META["categories"]

    groups: Dict[str, List[int]] = {
        "def_kw": [], "return_kw": [], "for_kw": [], "if_kw": [], "in_kw": [],
        "py_ident_good": [], "py_param": [], "open_paren": [], "close_colon": [],
        "newline_indent": [], "py_builtins": [], "py_ops": [],
        "analysis_step1": [], "analysis_step2": [], "analysis_step3": [],
        "complexity_terms": [], "comparison_terms": [], "stats_terms": [],
        "conv_greet_reply": [], "conv_help_words": [], "punct_end": [],
    }

    py_kws = {"def", "class", "return", "if", "elif", "else", "for", "while", "import", "from",
              "try", "except", "finally", "with", "yield", "lambda", "async", "await", "pass",
              "break", "continue", "raise", "assert", "in", "is", "not", "and", "or", "True", "False", "None"}
    def_kws = {"def", "class", "async def"}
    ret_kws = {"return", "yield"}
    ctrl_kws = {"if", "elif", "else", "for", "while", "try", "except", "finally", "with"}
    analysis_set = {"analysis", "analyze", "analytical", "evaluate", "compare", "comparison",
                    "complexity", "algorithm", "hypothesis", "evidence", "conclusion", "reason",
                    "reasoning", "logic", "logical", "variance", "mean", "median", "outlier",
                    "step", "steps", "because", "therefore", "thus", "hence", "however",
                    "first", "second", "third", "finally", "explain", "problem", "solution",
                    "time", "space", "memory", "linear", "logarithmic", "quadratic", "binary",
                    "search", "sort", "data", "pattern", "trend", "correlation", "factor"}
    logic_conn_set = {"first", "second", "third", "finally", "because", "therefore", "thus",
                      "hence", "however", "moreover", "furthermore", "consequently", "specifically",
                      "if", "then", "implies", "and", "or", "but"}
    complexity_set = {"complexity", "time", "space", "memory", "linear", "logarithmic",
                      "quadratic", "exponential", "constant", "asymptotic", "algorithm",
                      "binary", "search", "sort", "runtime", "latency", "fast", "slow", "optimal"}
    stats_set = {"mean", "median", "variance", "std", "average", "sum", "count", "ratio",
                 "rate", "distribution", "outlier", "outliers", "correlation", "probability", "data"}
    greet_set = {"hello", "hi", "hey", "greetings", "welcome", "thanks", "thank", "please",
                 "good", "morning", "fine", "great", "ready", "help", "assistant"}
    wh_set = {"what", "why", "how", "which", "who", "where", "when", "can", "could", "would",
              "explain", "describe", "analyze", "compare"}
    func_set = {"i", "you", "we", "it", "this", "that", "the", "a", "an", "is", "are", "was",
                "am", "can", "will", "to", "of", "in", "for", "with", "on", "at", "by", "as"}

    for tid_str, bl_tok in id_to_tok.items():
        tid = int(tid_str)
        raw = decode_bl(bl_tok)
        clean = raw.strip().lower()
        cat = cats.get(tid_str, "subword")

        # Character n-gram projection
        for i in range(len(raw)):
            h = (ord(raw[i]) * 31 + i * 7) % 128
            char_proj[tid, h] += 1.0
            if i + 1 < len(raw):
                h2 = (ord(raw[i]) * 131 + ord(raw[i + 1]) * 17) % 128
                char_proj[tid, h2] += 0.7
        nrm = np.linalg.norm(char_proj[tid])
        if nrm > 0:
            char_proj[tid] /= nrm

        # Explicit category coordinates
        E[tid, F_BIAS] = 2.0  # Constant token presence signal

        if cat == "python_kw" or clean in py_kws:
            E[tid, F_PY_KW] = 3.0
        if clean in def_kws or raw in ("def ", "class ", "async def "):
            E[tid, F_DEF_CLASS] = 4.0
            groups["def_kw"].append(tid)
        if clean in ret_kws or raw in ("return ", "yield "):
            E[tid, F_RETURN] = 4.0
            groups["return_kw"].append(tid)
        if clean in ctrl_kws or raw in ("for ", "while ", "if ", "elif ", "else:", "try:"):
            E[tid, F_CTRL_FLOW] = 3.5
            if clean == "for" or raw == "for ":
                groups["for_kw"].append(tid)
            if clean == "if" or raw == "if ":
                groups["if_kw"].append(tid)
        if raw in (" in", "in"):
            groups["in_kw"].append(tid)
        if cat in ("python_builtin", "python_lib"):
            E[tid, F_PY_BUILTIN] = 3.0
            if raw.startswith(" ") and clean in {"sum", "len", "range", "sorted", "enumerate", "zip", "print", "min", "max", "abs"}:
                groups["py_builtins"].append(tid)
        if cat == "python_syntax" or raw.strip() in {"(", ")", "():", "):", ":", "=", "==", "+=", "+", "-", "*", "/"}:
            E[tid, F_PY_OP] = 3.0
            if raw in ("(", " ("):
                E[tid, F_OPEN_PAREN] = 4.0
                groups["open_paren"].append(tid)
            if raw in (")", "):", "():", "]:", "}:"):
                E[tid, F_CLOSE_PAREN] = 4.0
            if ":" in raw:
                E[tid, F_COLON] = 4.0
                if raw in ("):", "():", ":\n"):
                    groups["close_colon"].append(tid)
            if "=" in raw and "==" not in raw:
                E[tid, F_ASSIGN] = 3.5
        if "\n" in raw or raw in ("    ", "        ", "\n    ", "\n        "):
            E[tid, F_INDENT] = 4.0
            if raw in ("\n    ", "    "):
                groups["newline_indent"].append(tid)

        if cat == "analysis" or clean in analysis_set:
            E[tid, F_ANALYSIS] = 3.5
        if clean in logic_conn_set:
            E[tid, F_LOGIC_CONN] = 3.5
        if clean in complexity_set:
            E[tid, F_COMPLEXITY] = 3.5
            if raw.startswith(" ") and len(clean) >= 4:
                groups["complexity_terms"].append(tid)
        if clean in stats_set:
            if raw.startswith(" ") and len(clean) >= 4:
                groups["stats_terms"].append(tid)
        if clean in {"compare", "comparison", "versus", "however", "whereas", "better", "faster", "tradeoff"}:
            if raw.startswith(" "):
                groups["comparison_terms"].append(tid)
        if clean in greet_set:
            E[tid, F_GREETING] = 3.5
        if "?" in raw or clean in wh_set:
            E[tid, F_QUESTION] = 3.5
        if clean in func_set:
            E[tid, F_FUNC_WORD] = 2.5
        if cat in ("python_id",) or (raw.strip().isidentifier() and clean not in py_kws):
            E[tid, F_IDENT] = 2.0
            if raw in (" n", " x", " arr", " data", " items", " self", "n", "x", "arr", "data"):
                groups["py_param"].append(tid)
            if raw in ("compute", "calculate", "solve", "process", "analyze", " compute", " calculate", " solve", " process", " analyze"):
                groups["py_ident_good"].append(tid)
        if cat == "number" or raw.strip().isdigit():
            E[tid, F_NUMBER] = 3.0
        if raw in (".", ". ", ".\n", "!", "! ", "?", "? "):
            E[tid, F_PUNCT_END] = 3.0
            groups["punct_end"].append(tid)

        # Specific target words for analytical steps & conversation
        if raw in ("First", " First", "Step", "Analysis"):
            E[tid, F_STEP_WORD] = 3.0
            groups["analysis_step1"].append(tid)
        if raw in (" because", " therefore", " thus", " since", " implies"):
            groups["analysis_step2"].append(tid)
        if raw in (" conclusion", " Therefore", " summary", " optimal", " efficient"):
            groups["analysis_step3"].append(tid)
        if raw in ("Hello", " hello", " ready", " I", " can", " help", " analyze", " Python"):
            groups["conv_greet_reply"].append(tid)
        if raw in (" I", " can", " help", " explain", " analyze", " write", " Python", " code", " data"):
            groups["conv_help_words"].append(tid)

    # Assemble [0:512] with controlled rms ~ 1.0 overall
    E[:, 0:128] = lex_primary * 18.0
    E[:, 128:256] = char_proj * 12.0
    E[:, 256:512] = lex_sec * 16.0

    # Normalize each embedding row slightly so RMS ~ 1.0 (norm ~ sqrt(1536) = 39.19)
    # Wait: keep explicit S_cat [512:640] unattenuated by scaling only S_lex!
    # Check RMS: mean(E[:, 0:512]^2) ~ (18^2/128 + 12^2/128 + 16^2/256) * (512/1536)...
    # Let's explicitly verify RMS(E) is around 1.0 - 1.2 so RMSNorm(x) ~ x.
    rms_per_row = np.sqrt(np.mean(E * E, axis=1, keepdims=True) + 1e-8)
    E = E / rms_per_row  # Now EVERY token embedding has exact RMS = 1.0!
    return E, groups


def make_fast_int8_capacity(
    out_dim: int, in_dim: int, seed: int, std: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast deterministic INT8 capacity block (uniform [-127, 127], std_q ~= 73.3).
    Returns (q_int8, scale_f16) directly in ~25ms per 9M-param matrix.
    """
    rng = np.random.default_rng(seed)
    q = rng.integers(-127, 128, size=(out_dim, in_dim), dtype=np.int8)
    s_val = np.float16(std / 73.32)
    scale = np.full((out_dim, 1), s_val, dtype=np.float16)
    return q, scale


def apply_row_edits_int8(
    q: np.ndarray, scale: np.ndarray, edited_rows: Dict[int, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """Overwrites specific rows of an INT8 matrix + scale with analytically engineered weights."""
    if not edited_rows:
        return q, scale
    idxs = sorted(edited_rows.keys())
    mat = np.stack([edited_rows[i] for i in idxs], axis=0).astype(np.float32)
    q_sub, s_sub = quantize_int8_per_row(mat)
    for pos, r_idx in enumerate(idxs):
        q[r_idx, :] = q_sub[pos, :]
        scale[r_idx, 0] = s_sub[pos, 0]
    return q, scale


def synthesize_stage(stage: int, E: np.ndarray, groups: Dict[str, List[int]], save_safetensors: bool = False):
    """
    Stages:
      0 = Baseline orthogonal/Rademacher INT8 (no analytical edits)
      1 = Stage 0 + Analytical Subspace Embeddings + Lex/Cat lm_head edits
      2 = Stage 1 + RoPE Previous-Token, Induction, and Intent Routing Attention Heads
      3 = Stage 2 + SwiGLU Python/Analysis/Conversation Circuits + Anti-Repetition + Full lm_head Transition Edits
    """
    rng = np.random.default_rng(999 + stage)
    t_start = time.time()

    # Precompute tensor list for safetensors streaming
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

    # Build JSON header for streaming safetensors if saving
    sf_file = None
    expected_specs = list(tensors_spec)
    emit_idx = 0
    if save_safetensors:
        header = {"__metadata__": {
            "format": "pt",
            "experiment": "AI_experiment_AI_800M_pure_analytical_int8",
            "training_steps": "0",
            "total_weight_parameters": "800260608",
        }}
        offset = 0
        for name, shape, dtype in tensors_spec:
            n_elems = 1
            for s in shape:
                n_elems *= s
            n_bytes = n_elems * (1 if dtype == "I8" else 2)
            header[name] = {
                "dtype": dtype,
                "shape": list(shape),
                "data_offsets": [offset, offset + n_bytes],
            }
            offset += n_bytes
        header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
        # Pad header to 8-byte alignment
        pad = (8 - (len(header_bytes) % 8)) % 8
        header_bytes += b" " * pad
        sf_path = OUT_DIR / "model.safetensors"
        sf_file = open(sf_path, "wb")
        sf_file.write(struct.pack("<Q", len(header_bytes)))
        sf_file.write(header_bytes)

    def emit_tensor(arr: np.ndarray, dtype: str):
        nonlocal emit_idx
        exp_name, exp_shape, exp_dtype = expected_specs[emit_idx]
        emit_idx += 1
        assert tuple(arr.shape) == tuple(exp_shape), f"Shape mismatch for {exp_name}: {arr.shape} != {exp_shape}"
        assert dtype == exp_dtype, f"Dtype mismatch for {exp_name}: {dtype} != {exp_dtype}"
        if sf_file is not None:
            if dtype == "I8":
                assert arr.dtype == np.int8
            elif dtype == "F16":
                assert arr.dtype == np.float16
                assert not np.isnan(arr).any() and not np.isinf(arr).any(), f"NaN/Inf in {exp_name}"
            sf_file.write(np.ascontiguousarray(arr).tobytes())

    # 1. Embed tokens
    if stage == 0:
        E_use = rng.standard_normal((V, D)).astype(np.float32)
        E_use /= (np.sqrt(np.mean(E_use * E_use, axis=1, keepdims=True)) + 1e-8)
    else:
        E_use = E

    q_emb, s_emb = quantize_int8_per_row(E_use)
    emit_tensor(q_emb, "I8")
    emit_tensor(s_emb, "F16")
    del q_emb, s_emb

    # Precompute RoPE freq array for Layer 0 Prev-Token head
    inv_freq = 1.0 / (10000.0 ** (np.arange(0, HD, 2, dtype=np.float32) / HD))

    # 2. Layers 0..20
    cap_std_attn = 0.012 / math.sqrt(D)
    cap_std_mlp = 0.010 / math.sqrt(DFF)

    for li in range(L):
        # input_layernorm
        ln1 = np.ones((D,), dtype=np.float16)
        emit_tensor(ln1, "F16")

        # Initialize Attention matrices as fast INT8 capacity blocks
        q_q, s_q = make_fast_int8_capacity(D, D, 1000 + li * 10 + 1, cap_std_attn)
        q_k, s_k = make_fast_int8_capacity(KV_D, D, 1000 + li * 10 + 2, cap_std_attn)
        q_v, s_v = make_fast_int8_capacity(KV_D, D, 1000 + li * 10 + 3, cap_std_attn)
        q_o, s_o = make_fast_int8_capacity(D, D, 1000 + li * 10 + 4, cap_std_attn)

        if stage >= 2:
            # Gain calibration: stage 2/3 used high induction gain (1.5 -> hijack); stage 4 calibrates energy!
            ind_gain = 1.5 if stage in (2, 3) else 0.45
            # --- LAYER 0: RoPE Previous-Token Copy Head + Global Intent Aggregator ---
            if li == 0:
                wq_blk = np.zeros((HD, D), dtype=np.float32)
                wk_blk = np.zeros((HD, D), dtype=np.float32)
                wv_blk = np.zeros((HD, D), dtype=np.float32)
                for j in range(24):
                    w = inv_freq[j]
                    c_amp = 3.2
                    wk_blk[2 * j, F_BIAS] = c_amp
                    wq_blk[2 * j, F_BIAS] = c_amp * math.cos(w)
                    wq_blk[2 * j + 1, F_BIAS] = -c_amp * math.sin(w)
                # Copy S_lex[0:96] + S_cat[512:544] of previous token into H10 (1280:1408)
                wv_blk[0:96, 0:96] = np.eye(96, dtype=np.float32) * 1.0
                wv_blk[96:128, 512:544] = np.eye(32, dtype=np.float32) * 1.0
                q_q[0:HD], s_q[0:HD] = quantize_int8_per_row(wq_blk)
                q_k[0:HD], s_k[0:HD] = quantize_int8_per_row(wk_blk)
                q_v[0:HD], s_v[0:HD] = quantize_int8_per_row(wv_blk)

                wo_blk = np.zeros((HD, D), dtype=np.float32)
                wo_blk[:, 0:HD] = np.eye(HD, dtype=np.float32) * 1.0
                q_o[1280:1408], s_o[1280:1408] = quantize_int8_per_row(wo_blk)
                q_o[:1280, 0:HD] = 0
                q_o[1408:, 0:HD] = 0

                # Head 3 / KV-Head 1: Global Prompt Intent Aggregator
                q_off = 3 * HD
                kv_off = 1 * HD
                wq_blk = np.zeros((HD, D), dtype=np.float32)
                wk_blk = np.zeros((HD, D), dtype=np.float32)
                wv_blk = np.zeros((HD, D), dtype=np.float32)
                for j in range(48, 64):
                    wq_blk[2 * j, F_BIAS] = 1.5
                    wk_blk[2 * j, F_BIAS] = 1.5
                wk_blk[120, F_PY_KW] = 2.0
                wk_blk[120, F_DEF_CLASS] = 2.0
                wk_blk[120, F_ANALYSIS] = 2.0
                wk_blk[120, F_GREETING] = 2.0
                wq_blk[120, F_BIAS] = 1.5
                wv_blk[0, F_PY_KW] = 0.6
                wv_blk[0, F_DEF_CLASS] = 0.8
                wv_blk[0, F_PY_OP] = 0.4
                wv_blk[1, F_ANALYSIS] = 0.8
                wv_blk[2, F_COMPLEXITY] = 0.9
                wv_blk[3, F_GREETING] = 0.8
                wv_blk[4, F_QUESTION] = 0.7
                q_q[q_off : q_off + HD], s_q[q_off : q_off + HD] = quantize_int8_per_row(wq_blk)
                q_k[kv_off : kv_off + HD], s_k[kv_off : kv_off + HD] = quantize_int8_per_row(wk_blk)
                q_v[kv_off : kv_off + HD], s_v[kv_off : kv_off + HD] = quantize_int8_per_row(wv_blk)

                q_o[:, q_off : q_off + HD] = 0
                edits_o = {}
                g_intent = 1.5 if stage in (2, 3) else 0.75
                for dst_idx, src_ch in [
                    (PY_MODE, 0), (LG_MODE, 1), (LG_COMPLEXITY, 2), (CV_MODE, 3), (CV_QUESTION, 4)
                ]:
                    row_f = q_o[dst_idx].astype(np.float32) * float(s_o[dst_idx, 0])
                    row_f[q_off : q_off + HD] = 0.0
                    row_f[q_off + src_ch] = g_intent
                    edits_o[dst_idx] = row_f
                q_o, s_o = apply_row_edits_int8(q_o, s_o, edits_o)

            # --- LAYER 2: Exact Induction Head ([A][B] ... [A] -> [B]) ---
            if li == 2:
                wq_blk = np.zeros((HD, D), dtype=np.float32)
                wk_blk = np.zeros((HD, D), dtype=np.float32)
                wv_blk = np.zeros((HD, D), dtype=np.float32)
                wq_blk[80:128, 0:48] = np.eye(48, dtype=np.float32) * 3.5
                wk_blk[80:128, 1280:1328] = np.eye(48, dtype=np.float32) * 3.5
                # In stage 4, sharpen match only on strong lexical alignment & attenuate non-match dilution
                wv_blk[0:HD, 0:HD] = np.eye(HD, dtype=np.float32) * ind_gain
                q_q[0:HD], s_q[0:HD] = quantize_int8_per_row(wq_blk)
                q_k[0:HD], s_k[0:HD] = quantize_int8_per_row(wk_blk)
                q_v[0:HD], s_v[0:HD] = quantize_int8_per_row(wv_blk)

                wo_blk = np.zeros((HD, D), dtype=np.float32)
                wo_blk[:, 0:HD] = np.eye(HD, dtype=np.float32) * ind_gain
                q_o[1408:1536], s_o[1408:1536] = quantize_int8_per_row(wo_blk)
                q_o[:1408, 0:HD] = 0

            # --- LAYER 4: Topic Keyword Echo into S_logic [903:1031] ---
            if li == 4:
                q_off = 6 * HD
                kv_off = 2 * HD
                wq_blk = np.zeros((HD, D), dtype=np.float32)
                wk_blk = np.zeros((HD, D), dtype=np.float32)
                wv_blk = np.zeros((HD, D), dtype=np.float32)
                for j in range(52, 64):
                    wq_blk[2 * j, F_BIAS] = 1.0
                    wk_blk[2 * j, F_BIAS] = 1.0
                wk_blk[124, F_COMPLEXITY] = 3.0
                wk_blk[124, F_IDENT] = 2.0
                wq_blk[124, F_BIAS] = 1.5
                g_topic = 0.9 if stage in (2, 3) else 0.35
                wv_blk[:, 0:HD] = np.eye(HD, dtype=np.float32) * g_topic
                q_q[q_off : q_off + HD], s_q[q_off : q_off + HD] = quantize_int8_per_row(wq_blk)
                q_k[kv_off : kv_off + HD], s_k[kv_off : kv_off + HD] = quantize_int8_per_row(wk_blk)
                q_v[kv_off : kv_off + HD], s_v[kv_off : kv_off + HD] = quantize_int8_per_row(wv_blk)

                wo_blk = np.zeros((HD, D), dtype=np.float32)
                wo_blk[:, q_off : q_off + HD] = np.eye(HD, dtype=np.float32) * g_topic
                q_o[903 : 903 + 128], s_o[903 : 903 + 128] = quantize_int8_per_row(wo_blk)
                q_o[:903, q_off : q_off + HD] = 0
                q_o[903 + 128 :, q_off : q_off + HD] = 0

        if stage >= 3:
            # --- LAYER 20: Anti-Repetition / Self-Inhibition Head ---
            if li == 20:
                q_off = 9 * HD
                kv_off = 3 * HD
                wq_blk = np.zeros((HD, D), dtype=np.float32)
                wk_blk = np.zeros((HD, D), dtype=np.float32)
                wv_blk = np.zeros((HD, D), dtype=np.float32)
                for j in range(48, 64):
                    wq_blk[2 * j, F_BIAS] = 1.2
                    wk_blk[2 * j, F_BIAS] = 1.2
                g_inh = 0.55 if stage == 3 else 0.45
                wv_blk[:, 0:HD] = np.eye(HD, dtype=np.float32) * g_inh
                q_q[q_off : q_off + HD], s_q[q_off : q_off + HD] = quantize_int8_per_row(wq_blk)
                q_k[kv_off : kv_off + HD], s_k[kv_off : kv_off + HD] = quantize_int8_per_row(wk_blk)
                q_v[kv_off : kv_off + HD], s_v[kv_off : kv_off + HD] = quantize_int8_per_row(wv_blk)

                wo_blk = np.zeros((HD, D), dtype=np.float32)
                wo_blk[:, q_off : q_off + HD] = np.eye(HD, dtype=np.float32) * g_inh
                q_o[0:HD], s_o[0:HD] = quantize_int8_per_row(wo_blk)
                q_o[HD:, q_off : q_off + HD] = 0

        for q_m, s_m in ((q_q, s_q), (q_k, s_k), (q_v, s_v), (q_o, s_o)):
            emit_tensor(q_m, "I8")
            emit_tensor(s_m, "F16")
        del q_q, s_q, q_k, s_k, q_v, s_v, q_o, s_o

        # post_attention_layernorm
        ln2 = np.ones((D,), dtype=np.float16)
        emit_tensor(ln2, "F16")

        # Initialize SwiGLU MLP matrices as fast INT8 capacity blocks
        q_g, s_g = make_fast_int8_capacity(DFF, D, 2000 + li * 10 + 1, cap_std_mlp)
        q_u, s_u = make_fast_int8_capacity(DFF, D, 2000 + li * 10 + 2, cap_std_mlp)
        q_d, s_d = make_fast_int8_capacity(D, DFF, 2000 + li * 10 + 3, cap_std_mlp)

        if stage >= 3 and li in (1, 6, 12, 16):
            wg_blk = np.zeros((32, D), dtype=np.float32)
            wu_blk = np.zeros((32, D), dtype=np.float32)

            # Prev-token S_cat offset in H10 is 1376 + (flag - 512)
            P_DEF = 1376 + (F_DEF_CLASS - 512)
            P_OPEN = 1376 + (F_OPEN_PAREN - 512)
            P_COLON = 1376 + (F_COLON - 512)
            P_INDENT = 1376 + (F_INDENT - 512)
            P_CTRL = 1376 + (F_CTRL_FLOW - 512)
            P_RET = 1376 + (F_RETURN - 512)

            wg_blk[0, F_DEF_CLASS] = 1.5
            wg_blk[0, P_DEF] = 1.5
            wu_blk[0, F_BIAS] = 1.2

            wg_blk[1, F_OPEN_PAREN] = 1.5
            wg_blk[1, P_OPEN] = 1.2
            wu_blk[1, PY_MODE] = 1.2

            wg_blk[2, F_COLON] = 1.5
            wg_blk[2, P_COLON] = 1.2
            wu_blk[2, PY_MODE] = 1.2

            wg_blk[3, F_INDENT] = 1.5
            wg_blk[3, P_INDENT] = 1.2
            wu_blk[3, PY_MODE] = 1.2

            wg_blk[4, F_CTRL_FLOW] = 1.5
            wg_blk[4, P_CTRL] = 1.2
            wu_blk[4, PY_MODE] = 1.2

            wg_blk[5, F_RETURN] = 1.5
            wg_blk[5, P_RET] = 1.2
            wu_blk[5, F_BIAS] = 1.2

            wg_blk[6, LG_MODE] = 1.5
            wu_blk[6, F_QUESTION] = 1.2
            wu_blk[6, F_COLON] = 1.2

            wg_blk[7, LG_MODE] = 1.5
            wu_blk[7, F_STEP_WORD] = 1.2
            wu_blk[7, F_COMPLEXITY] = 1.0

            wg_blk[8, LG_MODE] = 1.5
            wu_blk[8, F_LOGIC_CONN] = 1.2

            wg_blk[9, CV_MODE] = 1.5
            wu_blk[9, F_GREETING] = 1.2

            wg_blk[10, CV_QUESTION] = 1.5
            wu_blk[10, F_QUESTION] = 1.2

            q_g[0:32], s_g[0:32] = quantize_int8_per_row(wg_blk)
            q_u[0:32], s_u[0:32] = quantize_int8_per_row(wu_blk)

            q_d[:, 0:32] = 0
            # Stage 3 intentionally uses uncalibrated 2.5 (RMS explosion ablation); Stage 4 calibrates to 0.055!
            mlp_g = 2.5 if stage == 3 else 0.055
            edits_d = {}
            for dst_idx, src_neuron in [
                (PY_EXP_IDENT, 0),
                (PY_EXP_PAREN, 1),
                (PY_EXP_INDENT, 2),
                (PY_EXP_RETURN, 3),
                (PY_EXP_LOOP_IN, 4),
                (PY_EXP_EXPR, 5),
                (LG_STEP1, 6),
                (LG_STEP2, 7),
                (LG_STEP3, 8),
                (CV_GREETING, 9),
            ]:
                row_f = q_d[dst_idx].astype(np.float32) * float(s_d[dst_idx, 0])
                row_f[0:32] = 0.0
                row_f[src_neuron] = mlp_g
                if dst_idx == CV_GREETING:
                    row_f[10] = mlp_g * 0.8
                edits_d[dst_idx] = row_f
            q_d, s_d = apply_row_edits_int8(q_d, s_d, edits_d)

        for q_m, s_m in ((q_g, s_g), (q_u, s_u), (q_d, s_d)):
            emit_tensor(q_m, "I8")
            emit_tensor(s_m, "F16")
        del q_g, s_g, q_u, s_u, q_d, s_d

    # 3. Final norm
    final_ln = np.ones((D,), dtype=np.float16)
    emit_tensor(final_ln, "F16")

    # 4. lm_head (32000 x 1536) — Direct Weight Editing Stage!
    if stage == 0:
        lm_head = rng.standard_normal((V, D)).astype(np.float32) * (1.0 / math.sqrt(D))
    else:
        lm_head = np.zeros((V, D), dtype=np.float32)
        sem_scale = 0.25 if stage <= 3 else 0.06
        lm_head[:, 128:512] = E_use[:, 128:512] * sem_scale

        if stage >= 2:
            ind_lm_scale = 4.5 if stage in (2, 3) else 0.35
            topic_lm_scale = 0.6 if stage in (2, 3) else 0.22
            lm_head[:, 1408:1536] = E_use[:, 0:128] * ind_lm_scale
            lm_head[:, 903:1031] = E_use[:, 0:128] * topic_lm_scale

        if stage >= 3:
            inh_lm_scale = 1.4 if stage == 3 else 0.16
            lm_head[:, 0:128] = -E_use[:, 0:128] * inh_lm_scale

            # State gain factor (Stage 3 = 1.0 uncalibrated vs induction hijack; Stage 4 = balanced 1.25 with low induction)
            sg = 1.0 if stage == 3 else 1.25

            for tid in groups["py_ident_good"]:
                lm_head[tid, PY_EXP_IDENT] += 3.5 * sg
                lm_head[tid, PY_MODE] += 0.8 * sg
            for tid in groups["open_paren"]:
                lm_head[tid, F_IDENT] += 1.6 * sg
                lm_head[tid, PY_MODE] += 0.9 * sg
            for tid in groups["py_param"]:
                lm_head[tid, PY_EXP_PAREN] += 3.0 * sg
            for tid in groups["close_colon"]:
                lm_head[tid, PY_EXP_PAREN] += 2.0 * sg
                lm_head[tid, F_IDENT] += 1.2 * sg
                lm_head[tid, PY_MODE] += 0.8 * sg
            for tid in groups["newline_indent"]:
                lm_head[tid, PY_EXP_INDENT] += 3.8 * sg
            for tid in groups["return_kw"]:
                lm_head[tid, PY_EXP_RETURN] += 3.6 * sg
            for tid in groups["py_builtins"]:
                lm_head[tid, PY_EXP_EXPR] += 2.6 * sg
                lm_head[tid, PY_EXP_RETURN] += 2.0 * sg
            for tid in groups["in_kw"]:
                lm_head[tid, PY_EXP_LOOP_IN] += 3.2 * sg

            for tid in groups["analysis_step1"]:
                lm_head[tid, LG_STEP1] += 3.4 * sg
                lm_head[tid, LG_MODE] += 1.2 * sg
            for tid in groups["analysis_step2"]:
                lm_head[tid, LG_STEP2] += 3.4 * sg
                lm_head[tid, LG_COMPLEXITY] += 1.6 * sg
            for tid in groups["complexity_terms"]:
                lm_head[tid, LG_COMPLEXITY] += 2.2 * sg
                lm_head[tid, LG_MODE] += 0.9 * sg
            for tid in groups["stats_terms"]:
                lm_head[tid, LG_MODE] += 1.8 * sg
            for tid in groups["comparison_terms"]:
                lm_head[tid, LG_MODE] += 1.5 * sg
            for tid in groups["analysis_step3"]:
                lm_head[tid, LG_STEP3] += 3.2 * sg

            for tid in groups["conv_greet_reply"]:
                lm_head[tid, CV_GREETING] += 3.2 * sg
            for tid in groups["conv_help_words"]:
                lm_head[tid, CV_MODE] += 1.6 * sg
                lm_head[tid, CV_QUESTION] += 1.4 * sg
            for tid in groups["punct_end"]:
                lm_head[tid, CV_MODE] += 0.7 * sg
                lm_head[tid, LG_MODE] += 0.6 * sg

            # Suppress bare space token 'Ġ' (id=42) from dominating Stage 4
            if stage >= 4:
                lm_head[42, :] = 0.0
                lm_head[42, F_BIAS] = -2.5

    q_lm, s_lm = quantize_int8_per_row(lm_head)
    emit_tensor(q_lm, "I8")
    emit_tensor(s_lm, "F16")
    del lm_head, q_lm, s_lm

    if sf_file is not None:
        sf_file.close()
        elapsed = time.time() - t_start
        size_mb = (OUT_DIR / "model.safetensors").stat().st_size / (1024 * 1024)
        print(f"[Stage {stage}] Wrote model.safetensors ({size_mb:.2f} MB) in {elapsed:.2f}s", flush=True)


def main():
    E, groups = build_token_feature_table()
    results = {"stages": {}, "verification": {}}

    ablation_prompts = [
        ("induction_copy", "alpha = 42 ; beta = 99 ; alpha ="),
        ("python_def", "def "),
        ("analysis_complexity", "Analyze the time complexity of binary search:"),
        ("conversation_hello", "Hello! How can you help me?"),
    ]

    full_prompts = ablation_prompts + [
        ("python_loop", "for i in range(10):\n    "),
        ("analysis_data", "Compare variance and outliers in numerical data analysis:"),
    ]

    # Run Stages 0, 1, 2, 3 (ablations) then Stage 4 (calibrated final model.safetensors)
    for st in [0, 1, 2, 3, 4]:
        synthesize_stage(st, E, groups, save_safetensors=True)
        model = MmapInt8LlamaModel(OUT_DIR, "model.safetensors")
        prompts = full_prompts if st == 4 else ablation_prompts
        stage_out = {}
        for pname, prompt in prompts:
            res = model.generate(prompt, max_new_tokens=10 if st == 4 else 5, temperature=0.0)
            stage_out[pname] = res
            print(f"  [Stage {st} | {pname}] -> {repr(res['generated_text'])} (H={res['first_token_entropy_nats']:.2f}, {res['elapsed_sec']}s)", flush=True)
        results["stages"][f"stage_{st}"] = stage_out
        del model

    # Also test Stage 4 with mild sampling (temp=0.6, rep_penalty=1.3)
    model = MmapInt8LlamaModel(OUT_DIR, "model.safetensors")
    sampled_out = {}
    for pname, prompt in full_prompts:
        res = model.generate(prompt, max_new_tokens=10, temperature=0.6, top_k=20, repetition_penalty=1.3)
        sampled_out[pname] = res
        print(f"  [Stage 4 Sampled | {pname}] -> {repr(res['generated_text'])} ({res['elapsed_sec']}s)", flush=True)
    results["stage_4_sampled"] = sampled_out
    del model

    # Formal Safetensors verification of final Stage 3 model.safetensors
    sf_path = OUT_DIR / "model.safetensors"
    total_core_params = 0
    total_scale_params = 0
    tensor_count = 0
    nan_count = 0
    i8_min_seen, i8_max_seen = 127, -128
    with safe_open(str(sf_path), framework="numpy") as f:
        keys = list(f.keys())
        tensor_count = len(keys)
        for k in keys:
            t = f.get_tensor(k)
            if k.endswith(".weight_scale"):
                total_scale_params += t.size
            else:
                total_core_params += t.size
            if np.issubdtype(t.dtype, np.floating):
                if np.isnan(t).any() or np.isinf(t).any():
                    nan_count += 1
            elif t.dtype == np.int8:
                i8_min_seen = min(i8_min_seen, int(t.min()))
                i8_max_seen = max(i8_max_seen, int(t.max()))

    results["verification"] = {
        "file": str(sf_path),
        "size_bytes": sf_path.stat().st_size,
        "size_mb": round(sf_path.stat().st_size / (1024 * 1024), 2),
        "tensor_count": tensor_count,
        "core_weight_parameters": total_core_params,
        "quantization_scale_scalars": total_scale_params,
        "nan_or_inf_tensors": nan_count,
        "int8_dynamic_range_used": [i8_min_seen, i8_max_seen],
    }

    (OUT_DIR / "experiment_results.json").write_text(json.dumps(results, indent=2))
    print("\nVERIFICATION:", json.dumps(results["verification"], indent=2))


if __name__ == "__main__":
    main()
