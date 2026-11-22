#!/usr/bin/env python3
"""
Standalone Memory-Mapped INT8 Safetensors Inference Engine for LlamaForCausalLM (800M)
Supports:
  - Per-channel symmetric INT8 dequantization (weight_i8 * weight_scale_f16)
  - RMSNorm, Rotary Position Embeddings (RoPE), Grouped-Query Attention (GQA), SwiGLU
  - KV-Cache for fast autoregressive decoding in <1 GB RAM
"""
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from safetensors import safe_open
from tokenizers import Tokenizer


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    var = np.mean(x * x, axis=-1, keepdims=True)
    return (x / np.sqrt(var + eps)) * weight


def silu(x: np.ndarray) -> np.ndarray:
    # Numerically stable SiLU
    x_clip = np.clip(x, -40.0, 40.0)
    return x * (1.0 / (1.0 + np.exp(-x_clip)))


def build_rope_cache(max_seq: int, head_dim: int, theta: float = 10000.0) -> Tuple[np.ndarray, np.ndarray]:
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    pos = np.arange(max_seq, dtype=np.float32)
    freqs = np.outer(pos, inv_freq)  # (max_seq, head_dim/2)
    cos = np.cos(freqs)
    sin = np.sin(freqs)
    return cos, sin


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray, start_pos: int) -> np.ndarray:
    """
    x: (seq_len, n_heads, head_dim)
    Applies NeoX/Llama half-split or interleaved RoPE. We use interleaved pairs (2j, 2j+1)
    consistently across weight synthesis and inference.
    """
    seq_len, n_heads, head_dim = x.shape
    c = cos[start_pos : start_pos + seq_len][:, None, :]  # (seq, 1, head_dim/2)
    s = sin[start_pos : start_pos + seq_len][:, None, :]
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out_even = x_even * c - x_odd * s
    out_odd = x_even * s + x_odd * c
    out = np.empty_like(x)
    out[..., 0::2] = out_even
    out[..., 1::2] = out_odd
    return out


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(np.clip(x - x_max, -60.0, 0.0))
    return e / np.sum(e, axis=axis, keepdims=True)


class MmapInt8LlamaModel:
    def __init__(self, model_dir: str | Path, weights_filename: str = "model.safetensors"):
        self.model_dir = Path(model_dir)
        self.config = json.loads((self.model_dir / "config.json").read_text())
        self.tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        self.weights_path = self.model_dir / weights_filename
        self.sf = safe_open(str(self.weights_path), framework="numpy")

        self.d = self.config["hidden_size"]
        self.d_ff = self.config["intermediate_size"]
        self.n_layers = self.config["num_hidden_layers"]
        self.n_heads = self.config["num_attention_heads"]
        self.n_kv_heads = self.config["num_key_value_heads"]
        self.head_dim = self.config["head_dim"]
        self.gqa_groups = self.n_heads // self.n_kv_heads
        self.eps = self.config["rms_norm_eps"]
        self.cos, self.sin = build_rope_cache(
            self.config["max_position_embeddings"], self.head_dim, self.config["rope_theta"]
        )

    def _linear_int8(self, x: np.ndarray, prefix: str) -> np.ndarray:
        """
        Computes x @ (W_int8 * scale).T = (x @ W_int8.T) * scale.T
        Applying per-output-channel scale AFTER matmul avoids allocating/scaling a full 2D float32 weight buffer.
        """
        w_q = self.sf.get_tensor(f"{prefix}.weight")
        scale = self.sf.get_tensor(f"{prefix}.weight_scale").astype(np.float32)
        if w_q.shape[0] > 8192:
            out = np.empty((x.shape[0], w_q.shape[0]), dtype=np.float32)
            chunk = 4096
            for i in range(0, w_q.shape[0], chunk):
                w_blk = w_q[i : i + chunk].astype(np.float32)
                out[:, i : i + chunk] = (x @ w_blk.T) * scale[i : i + chunk].T
            return out
        return (x @ w_q.astype(np.float32).T) * scale.T

    def _embed(self, token_ids: List[int]) -> np.ndarray:
        w_q = self.sf.get_tensor("model.embed_tokens.weight")
        scale = self.sf.get_tensor("model.embed_tokens.weight_scale").astype(np.float32)
        ids = np.array(token_ids, dtype=np.int64)
        return w_q[ids].astype(np.float32) * scale[ids]

    def forward(
        self,
        token_ids: List[int],
        kv_cache: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        start_pos: int = 0,
    ) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
        h = self._embed(token_ids)  # (seq, d)
        seq_len = h.shape[0]
        if kv_cache is None:
            kv_cache = [None] * self.n_layers

        new_cache = []
        scale_attn = 1.0 / math.sqrt(self.head_dim)

        for layer_idx in range(self.n_layers):
            lp = f"model.layers.{layer_idx}"
            ln1 = self.sf.get_tensor(f"{lp}.input_layernorm.weight").astype(np.float32)
            x_norm = rms_norm(h, ln1, self.eps)

            q = self._linear_int8(x_norm, f"{lp}.self_attn.q_proj").reshape(seq_len, self.n_heads, self.head_dim)
            k = self._linear_int8(x_norm, f"{lp}.self_attn.k_proj").reshape(seq_len, self.n_kv_heads, self.head_dim)
            v = self._linear_int8(x_norm, f"{lp}.self_attn.v_proj").reshape(seq_len, self.n_kv_heads, self.head_dim)

            q = apply_rope(q, self.cos, self.sin, start_pos)
            k = apply_rope(k, self.cos, self.sin, start_pos)

            if kv_cache[layer_idx] is not None:
                prev_k, prev_v = kv_cache[layer_idx]
                k_all = np.concatenate([prev_k, k], axis=0)
                v_all = np.concatenate([prev_v, v], axis=0)
            else:
                k_all, v_all = k, v
            new_cache.append((k_all, v_all))

            total_len = k_all.shape[0]
            # Expand KV heads for GQA: (total_len, n_heads, head_dim)
            k_exp = np.repeat(k_all, self.gqa_groups, axis=1)
            v_exp = np.repeat(v_all, self.gqa_groups, axis=1)

            # Attention per head: (n_heads, seq_len, total_len)
            q_t = np.transpose(q, (1, 0, 2))
            k_t = np.transpose(k_exp, (1, 2, 0))
            v_t = np.transpose(v_exp, (1, 0, 2))

            scores = (q_t @ k_t) * scale_attn
            # Causal mask
            q_pos = np.arange(start_pos, start_pos + seq_len)[:, None]
            k_pos = np.arange(total_len)[None, :]
            mask = np.where(k_pos <= q_pos, 0.0, -1e9).astype(np.float32)
            scores = scores + mask[None, :, :]

            attn_probs = softmax(scores, axis=-1)
            attn_out = attn_probs @ v_t  # (n_heads, seq_len, head_dim)
            attn_out = np.transpose(attn_out, (1, 0, 2)).reshape(seq_len, self.d)
            h = h + self._linear_int8(attn_out, f"{lp}.self_attn.o_proj")

            # SwiGLU MLP
            ln2 = self.sf.get_tensor(f"{lp}.post_attention_layernorm.weight").astype(np.float32)
            m_norm = rms_norm(h, ln2, self.eps)
            gate = silu(self._linear_int8(m_norm, f"{lp}.mlp.gate_proj"))
            up = self._linear_int8(m_norm, f"{lp}.mlp.up_proj")
            down = self._linear_int8(gate * up, f"{lp}.mlp.down_proj")
            h = h + down

        final_ln = self.sf.get_tensor("model.norm.weight").astype(np.float32)
        h_final = rms_norm(h[-1:, :], final_ln, self.eps)
        logits = self._linear_int8(h_final, "lm_head")[0]  # (vocab_size,)
        return logits, new_cache

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 20,
        temperature: float = 0.0,
        top_k: int = 40,
        repetition_penalty: float = 1.0,
    ) -> Dict:
        t0 = time.time()
        enc = self.tokenizer.encode(prompt)
        prompt_ids = enc.ids if len(enc.ids) > 0 else [0]
        logits, cache = self.forward(prompt_ids, kv_cache=None, start_pos=0)

        first_probs = softmax(logits)
        top5_idx = np.argsort(logits)[-5:][::-1]
        top5 = [(self.tokenizer.decode([int(i)]), float(logits[i]), float(first_probs[i])) for i in top5_idx]
        entropy = float(-np.sum(first_probs * np.log(np.clip(first_probs, 1e-12, 1.0))))

        generated_ids = []
        cur_pos = len(prompt_ids)
        rng = np.random.default_rng(42)

        for step in range(max_new_tokens):
            step_logits = logits.copy()
            if repetition_penalty != 1.0:
                seen = set(prompt_ids + generated_ids)
                for tid in seen:
                    if step_logits[tid] > 0:
                        step_logits[tid] /= repetition_penalty
                    else:
                        step_logits[tid] *= repetition_penalty

            if temperature <= 0.0:
                next_id = int(np.argmax(step_logits))
            else:
                scaled = step_logits / temperature
                if top_k > 0:
                    idx = np.argpartition(scaled, -top_k)[-top_k:]
                    probs = softmax(scaled[idx])
                    next_id = int(rng.choice(idx, p=probs))
                else:
                    probs = softmax(scaled)
                    next_id = int(rng.choice(len(probs), p=probs))

            generated_ids.append(next_id)
            if next_id == self.config["eos_token_id"]:
                break
            if step + 1 < max_new_tokens:
                logits, cache = self.forward([next_id], kv_cache=cache, start_pos=cur_pos)
                cur_pos += 1

        elapsed = time.time() - t0
        return {
            "prompt": prompt,
            "prompt_tokens": len(prompt_ids),
            "generated_ids": generated_ids,
            "generated_text": self.tokenizer.decode(generated_ids),
            "top5_first_token": top5,
            "first_token_entropy_nats": entropy,
            "elapsed_sec": round(elapsed, 2),
        }


if __name__ == "__main__":
    import sys
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "/home/user/ai_experiment"
    m = MmapInt8LlamaModel(model_dir)
    res = m.generate("def fibonacci(n):", max_new_tokens=12, temperature=0.0)
    print(json.dumps(res, indent=2))
