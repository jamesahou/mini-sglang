"""
A/B test: does passing explicit max_seqlen_k improve FA4 performance?

Current behavior: flash_attn_with_kvcache drops max_seqlen_k, so the CUTE
interface derives max_seqlen_k = num_pages * page_size (inflated by batch_size).

Fix: call flash_attn_varlen_func directly with max_seqlen_k = max(kv_lens).

Run:
    python test_maxseqlenk.py
"""

import math
import time

import numpy as np
import torch
from minisgl.kernel.flash_attention_v4 import (
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
)

# ---------------------------------------------------------------------------
# Config (Qwen3-14B dims, block_size=128 = TMA path)
# ---------------------------------------------------------------------------
NUM_Q_HEADS = 40
NUM_KV_HEADS = 8
HEAD_DIM = 128
BLOCK_SIZE = 128
KV_LEN = 60000
DEVICE = "cuda:0"
DTYPE = torch.float16
WARMUP = 10
REPEATS = 20
SCALE = 1.0 / math.sqrt(HEAD_DIM)

BATCH_SIZES = [1, 4, 16, 64, 128]
Q_LEN = 1  # decode


def build_tensors(batch_size):
    total_q = batch_size * Q_LEN
    max_blocks_per_req = math.ceil(KV_LEN / BLOCK_SIZE)
    num_blocks_total = batch_size * max_blocks_per_req

    q = torch.randn(total_q, NUM_Q_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    k_cache = torch.zeros(num_blocks_total, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    v_cache = torch.zeros(num_blocks_total, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)

    page_table = torch.arange(num_blocks_total, dtype=torch.int32, device=DEVICE).view(batch_size, max_blocks_per_req)
    cache_seqlens = torch.full((batch_size,), KV_LEN, dtype=torch.int32, device=DEVICE)

    cu_seqlens_q = torch.arange(0, batch_size + 1, dtype=torch.int32, device=DEVICE) * Q_LEN
    cu_seqlens_k = torch.arange(0, batch_size + 1, dtype=torch.int32, device=DEVICE) * KV_LEN

    return q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, cu_seqlens_k


def bench(fn, warmup=WARMUP, repeats=REPEATS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # ms
    return times


def main():
    print(f"{'batch':>6} | {'inflated (ms)':>14} {'fixed (ms)':>12} {'speedup':>8} | "
          f"{'inflated max_seqlen_k':>22} {'fixed':>8}")
    print("-" * 85)

    for bs in BATCH_SIZES:
        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, cu_seqlens_k = build_tensors(bs)

        max_seqlen_q = Q_LEN
        max_seqlen_k = KV_LEN  # correct value

        # Derived (inflated) value — what the CUTE interface computes when max_seqlen_k=None
        num_pages = k_cache.shape[0]
        page_size = k_cache.shape[1]
        inflated_max_seqlen_k = num_pages * page_size

        # --- A: current behavior (inflated max_seqlen_k) ---
        def call_inflated():
            flash_attn_with_kvcache(
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                page_table=page_table,
                cache_seqlens=cache_seqlens,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k_new=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                softmax_scale=SCALE,
                causal=True,
            )

        # --- B: fixed (explicit max_seqlen_k) ---
        def call_fixed():
            flash_attn_varlen_func(
                q=q,
                k=k_cache,
                v=v_cache,
                cu_seqlens_q=cu_seqlens_q,
                seqused_k=cache_seqlens,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                page_table=page_table,
                softmax_scale=SCALE,
                causal=True,
                num_splits=1,
            )

        times_inflated = bench(call_inflated)
        times_fixed = bench(call_fixed)

        mean_inf = np.mean(times_inflated)
        mean_fix = np.mean(times_fixed)
        speedup = mean_inf / mean_fix if mean_fix > 0 else float('inf')

        print(f"{bs:>6} | {mean_inf:>12.3f}ms {mean_fix:>10.3f}ms {speedup:>7.2f}x | "
              f"{inflated_max_seqlen_k:>22,} {max_seqlen_k:>8,}")


if __name__ == "__main__":
    main()
