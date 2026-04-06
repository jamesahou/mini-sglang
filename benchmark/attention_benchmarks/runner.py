"""
Mini-SGLang-James attention benchmark runner.

Calls FA4 and TRTLLM-MHA kernels directly, without model loading.
Mirrors the structure of vllm/benchmarks/attention_benchmarks/runner.py.
"""

import math

import numpy as np
import torch
from batch_spec import parse_batch_spec
from common import BenchmarkConfig, BenchmarkResult, get_attention_scale

# ---------------------------------------------------------------------------
# Kernel imports
# ---------------------------------------------------------------------------

try:
    from minisgl.kernel.flash_attention_v4 import (
        flash_attn_with_kvcache as _fa4_with_kvcache,
    )
except ImportError:
    _fa4_with_kvcache = None

try:
    import flashinfer.decode as _fi_decode
    import flashinfer.prefill as _fi_prefill
except ImportError:
    _fi_decode = None
    _fi_prefill = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WORKSPACE_BYTES = 128 * 1024 * 1024  # 128 MB, matches minisgl/attention/trtllm.py


def _require(obj, name: str):
    if obj is None:
        raise ImportError(f"Required module not available: {name}")


# ---------------------------------------------------------------------------
# Metadata building
# ---------------------------------------------------------------------------

def _build_metadata(
    q_lens: list[int],
    kv_lens: list[int],
    block_size: int,
    device: torch.device,
):
    """Build paged-attention metadata tensors."""
    batch_size = len(q_lens)
    max_kv = max(kv_lens)
    max_blocks_per_req = math.ceil(max_kv / block_size)
    num_blocks_total = batch_size * max_blocks_per_req

    # Contiguous block assignment: request i gets blocks [i*B .. (i+1)*B)
    page_table = torch.arange(
        num_blocks_total, dtype=torch.int32, device=device
    ).view(batch_size, max_blocks_per_req)

    cache_seqlens = torch.tensor(kv_lens, dtype=torch.int32, device=device)

    cu_seqlens_k = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cu_seqlens_k[1:] = torch.tensor(kv_lens, dtype=torch.int32).cumsum(0).to(device)

    cu_seqlens_q = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cu_seqlens_q[1:] = torch.tensor(q_lens, dtype=torch.int32).cumsum(0).to(device)

    max_seqlen_q = max(q_lens)
    max_seqlen_k = max(kv_lens)

    return page_table, cache_seqlens, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k


# ---------------------------------------------------------------------------
# Tensor creation
# ---------------------------------------------------------------------------

def _create_input_tensors(
    config: BenchmarkConfig,
    total_q: int,
    device: torch.device,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    """Create one Q tensor per layer."""
    return [
        torch.randn(total_q, config.num_q_heads, config.head_dim, dtype=dtype, device=device)
        for _ in range(config.num_layers)
    ]


def _create_kv_cache(
    config: BenchmarkConfig,
    num_blocks: int,
    device: torch.device,
    dtype: torch.dtype,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    Allocate paged KV caches in NHD layout: [num_blocks, block_size, num_kv_heads, head_dim].

    Both FA4 and TRTLLM use NHD layout in minisgl (kv_layout="NHD").
    """
    return [
        (
            torch.zeros(num_blocks, config.block_size, config.num_kv_heads, config.head_dim,
                        dtype=dtype, device=device),
            torch.zeros(num_blocks, config.block_size, config.num_kv_heads, config.head_dim,
                        dtype=dtype, device=device),
        )
        for _ in range(config.num_layers)
    ]


# ---------------------------------------------------------------------------
# Per-backend forward functions
# ---------------------------------------------------------------------------

def _forward_fa4(
    q: torch.Tensor,
    kv_caches: list[tuple[torch.Tensor, torch.Tensor]],
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    scale: float,
):
    _require(_fa4_with_kvcache, "minisgl.kernel.flash_attention_v4")
    for k_cache, v_cache in kv_caches:
        _fa4_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k_new=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            softmax_scale=scale,
            causal=True,
        )


def _forward_trtllm(
    q: torch.Tensor,
    kv_caches: list[tuple[torch.Tensor, torch.Tensor]],
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    scale: float,
    workspace: torch.Tensor,
    is_decode: bool,
    batch_size: int,
):
    _require(_fi_decode, "flashinfer.decode")
    _require(_fi_prefill, "flashinfer.prefill")
    for k_cache, v_cache in kv_caches:
        kv_cache = (k_cache, v_cache)
        if is_decode:
            _fi_decode.trtllm_batch_decode_with_kv_cache(
                query=q,
                kv_cache=kv_cache,
                workspace_buffer=workspace,
                block_tables=page_table,
                seq_lens=cache_seqlens,
                max_seq_len=max_seqlen_k,
                bmm1_scale=scale,
                bmm2_scale=1.0,
                kv_layout="NHD",
                out_dtype=q.dtype,
            )
        else:
            _fi_prefill.trtllm_batch_context_with_kv_cache(
                query=q,
                kv_cache=kv_cache,
                workspace_buffer=workspace,
                block_tables=page_table,
                seq_lens=cache_seqlens,
                max_q_len=max_seqlen_q,
                max_kv_len=max_seqlen_k,
                bmm1_scale=scale,
                bmm2_scale=1.0,
                cum_seq_lens_q=cu_seqlens_q,
                cum_seq_lens_kv=cu_seqlens_k,
                batch_size=batch_size,
                kv_layout="NHD",
                out_dtype=q.dtype,
            )


# ---------------------------------------------------------------------------
# Benchmark execution  (mirrors vllm/benchmarks/attention_benchmarks/runner.py)
# ---------------------------------------------------------------------------

def _run_single_benchmark(
    call_all_layers,
    config: BenchmarkConfig,
    device: torch.device,
) -> tuple[list[float], dict]:
    """Run warmup, optional CUDA graph capture, and timed loop.

    Returns (times_per_layer_seconds, mem_stats).
    Mirrors _run_single_benchmark in the vLLM runner.
    """
    # Warmup
    for _ in range(config.warmup_iters):
        call_all_layers()
    torch.cuda.synchronize()

    # Optionally capture a CUDA graph after warmup
    if config.use_cuda_graphs:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call_all_layers()
        benchmark_fn = graph.replay
    else:
        benchmark_fn = call_all_layers

    # Delimit the timed window for nsys --capture-range=cudaProfilerApi.
    # No-op when nsys is not attached.
    torch.cuda.cudart().cudaProfilerStart()

    # Timed loop
    times = []
    for i in range(config.repeats):
        torch.cuda.nvtx.range_push(f"iter_{i}")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        benchmark_fn()
        end.record()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        # Normalise to seconds-per-layer so multi-layer configs are comparable
        times.append(start.elapsed_time(end) / 1000.0 / config.num_layers)

    torch.cuda.cudart().cudaProfilerStop()

    mem_stats = {}
    if config.profile_memory:
        mem_stats = {
            "allocated_mb": torch.cuda.memory_allocated(device) / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved(device) / 1024**2,
        }

    return times, mem_stats


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_attention_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    """
    Run a direct-kernel attention benchmark for mini-sglang-james FA4 or TRTLLM-MHA.

    Args:
        config: Benchmark configuration. config.backend must be "fa4" or "trtllm".

    Returns:
        BenchmarkResult with timing and memory statistics.
    """
    device = torch.device(config.device)
    dtype = config.dtype

    requests = parse_batch_spec(config.batch_spec)
    q_lens = [r.q_len for r in requests]
    kv_lens = [r.kv_len for r in requests]
    total_q = sum(q_lens)
    batch_size = len(q_lens)

    page_table, cache_seqlens, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k = (
        _build_metadata(q_lens, kv_lens, config.block_size, device)
    )

    num_blocks_total = batch_size * math.ceil(max(kv_lens) / config.block_size)
    scale = get_attention_scale(config.head_dim)
    is_decode = max_seqlen_q == 1

    q_list = _create_input_tensors(config, total_q, device, dtype)
    kv_caches = _create_kv_cache(config, num_blocks_total, device, dtype)

    backend = config.backend.lower()

    if backend == "fa4":
        def call_all_layers():
            for q in q_list:
                _forward_fa4(
                    q, kv_caches, page_table, cache_seqlens,
                    cu_seqlens_q, cu_seqlens_k, max_seqlen_q, scale,
                )

    elif backend == "trtllm":
        workspace = torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device=device)

        def call_all_layers():
            for q in q_list:
                _forward_trtllm(
                    q, kv_caches, page_table, cache_seqlens,
                    cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                    scale, workspace, is_decode, batch_size,
                )

    else:
        raise ValueError(
            f"Unknown backend: '{config.backend}'. Valid options: fa4, trtllm"
        )

    times, mem_stats = _run_single_benchmark(call_all_layers, config, device)

    mean_time = float(np.mean(times))
    throughput = total_q / mean_time if mean_time > 0 else 0.0

    return BenchmarkResult(
        config=config,
        mean_time=mean_time,
        std_time=float(np.std(times)),
        min_time=float(np.min(times)),
        max_time=float(np.max(times)),
        throughput_tokens_per_sec=throughput,
        memory_allocated_mb=mem_stats.get("allocated_mb"),
        memory_reserved_mb=mem_stats.get("reserved_mb"),
    )
