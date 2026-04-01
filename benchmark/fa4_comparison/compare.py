"""Compare FA4 benchmark results from mini-sglang-james and vLLM.

Usage:
    python compare.py fa4_minisgl_result.json fa4_vllm_result.json
"""

import json
import sys


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def fmt(val: float, unit: str = "") -> str:
    return f"{val:.1f}{unit}"


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    a = load(sys.argv[1])
    b = load(sys.argv[2])

    # Sort so mini-sglang is always first if present
    results = sorted([a, b], key=lambda r: (0 if "mini-sglang" in r["system"] else 1))

    print(f"\n{'='*60}")
    print(f"  FlashAttention 4 Throughput Comparison")
    print(f"  Model: {results[0]['model']}")
    print(f"  Seqs/trial: {results[0]['num_seqs']}  |  Warmup batches: {results[0]['num_warmup']}  |  Trials: {results[0]['num_trials']} (min+max discarded)")
    print(f"{'='*60}")

    for r in results:
        trials_str = "  ".join(fmt(x) for x in r["trial_throughputs"])
        print(f"\n  {r['system']} (FA4)")
        print(f"    Throughput:  {fmt(r['throughput_mean'])} ± {fmt(r['throughput_std'])} tok/s")
        print(f"    All trials:  [{trials_str}] tok/s")
        print(f"    Output toks: {r['total_tokens']:,} per trial")

    if len(results) == 2:
        r0, r1 = results[0], results[1]
        ratio = r0["throughput_mean"] / r1["throughput_mean"]
        faster = r0["system"] if ratio >= 1.0 else r1["system"]
        slower = r1["system"] if ratio >= 1.0 else r0["system"]
        ratio_display = ratio if ratio >= 1.0 else 1.0 / ratio
        print(f"\n  Ratio ({r0['system']} / {r1['system']}): {ratio:.3f}x")
        print(f"  Winner: {faster} is {ratio_display:.2f}x faster than {slower}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
