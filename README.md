# vLLM Inference Benchmarking: Prefix Caching & Batch Size Sweep

Benchmarks the effect of prefix caching and `max_num_seqs` (server-side batch
concurrency limit) on throughput and latency when serving Qwen2.5-7B-Instruct
with vLLM, using a fixed 5-shot MMLU evaluation as the workload.

## Setup

- **Model:** Qwen/Qwen2.5-7B-Instruct
- **Serving engine:** vLLM (`vllm serve`), OpenAI-compatible API
- **Hardware:** NVIDIA RTX PRO 6000 Blackwell Server Edition (single GPU)
- **Workload:** 500 MMLU test questions, each with a fixed 5-shot prompt
  prefix (same 5 examples prepended to every question), sampled from
  `cais/mmlu` ("all" config)
- **Client:** async OpenAI Python client, requests fired concurrently and
  bounded with an `asyncio.Semaphore`

## What was varied

| Variable | Values |
|---|---|
| `--enable-prefix-caching` | on / off |
| `--max-num-seqs` | 32, 128, 256 |
| Prompt order | sorted by length / shuffled |

Client-side concurrency was matched to each run's `max_num_seqs` value, so
the server always had enough in-flight requests to actually exercise its
batch limit.

## Results

Throughput (req/sec), averaged across sorted/shuffled prompt order, one run
per configuration:

| max_num_seqs | caching off | caching on |
|---|---|---|
| 32 | ~282 | ~245 |
| 128 | ~85 | ~353 |
| 256 | ~76 | ~80 |

Accuracy was 0.736-0.742 in every configuration, essentially unchanged
throughout, as expected since these flags affect serving performance, not
model correctness.

**Findings:**

- Prefix caching gave a large, real throughput improvement (~4x) at
  `max_num_seqs=128`, consistent with a measured 97.9% prefix cache hit
  rate in the server logs at that setting. All 500 requests share the same
  5-shot prefix, so this is the regime the optimization is designed for.
- At `max_num_seqs=256`, throughput for both caching on and off collapsed
  to roughly the same low value. This points to GPU memory pressure from
  holding that many concurrent KV caches becoming the dominant bottleneck,
  large enough that it swamps any benefit from caching.
- At `max_num_seqs=32`, caching off outperformed caching on. This is the
  one result that runs against the overall trend, and with only a single
  run per configuration at low concurrency, it's plausible this specific
  data point is noise rather than a real effect. It's reported as-is
  rather than smoothed over.
- A first attempt at this sweep sent requests sequentially (one at a time),
  which showed no meaningful difference across any configuration. That
  result was correct but uninformative: `max_num_seqs` is a concurrency
  batching control, so it can't show an effect unless multiple requests
  are actually in flight at once. The results above come from a rebuilt
  concurrent version of the harness.

## Known limitations

- Each configuration was run once (per prompt order), not averaged over
  multiple trials. The 128-seq caching gap is large enough to be a
  confident result; smaller gaps (notably the 32-seq case) are within the
  range that run-to-run noise could produce.
- `max_num_seqs=256` was not investigated further to find the exact point
  where throughput starts degrading; the sweep only shows that it happens
  somewhere between 128 and 256 for this model/GPU combination.
- Single GPU, single model, single prompt shape (MMLU 5-shot). Results are
  specific to this setup and shouldn't be generalized to other models or
  workloads without re-testing.

## Files

- `sweep_results_concurrent/` — per-run summary and detail JSON, plus raw
  server logs, for all 12 runs (2 caching states × 3 max_num_seqs × 2
  prompt orders)
- `throughput_comparison.png` — bar chart of throughput by caching state
  and max_num_seqs
- `vLLM.ipynb` — notebook containing the eval harness, sweep runner, and
  analysis
