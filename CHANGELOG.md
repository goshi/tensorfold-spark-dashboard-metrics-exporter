# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.1.0] - 2026-10-22

### Added
- **`vllm-v1` engine auto-detection + passthrough.** A native vLLM upstream
  (the exlr8 GLM build: `vllm:prompt_tokens_total{engine=…,model_name=…}`) is
  now recognised on every poll. Its colon families are renamed 1:1 to the
  underscore names Spark Dashboard's `VllmAdapter` matches, labels
  (`engine`, `model_name`, `finished_reason`, `le`) are preserved verbatim,
  and every non-vLLM family (`http_*`, `process_*`, `python_*`) is dropped.
  Nothing is synthesized: TTFT/E2E/ITL/TPOT histograms, KV-cache usage,
  prefix-cache counters and spec-decode counters come straight from the
  engine, so percentiles, SLO goodput and cache panels read engine truth.
- **Windowed live rates for the vllm-v1 profile** — `tfexporter_live_decode_tps`
  = Δgeneration tokens / Δdecode seconds and `tfexporter_live_prefill_tps` =
  Δ(prompt − cached tokens) / ΔTTFT seconds, plus
  `tfexporter_cache_hit_ratio`, all from the engine's own counters over the
  600 s window (held while busy, 0 when idle).
- **Documented zero for `vllm_num_requests_swapped`** when the engine build
  does not export it, so the Requests panel sums a complete
  running/waiting/swapped set.

### Changed
- `tfexporter_engine_profile` is now `2` for the native vLLM passthrough
  (`0` TensorFold health, `1` TensorFold native); its `# HELP` enumerates all
  three profiles.
- The probe subsystem is **disabled** in the vllm-v1 profile — a native vLLM
  engine measures its own request latencies, so probing would only add load
  and duplicate honest series (`probe_enabled()`).
- The raw upstream exposition is no longer replayed by the self-status gauges
  for vllm-v1 (it would inject colon families into the output).
- Version banner reports `v1.1.0`.

### Tests
- 20 new unit tests (50 total) covering vllm-v1 detection (including mixed
  expositions), the passthrough contract (label preservation, metadata before
  samples, metadata emitted once, duplicate suppression, non-vLLM families
  dropped, swapped-zero behaviour, adapter series inventory), live-rate math
  and idle decay, probe gating, and the raw-exposition leak regression.

## [1.0.0] - 2026-10-21

First public release.

### Added
- **Engine auto-detection** (`zig-v1` underscore families / `python-v1` colon
  families / unknown pass-through) — one exporter serves both TensorFold
  engine builds.
- **Persistent counters** — cumulative series continue across exporter
  restarts via atomic offsets (`data/counters.json`); a genuine engine
  restart (uptime regression) opens a new counter era and resets offsets,
  probe histograms and the rate hold.
- **zig-v1 bridge contract** — raw TensorFold families rewritten to
  `vllm_*` names, plus engine-derived synthesis:
  - `vllm_request_time_per_output_token_seconds_{sum,count}` from
    decode-seconds/completion-tokens → honest Decode *Per-Req Avg*;
  - `vllm_spec_decode_num_drafts_total` from `/health` rounds_total →
    Mean Acceptance Length panel;
  - real `tensorfold_requests_total` → TTFT histogram count → Requests panel.
- **Probe subsystem** — streaming chat probe every 45 s with a unique
  ~1.4 KB prompt per run and role-stub filtering (only chunks carrying an
  actual output token are stamped) feeding TTFT/E2E/ITL/TPOT histograms →
  dashboard latency percentiles and SLO goodput tiles.
- **TTFT hybrid histogram** — buckets from probe measurements, sum/count
  estimated from engine counters (`prompt_total / windowed_prefill_rate`,
  `requests_total`) so the dashboard's Prefill *Per-Req Avg*
  (`prompt_total / ttft_sum`) equals the true current prefill rate.
- **Windowed rates** — `tfexporter_live_{decode,prefill}_tps` over a 600 s
  window (cache tokens excluded from the prefill rate), held through idle
  phases; windowed `tfexporter_cache_hit_ratio`.
- **Self-observability** — schema info, upstream lag/ok, memory available,
  stalled/fatal counters, engine restarts.
- **30 unit tests** (`tests/test_exporter_metrics.py`) covering schema
  detection, rate math, persistence, the bridge contract, histogram
  bucketing, SLO bucket edges and a deadlock regression guard; GitHub
  Actions CI on Python 3.8/3.12.

### Design notes
- Families without an honest TensorFold source (KV-cache percent, queue
  time) are omitted or emitted as documented zeros — never fabricated.
- The rewrite is written against the flash17 zig build family names; the
  python glm53 path keeps the legacy rename + health-counter synthesis.
