# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
