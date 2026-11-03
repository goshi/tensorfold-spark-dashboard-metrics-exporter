# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.3.0] - 2026-11-03

### Added
- **Engine-supplied live decode rate.** When the TensorFold server publishes a
  `live` object in `/health` (`connections`, `waiting`,
  `decode_tokens_per_second`, `prefill_tokens_per_second`), its decode rate now
  feeds `tfexporter_live_decode_tps` directly. The engine keeps its own 2 s
  sliding window (`core/lane_host.zig`: tokens over `window_ns`, divided by
  2.0), so the exporter no longer re-derives a number the engine already
  measured. A *fresh* 0 is taken as the truth ("nothing decoded in the last two
  seconds") rather than being replaced by our held window's last nonzero rate.
- **Provenance and cross-check series.** `tfexporter_live_decode_tps_source`
  (1 = the engine's own window, 0 = exporter-derived) and
  `tfexporter_live_decode_tps_derived` (our window, always emitted) ship next to
  the tile, so a divergence between the two is visible instead of hidden.
- **Queue/connection series from the same snapshot** as the live rate:
  `tfexporter_queue_waiting`, `tfexporter_live_connections`.
- **The engine's prefill figure, kept informational.**
  `tfexporter_engine_prefill_lastjob_tps` plus a `tfexporter_prefilling` 0/1
  flag. It is deliberately *not* used as the prefill rate: the engine writes it
  once per finished prompt pass, timed from admission (so it carries queue
  time), into one global last-writer-wins slot. Measured on the live pair, it
  reported 197/603/1458 tok/s while the same engine was serving a 49K-token
  prompt at ~1.9K tok/s.

### Changed
- No double smoothing: the exporter applies no EMA on top of the engine's
  window. Stacking our filter on the engine's 2 s window would only add lag.
- Prefill HELP now matches the code per profile: `zig-v1` divides by
  `tensorfold_prefill_seconds_total` (engine compute, no queue time), while
  `python-v1`/health profiles divide by TTFT sums (queue included). Both state
  that the window value is an aggregate, not a per-job rate - concurrent
  requests and prefix-cache hits can push it far above any single request.

### Fixed
- `tfexporter_live_decode_tps` is emitted from two places (`self_status_lines()`
  for the `vllm-v1` passthrough, `rewrite()` for both TensorFold profiles). The
  first cut of this release changed only one of them and a live run against the
  engine caught the tile still serving the derived number; the decision now
  lives in one helper, `live_decode_choice()`, that both emitters call, and a
  test asserts the two agree.

### Notes
- The tile can lag the engine by up to one poll interval (`TF_POLL_INTERVAL`,
  default 2 s): the exporter publishes the last `/health` snapshot it fetched.
- Freshness is `TF_LIVE_FRESH_S` (default 5 s). Past it - engine restarted,
  endpoint unreachable, or an old build without the block - the derived window
  takes over and `…_source` flips to 0.
- Health-profile engines without a live block keep the live completion-counter
  diff as both the tile and the derived value; if such an engine grows the
  block, it upgrades automatically (covered by a test).

### Tests
- 19 new tests (53 -> 72), one behaviour each, covering every fallback
  transition: block present, block absent, block stale, block malformed
  (`true`/`"137"` must not become a rate), fresh-zero taken over a held
  derived rate, engine prefill staying informational, the engine-only series
  absent when there is no block, both emitters agreeing, the health profile
  keeping its diff and upgrading when a block appears, the HELP text naming the
  denominator the code actually uses, and a worker hammering both helpers to
  prove no lock re-entrancy. Assertions check emitted sample lines, not raw
  substrings: a metric name mentioned in another gauge's HELP is not emission.

## [1.2.0] - 2026-11-01

### Added
- **Synthetic probe auto-disable.** `probe_enabled()` now also refuses to probe
  when the upstream already publishes its own latency histograms
  (`time_to_first_token_seconds`, `request_time_per_output_token_seconds`).
  The detector `note_engine_latency_hist()` runs on every poll against the
  **upstream** `/metrics` body only — never against the exporter's own output,
  which carries the same names after rewriting. When the families disappear
  again (older engine, upstream change) the probe returns on the next poll.
  Rationale: current TensorFold engines (1.1.x) serve real TTFT/E2E/ITL/TPOT
  histograms over all traffic, so the 45 s synthetic request was pure cost —
  ~2.8k prompt tokens per probe, a few percent of prefill capacity, while
  contaminating those honest histograms with one fixed-size artificial
  request. The probe stays exactly where it is needed: engines that cannot
  measure their own latency.
  Note `note_engine_latency_hist()` writes a plain module global and takes no
  lock, so it is safe to call while `STATE.lock` is held elsewhere.

### Tests
- 3 new unit tests extend the probe-gate contract: probe off when the engine
  publishes histograms, on when it does not, `vllm-v1` stays off regardless,
  and both transitions of the detector in one test. The module-level detector
  flag is saved and restored per test so it cannot leak into other cases.
- The probe-nonce test now closes the source file it reads (drops a
  `ResourceWarning` and an unused import from the suite output).

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
