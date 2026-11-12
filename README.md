# tensorfold-metrics-exporter

A zero-dependency Python bridge that exposes [TensorFold](https://github.com/0rand/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark)
serving metrics under **vLLM** metric family names — so
[Spark Dashboard](https://github.com/niklasfrick/spark-dashboard) can monitor a
TensorFold engine out of the box, with honest numbers on every panel.

```
                 +------------------------+         +---------------------------+
 TensorFold ---> |                        | <------ |                           |
 :8002 /metrics  |   tf-exporter :8004    |  /ws    |    Spark Dashboard :3000  |
  /v1/chat/...   |  bridge + probe + est. |  poll   |     engine = "vLLM"       |
                 +------------------------+         +---------------------------+
```

The exporter polls the upstream `/metrics`, rewrites the TensorFold families
into their `vllm_*` equivalents, synthesizes the derived series the dashboard
expects (per-request decode/prefill averages, live rates, cache ratio,
speculative-decoding acceptance), measures request-latency histograms with its
own streaming probe **only when the engine cannot measure them itself**
(TTFT / E2E / ITL / TPOT — feeding the SLO goodput tiles),
and proxies every other GET path (`/v1/models`, `/health`) to the upstream
unchanged.

A third profile covers a **native vLLM engine** (e.g. the exlr8 GLM build):
there the upstream already serves honest `vllm:*` families, so the exporter
only renames the colon namespace to the underscore names the adapter matches —
no synthesis, no probe.

No changes to the engine or the dashboard are required.

## Why not a plain rewrite?

The dashboard's `VllmAdapter` computes several panel values with formulas that
assume vLLM's own metric semantics:

| Dashboard value | Adapter formula | TensorFold reality | What the exporter does |
|---|---|---|---|
| Decode *Per-Req Avg* | `count / sum` of `vllm_request_time_per_output_token_seconds` | engine only reports aggregate decode seconds + completion tokens | emits that family from engine counters → true per-stream average over **all** traffic |
| Prefill *Per-Req Avg* | `prompt_tokens_total / ttft_seconds_sum` | engine has no TTFT histogram; probe sum alone would cover ~seconds of traffic → garbage ratio | emits the TTFT histogram **hybrid**: buckets from live probes, `sum = prompt_total / windowed_prefill_rate`, `count = engine requests_total` → the panel shows the true current prefill rate |
| *Live* decode/prefill rates | counter deltas between its own (sparse) polls | engine counters move continuously | windowed rates over a 600 s window, held while idle, exported as `tfexporter_live_{decode,prefill}_tps` and the counters kept continuous |
| SLO goodput, percentiles | histogram buckets | engine exposes none | exporter-run streaming probe measures real requests |
| Mean acceptance length | `accepted / drafts` | engine reports rounds, not "drafts" | maps `rounds_total` → `vllm_spec_decode_num_drafts_total` |

Every synthesized series is labeled in its `# HELP` text. Families without an
honest source (KV-cache percent, queue time) are **omitted, not faked** — the
dashboard shows `--` rather than a made-up number.

## Engine auto-detection

`detect_schema()` classifies the upstream exposition on every poll:

- **`vllm-v1`** — a native vLLM engine (`vllm:prompt_tokens_total{engine=…,model_name=…}`),
  e.g. the exlr8 GLM build. Passed through: colon families renamed 1:1 to
  underscore names, labels preserved, non-vLLM families dropped.
- **`velo-v1`** — `velogb10_prompt_tokens_total` present: the veloGB10 GB10
  engine (EXL3 packs, MTP spec-decode, per-lane KV pre-allocation). See
  "Metric mapping (velo-v1 engine)" below.
- **`flare-v1`** — `flare_prefill_tokens_total` present: the Flare (Hyperflare)
  Rust engine (NVFP4 cores, b12x loader, draft-spec decode). See
  "Metric mapping (flare-v1 engine)" below.
- **`zig-v1`** — underscore families (`tensorfold_prompt_tokens_total{model=…}`),
  the flash17 Zig engine. Primary target: raw families + `/health` JSON
  (in-flight, spec-decode counters, streams).
- **`python-v1`** — colon families (`tensorfold:prompt_tokens_total`), the
  Python glm53 engine. Legacy path: family rename + health-counter synthesis.
- **`unknown`** — nothing rewritten; the exposition passes through untouched.

## Metric mapping (vllm-v1 engine — native vLLM passthrough)

Nothing is synthesized in this profile: the engine's own series are renamed
and served as-is, so every dashboard panel reads engine truth.

| Native vLLM source | Exported as | Notes |
|---|---|---|
| `vllm:prompt_tokens_total` | `vllm_prompt_tokens_total` | counter, labels kept |
| `vllm:generation_tokens_total` | `vllm_generation_tokens_total` | counter, labels kept |
| `vllm:num_requests_{running,waiting}` | `vllm_num_requests_{running,waiting}` | gauges |
| `vllm:num_requests_swapped` | `vllm_num_requests_swapped` | documented zero when the build does not export it |
| `vllm:kv_cache_usage_perc` | `vllm_kv_cache_usage_perc` | ratio 0…1 (panel ×100) |
| `vllm:prefix_cache_{hits,queries}_total` | `vllm_prefix_cache_{hits,queries}_total` | hit rate = hits/queries |
| `vllm:time_to_first_token_seconds_{bucket,sum,count}` | `vllm_time_to_first_token_seconds_*` | real histogram → percentiles + SLO goodput |
| `vllm:request_time_per_output_token_seconds_{sum,count}` | `vllm_request_time_per_output_token_seconds_*` | Decode *Per-Req Avg* |
| `vllm:e2e_request_latency_seconds_*`, `vllm:inter_token_latency_seconds_*` | same names, underscores | E2E / ITL panels |
| `vllm:request_queue_time_seconds_*`, `vllm:num_preemptions_total`, `vllm:iteration_tokens_total_*` | same names, underscores | queue, preemption, batch panels |
| `vllm:spec_decode_num_{drafts,draft_tokens,accepted_tokens}_total` | same names, underscores | acceptance rate + mean acceptance length |
| `tfexporter_live_{decode,prefill}_tps` | derived | windowed deltas of the engine counters (held while busy, 0 idle) |
| `tfexporter_engine_profile` | `2` | profile marker (`0` health, `1` TensorFold native, `2` native vLLM) |

`http_*`, `process_*` and `python_*` families of the engine exposition are
dropped — the dashboard consumes only `vllm_*` and `tfexporter_*`.

## Persistent counters

Cumulative series survive **exporter restarts**: every poll persists
`{family: {engine, emitted}}` offsets to `data/counters.json` (atomic write)
and serves `emitted = engine + offset`. Your dashboards never see a counter
regression just because the exporter was restarted.

A genuine **engine restart** (uptime regression) opens a new counter era:
offsets, probe histograms and the windowed-rate hold are cleared so the new
era starts honest.

## Probe subsystem

Every 45 s the exporter streams a real chat request through the engine:

- the prompt is **unique per probe** (~1.4 KB of uuid filler) — a repeated
  prompt would be fully prefix-cached and measure nothing;
- only chunks carrying an **actual output token** are stamped — the Zig engine
  emits an empty `{role}` stub *before* prefill completes, which would
  otherwise fake sub-millisecond TTFTs;
- feeds `vllm_time_to_first_token_seconds`, `vllm_e2e_request_latency_seconds`,
  `vllm_inter_token_latency_seconds` and `vllm_time_per_output_token_seconds`
  histograms → the dashboard's latency percentiles and SLO goodput tiles
  (TTFT ≤ 500 ms, ITL/TPOT ≤ 50 ms, E2E ≤ 5 s).

### Auto-disable: never probe an engine that measures its own latency

The probe is a fallback for engines that expose no latency data, not a
permanent fixture. On every poll the exporter checks the **upstream** body for
`time_to_first_token_seconds` / `request_time_per_output_token_seconds`
(`note_engine_latency_hist()` → `probe_enabled()`); when they are present the
probe thread stays asleep. Detection deliberately ignores the exporter's own
exposition, which carries the same names after rewriting.

Probing such an engine would cost real prefill on the serving GPU (~2.8k
prompt tokens per request at the current filler size) and add one fixed-size
artificial sample to the honest histogram it is trying to imitate. Current
TensorFold engines (1.1.x) publish the real families, and the `vllm-v1`
profile has never needed the probe; if an upstream stops publishing them, the
probe comes back automatically on the next poll. The decision is logged:
`tf-exporter: upstream latency histograms present -> synthetic probe disabled`.

## Metric mapping (zig-v1 engine)

| TensorFold source | Exported as | Notes |
|---|---|---|
| `tensorfold_prompt_tokens_total` | `vllm_prompt_tokens_total` | persistent |
| `tensorfold_completion_tokens_total` | `vllm_generation_tokens_total` | persistent |
| `tensorfold_requests_inflight` | `vllm_num_requests_running` | zig admits directly → waiting/swapped = 0 |
| `tensorfold_cached_tokens_total` | `vllm_prefix_cache_hits_total` (+ queries = prompt) | persistent; hit-rate = hits/queries |
| `tensorfold_decode_seconds_total` | `vllm_request_time_per_output_token_seconds_{sum,count}` | decode seconds / completion tokens → Decode Per-Req Avg |
| `tensorfold_prefill_seconds_total` | (estimator input) | windowed rate = (Δprompt − Δcached)/Δprefill_seconds — cached tokens add no prefill work |
| `tensorfold_requests_total` | TTFT `count` | real engine request counter → the Requests panel total |
| `tensorfold_uptime_seconds` | (restart detector) | regression → new counter era |
| `tensorfold_requests_stalled` / `tensorfold_engine_fatal` | `tfexporter_requests_stalled` / `tfexporter_engine_fatal` | self-status |
| `/health` drafted/accepted/rounds | `vllm_spec_decode_num_{draft,accepted}_tokens_total`, `…num_drafts_total` | acceptance rate + mean acceptance length |
| probe histograms | `vllm_time_to_first_token_seconds` et al. | buckets = probe-measured; TTFT sum/count = engine estimate |

Self-observability: `tfexporter_engine_schema_info`, `tfexporter_live_decode_tps`
(+ `_source`, `_derived`), `tfexporter_live_prefill_tps`,
`tfexporter_engine_prefill_lastjob_tps`, `tfexporter_prefilling`,
`tfexporter_queue_waiting`, `tfexporter_live_connections`,
`tfexporter_engine_device_memory_bytes{kind}`,
`tfexporter_engine_process_resident_bytes{kind}`,
`tfexporter_client_disconnections_total`, `tfexporter_generation_tokens_running`,
`tfexporter_engine_warming`, `tfexporter_engine_lanes`, `tfexporter_engine_info`,
`tfexporter_cache_hit_ratio` (600 s window, only when the engine counts cached tokens),
`tfexporter_mem_available_gib`, `tfexporter_engine_restarts_total`,
`tfexporter_upstream_ok`, `tfexporter_upstream_lag_seconds`.

**Graceful degradation:** after an upstream failure the last good snapshot is
served for 10 seconds, then the exporter answers `502`. `/healthz` reflects
upstream liveness.

## Metric mapping (velo-v1 engine — veloGB10)

veloGB10 exposes `velogb10_*` families only and no `/health` `live` block, so
the adapter synthesizes the vLLM-named series the dashboard expects. Every
synthesis is documented in its own HELP text; the provenance in one place:

| Emitted | Source | Honest caveat |
|---|---|---|
| `vllm_prefix_cache_hits_total` | `velogb10_prompt_tokens_cached_total` | direct rename |
| `vllm_prefix_cache_queries_total` | `velogb10_prompt_tokens_total` | zig convention: every prompt token is a query |
| `vllm_e2e_request_latency_seconds*` | `velogb10_request_duration_seconds*` | direct rename, buckets passed through |
| `vllm_spec_decode_num_draft/accepted_tokens_total` | `velogb10_spec_draft/accepted_tokens_total` | direct rename |
| `vllm_request_time_per_output_token_seconds_sum/_count` | `request_duration_seconds_sum − time_to_first_token_seconds_sum`, completion tokens | engine has no decode-seconds counter; both totals share the finished-request population (cross-checked vs client measurement: 0.4%) |
| `vllm_inter_token_latency_seconds_sum/_count` | same pair as above | vLLM-named alias; per-request, not per-token samples |
| `vllm_*_latency_seconds_bucket` (ITL/TPOT, SLO goodput) | `velogb10_decode_tokens_per_second_bucket` | exact reciprocal TPOT=1/tps of a per-request measurement — a change of variable, not a modeled distribution; bucket-edge requests count as SLO misses |
| `vllm_iteration_tokens_total_sum/_count` (Batch tile) | `generation_tokens_total / scheduler_steps_total` | tokens per decode step; with MTP the effective step width |
| `vllm_request_queue_time_seconds_sum/_count` (Queue tile) | time-integral of `velogb10_requests_waiting` ÷ completions | **Little's law** over the exporter's 2s poll timeline; the engine publishes no queue-seconds counter; window opens at exporter attach |
| `vllm_kv_cache_usage_perc` | `kv_positions_used / capacity` | **ratio 0..1** — the dashboard multiplies by 100 |
| `tfexporter_mem_available_gib` | `velogb10_mem_available_bytes` | engine host's reading, correct for remote exporters |

Functional gap kept honest: veloGB10 never emits per-request queue seconds,
so Queue is a poll-sampled estimate (documented in HELP), not an engine
measurement. The engine counts decode rounds as
`velogb10_spec_rounds_total`; it is surfaced natively.

## Metric mapping (flare-v1 engine — Flare / Hyperflare)

Flare (Rust, `flare_*` families only, no `/health` block beyond
`{"status":"ok"}`) shares no family-name suffix with the vLLM namespace, so
the whole dashboard set is synthesized in the `engine_native_series`
flare branch. Sources verified against a live scrape and the engine's own
`docs/api.md` serving-terminal table (2026-11-12 build):

| Emitted | Source | Honest caveat |
|---|---|---|
| `vllm_prompt_tokens_total` | `flare_prefill_tokens_total + flare_cache_reused_tokens_total` | vLLM semantics: prompt totals include prefix-reused tokens |
| `vllm_generation_tokens_total` | `flare_generated_tokens_total` | direct rename |
| `vllm_num_requests_running` / `_waiting` | `flare_active_requests` / `flare_pending_requests` | direct rename |
| `vllm_num_requests_swapped` | constant 0 | documented zero: Flare counts preemptions, not host-KV swap-outs (`tfexporter_flare_preemptions_total`) |
| `vllm_kv_cache_usage_perc` | `(total_kv_tokens − free_kv_tokens) / total_kv_tokens` | **ratio 0..1** — the dashboard multiplies by 100 |
| `vllm_prefix_cache_hits_total` / `_queries_total` | `flare_cache_reused_tokens_total` / prompt total | token units, same convention as velo-v1 |
| `vllm_time_to_first_token_seconds_sum`/`_count` | `flare_first_response_median_seconds × samples` / `samples` | **gauge-shaped**: the engine publishes a 600 s rolling median, not a cumulative histogram; sum/count = current window median, `tfexporter_flare_first_response_truncated` marks window-edge truncation |
| `vllm_request_time_per_output_token_seconds*`, `vllm_inter_token_latency_seconds*` | time-integral of `flare_active_requests` dt ÷ completion tokens | **occupancy-derived**: Flare has no per-request seconds anywhere; sum/count = mean TPOT across active requests (state on client totals, not fabricated samples) |
| `vllm_request_queue_time_seconds_sum`/`_count` | time-integral of `flare_pending_requests` dt ÷ window completions | **Little's law** over the exporter's 2 s poll timeline (same discipline as velo-v1 in 1.5.5) |
| `vllm_iteration_tokens_total_sum`/`_count` (Batch tile) | completion tokens ÷ `flare_scheduler_executions_total` (reason labels summed) | tokens per physical step (docs/api.md batch table) |
| `vllm_spec_decode_num_draft_tokens_total` / `_accepted_` | `flare_speculative_offered/accepted_drafts_total` | direct rename |
| `tfexporter_flare_verification_depth` | `flare_verification_draft_tokens_total / flare_verification_request_rounds_total` | docs/api.md DEPTH definition |
| `tfexporter_flare_spec_accept_ratio` | accepted/offered drafts | lifetime ratio |
| `tfexporter_flare_generated_tokens_last_minute` | native engine gauge | engine's own live decode rate |
| `tfexporter_flare_free_request_slots` | `flare_free_request_slots` | direct |

Functional gaps kept honest: Flare publishes **no** per-token or per-request
latency histograms, so the SLO Goodput ITL/TPOT bucket tiles and
`vllm_e2e_request_latency_seconds*` stay empty — unlike velo-v1 (1.5.5) there
is no per-request measurement to transform, and inventing a modeled
distribution would be worse than a blank tile.

## Metric mapping (tandem-v1 engine — TandemLLM)

TandemLLM (`qwen38-spark-engine`) publishes a real Prometheus `/metrics` in the
`qse_*` namespace: genuine per-request histograms for TTFT / E2E / TPOT / queue,
real spec-decode counters, its own prompt accounting split. Detection marker:
`qse_requests_running`. Unlike flare-v1 nothing is modeled here — the dashboard
set is a rename of measured engine series, plus two /health-derived pieces and
two architectural zeros:

| dashboard family | source | kind |
| --- | --- | --- |
| `vllm_time_to_first_token_seconds_{sum,count}` | `qse_time_to_first_token_seconds_*` | renamed, measured |
| `vllm_e2e_request_latency_seconds_{sum,count}` | `qse_e2e_request_latency_seconds_*` | renamed, measured |
| `vllm_request_time_per_output_token_seconds_*`, `vllm_time_per_output_token_seconds_*`, `vllm_inter_token_latency_seconds_*` | `qse_time_per_output_token_seconds_*` | one measured population aliased to the three tile names |
| `vllm_request_queue_time_seconds_{sum,count}` | `qse_request_queue_seconds_*` | renamed, measured |
| `vllm_prompt_tokens_total` | `qse_prompt_tokens_total − qse_prompt_tokens_cached_total` | FRESH only (forwarded counter carries chunk padding, see CHANGELOG 1.7.0) |
| `vllm_prefix_cache_queries_total` / `hits_total` | `qse_prompt_tokens_total` / `qse_prompt_tokens_cached_total` (clamped) | identity queries − prompt = hits exact |
| `vllm_generation_tokens_total` | max(`qse_generation_tokens_total`, Σ per-slot kv growth of decoding lanes) | live hybrid: engine counts only on finish; phases decode\|runner\|solo, per-slot seeding |
| `vllm_kv_cache_usage_perc` / `gpu_cache_usage_perc` | /health `parallel.slots`: Σ busy kv_length ÷ (QSE_PARALLEL × max_len) | 0..1 ratio |
| `vllm_spec_decode_num_*` | `qse_spec_decode_num_*` | renamed, measured |
| `vllm_iteration_tokens_total_{sum,count}` | generation population / `qse_verify_seconds_count` | tokens per verify step for the Batch tile |
| `vllm_num_requests_swapped`, `vllm_num_preemptions_total` | — | documented zeros: no batching scheduler exists (server/metrics.py) |
| `tfexporter_tandem_spec_accept_ratio` / `_first_draft_accept_pct` | accepted ÷ draft tokens | derived gauge |
| `tfexporter_tandem_prefill_tps_mean` / `_decode_tps_mean` | engine histogram sum ÷ count | mean of measured per-request rates |
| `tfexporter_tandem_live_decode_tps` | kv_length growth window | live gauge (flickers with load, like all live tiles) |
| `tfexporter_tandem_reasoning_tokens_total` | `qse_reasoning_tokens_total` | renamed counter |

Raw `qse_*` families are served unchanged beside the `vllm_*` set (verified
zero name collisions), so Grafana can read the engine's own detail — the full
cache cascade (resident / state store / disk tier), draft-width choice, accept
per speculation depth. Synthetic probes are disabled for this schema.

## Live rates: engine first, ours as fallback

The engine publishes a `live` object in `/health`:

```json
{"status": "ok", "live": {"connections": 2, "waiting": 1,
 "decode_tokens_per_second": 72.0, "prefill_tokens_per_second": 1458.2}}
```

| Series | Source | Note |
|--------|--------|------|
| `tfexporter_live_decode_tps` | `live.decode_tokens_per_second` while fresh, else our window | the engine's window is 2 s; a fresh `0` is honest idle, not a gap to paper over |
| `tfexporter_live_decode_tps_source` | 1 engine / 0 derived | read this before trusting a disagreement |
| `tfexporter_live_decode_tps_derived` | our window, always | the cross-check |
| `tfexporter_live_prefill_tps` | our window only | `zig-v1`: `delta(prompt-cached)/delta(prefill_seconds)`, compute only; `python-v1`/health: TTFT-based, queue included |
| `tfexporter_engine_prefill_lastjob_tps` | `live.prefill_tokens_per_second` | **informational**: one job's figure, timed from admission, last-writer-wins |
| `tfexporter_prefilling` | same field > 0 | coarse "a prompt pass just happened" |
| `tfexporter_queue_waiting`, `tfexporter_live_connections` | same `/health` snapshot | consistent with the decode tile |

## Native engine counters the vLLM namespace has no name for

The engine counts more than the vLLM family set can express. These are 1:1 renames of
families the engine publishes - not estimates, and never zero-filled:

| Series | Engine family | Why it matters |
|--------|---------------|----------------|
| `vllm_spec_decode_num_drafts_total` | `tensorfold:decode_rounds_total` | draft/acceptance panels were blank; the old source was a `/health` field this engine never had |
| `tfexporter_engine_device_memory_bytes{kind="active\|peak\|pinned"}` | `tensorfold:device_memory_bytes`, `…_peak_bytes`, `tensorfold:pinned_memory_bytes` | allocator footprint; `peak` resets only on `/health?reset_peak=1`, which this exporter never calls |
| `tfexporter_engine_process_resident_bytes{kind="rss\|anon\|file\|shmem\|peak"}` | `tensorfold:process_resident_*_bytes` | separates the anon/shmem split that decides whether the memory floor is real |
| `tfexporter_client_disconnections_total` | `tensorfold:client_disconnections_total` | streams the engine dropped mid-reply |
| `tfexporter_generation_tokens_running` | `tensorfold:generation_tokens_running` | tokens from requests still in flight |
| `tfexporter_engine_warming` | `/health` `warming` | 1 while graphs/pools warm |
| `tfexporter_engine_lanes` | `/health` `max_batch_size` | concurrency cap; `vllm_num_requests_running` stuck here = saturated, not slow |
| `tfexporter_engine_info{version,backend,model}` | `/health` `version`/`backend`/`model` | fleet inventory without ssh |

Anything the engine does not count stays unexported. On the flash17 pin H engine that
includes every prefix-cache figure - it publishes no cached-tokens counter under any
name - so `tfexporter_cache_hit_ratio`, `tfexporter_cached_tokens_total` and
`vllm_prefix_cache_hits_total` are absent there. `vllm_num_requests_swapped` is emitted
as a documented zero (the engine preempts rather than swaps, and does count it in
`tensorfold:preemptions_total`, renamed to `vllm_num_preemptions_total`);
`vllm_request_queue_time_seconds_*` is not emitted at all on this schema, because no
queue-time source exists. Reading `--` on those panels means "no source", not
"no activity".

Two consequences worth knowing before you graph them. First, the exporter
applies no smoothing of its own on top of the engine's window - stacking a
filter on a 2 s window only doubles the lag - but it can still trail the engine
by up to one `TF_POLL_INTERVAL` because it publishes the last snapshot it
fetched. Second, the engine's prefill figure is not a rate: measured on a live
TP2 pair it read 197/603/1458 tok/s while that engine served a 49K-token prompt
at ~1.9K tok/s, because it is written once per finished pass and includes the
time spent waiting for a lane. Our derived prefill rate is an aggregate over
the window, so with concurrent requests and prefix-cache hits it can exceed any
single request's rate; for honest per-request latency distributions use the
engine's own histograms instead - `vllm_time_to_first_token_seconds` (time to
first token, queue included on this schema) and
`vllm_request_time_per_output_token_seconds`.

## Requirements

- Python 3.8+ — **stdlib only, zero dependencies**.
- A TensorFold server exposing `GET /metrics` (zig flash17 build or python glm53).
- Optional: Spark Dashboard 0.14+ as the consumer.

## Quick start

```bash
TF_UPSTREAM=http://localhost:8002 python3 exporter.py
# exporter listening on 0.0.0.0:8004

curl -s localhost:8004/metrics | grep -E '^vllm_' | head
```

### Configuration (environment variables)

| Variable            | Default                 | Purpose                                |
|---------------------|-------------------------|----------------------------------------|
| `TF_UPSTREAM`       | `http://localhost:8002` | Base URL of the TensorFold server      |
| `TF_PORT`           | `8004`                  | Port for the exporter to listen on     |
| `TF_BIND`           | `0.0.0.0`               | Bind address                           |
| `TF_POLL_INTERVAL`  | `2`                     | Upstream poll interval, seconds        |
| `TF_LIVE_FRESH_S`   | `5`                     | How long an engine `/health` live reading is trusted before falling back |

### Run as a systemd service

```bash
./install.sh head     # on the head node: upstream = localhost:8002
./install.sh worker   # on a worker node: upstream = <head-fabric-ip>:8002
```

`install.sh` writes `tf-exporter.service` into `/etc/systemd/system/` (needs
`sudo`) and enables it. Adjust `WorkingDirectory`/`ExecStart` for your layout.

## Spark Dashboard wiring

```ini
# /etc/spark-dashboard/config.env
SPARK_DASHBOARD_ENGINE=vllm
SPARK_DASHBOARD_ENGINE_URL=http://localhost:8004
```

then `sudo systemctl restart spark-dashboard`.

## Tests

50 unit tests cover schema detection (TensorFold zig/python **and** native
vLLM), the vllm-v1 colon → underscore passthrough (label preservation,
metadata ordering, duplicate suppression, honest swapped zero), windowed rate
math (cache-token
exclusion, busy/idle hold, engine-era reset), the persistence round-trip and
offset continuation, the bridge contract (series presence, honest omissions,
sample uniqueness), histogram bucketing + SLO bucket edges, and a deadlock
regression guard:

```bash
python3 -m unittest tests.test_exporter_metrics -v   # unit tests
python3 tests/test_rewrite.py                        # python-v1 rewrite smoke
```

CI runs both on Python 3.8 and 3.12 (`.github/workflows/ci.yml`).

## Known limitations

- KV-cache percent and queue time have no honest TensorFold source — panels
  show `--` (queue counters are emitted as documented zeros).
- ITL/TPOT distributions are probe-based **only on engines that expose no
  latency histograms**: they then reflect the probe's requests under current
  load, not the fleet-wide distribution. On engines that publish
  `time_to_first_token_seconds` themselves (TensorFold 1.1.x, native vLLM) the
  probe is disabled and the panels read the engine's own all-traffic
  histograms instead.
- Prefill *Per-Req Avg* follows the current load regime (600 s window, held
  through idle phases); after an engine restart it needs one prefill-active
  window to produce a fresh rate (cumulative fallback until then).
- The live decode tile trails the engine by up to one poll interval, and it is
  an instantaneous window rather than a smoothed series - expect it to hit 0
  during long prefills, which is what "nothing decoded in the last 2 s" means.
- In the `vllm-v1` profile the probe subsystem is **off** (the engine measures
  its own latencies) and counters are served straight from the engine: an
  engine restart resets them exactly as a plain vLLM scrape would, with no
  offset continuation.

## Contributing

Issues and PRs are welcome. The project is intentionally dependency-free;
please keep it that way. Bug reports with a sample of the upstream `/metrics`
exposition (sensitive labels removed) are especially helpful.

## License

[MIT](LICENSE) © 2026 goshi
