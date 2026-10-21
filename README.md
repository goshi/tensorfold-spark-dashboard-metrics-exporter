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
own streaming probe (TTFT / E2E / ITL / TPOT — feeding the SLO goodput tiles),
and proxies every other GET path (`/v1/models`, `/health`) to the upstream
unchanged.

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

- **`zig-v1`** — underscore families (`tensorfold_prompt_tokens_total{model=…}`),
  the flash17 Zig engine. Primary target: raw families + `/health` JSON
  (in-flight, spec-decode counters, streams).
- **`python-v1`** — colon families (`tensorfold:prompt_tokens_total`), the
  Python glm53 engine. Legacy path: family rename + health-counter synthesis.
- **`unknown`** — nothing rewritten; the exposition passes through untouched.

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

Self-observability: `tfexporter_engine_schema_info`, `tfexporter_live_decode_tps`,
`tfexporter_live_prefill_tps`, `tfexporter_cache_hit_ratio` (600 s window),
`tfexporter_mem_available_gib`, `tfexporter_engine_restarts_total`,
`tfexporter_upstream_ok`, `tfexporter_upstream_lag_seconds`.

**Graceful degradation:** after an upstream failure the last good snapshot is
served for 10 seconds, then the exporter answers `502`. `/healthz` reflects
upstream liveness.

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

30 unit tests cover schema detection, windowed rate math (cache-token
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
- ITL/TPOT distributions are probe-based: they reflect the probe's requests
  under current load, not the fleet-wide distribution (the engine exposes
  none).
- Prefill *Per-Req Avg* follows the current load regime (600 s window, held
  through idle phases); after an engine restart it needs one prefill-active
  window to produce a fresh rate (cumulative fallback until then).

## Contributing

Issues and PRs are welcome. The project is intentionally dependency-free;
please keep it that way. Bug reports with a sample of the upstream `/metrics`
exposition (sensitive labels removed) are especially helpful.

## License

[MIT](LICENSE) © 2026 goshi
