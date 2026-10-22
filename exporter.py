#!/usr/bin/env python3
"""TensorFold -> vLLM Prometheus metrics bridge for Spark Dashboard.

Spark Dashboard / Grafana scrape `vllm_*` metric families; TensorFold serves
its own families under the `tensorfold:` prefix. This proxy polls the upstream
TensorFold /metrics endpoint, rewrites the families into vLLM-named
equivalents and serves the result on its own port. Any other GET path is
proxied to the upstream unchanged.

Supports THREE upstream engine profiles (auto-detected per poll):

  * "health" profile (older TensorFold, e.g. the GLM pair): exposes
    `tensorfold_health:*` counters. The live completion counter feeds
    vllm_generation_tokens_total directly; aggregate TPOT comes from
    decode_seconds_total / completion_tokens_total.

  * "native" profile (TensorFold serving DeepSeek V4.1): ALL token/time
    counters are finished-request-only and move in completion bursts. v3
    keeps them HONEST:
      - vllm_generation_tokens_total / vllm_prompt_tokens_total pass through
        the real finished-only counters (no synthetic inflation);
      - aggregate TPOT = request_decode_seconds_sum / generation_tokens_total
        (the engine's own documented decode-rate identity);
      - live rates are HELD ESTIMATES over a trailing window of finished
        deltas, exported only as tfexporter_live_decode_tps and
        tfexporter_live_prefill_tps gauges (0 when idle). They never feed
        counters, so dashboards can graph a smooth "now" line without
        corrupting totals. Point Live/throughput panels at the
        tfexporter_live_* gauges; keep rate() on the vllm_* counters for real
        token totals (completion bursts are real).

  * "vllm" profile (a native vLLM engine, e.g. the exlr8 GLM build): the
    upstream already serves REAL `vllm:*` families (TTFT/E2E/ITL/TPOT
    histograms, KV-cache usage, prefix-cache and spec-decode counters).
    Nothing is synthesized: the colon namespace is renamed 1:1 to the
    underscore names Spark Dashboard's VllmAdapter matches
    (`vllm:prompt_tokens_total` -> `vllm_prompt_tokens_total`), labels are
    preserved and every non-vLLM family is dropped. The probe subsystem is
    disabled here - the engine measures its own latencies, so probing would
    only add load.

v3 fixes vs v2 (2026-10-19, all observed live on spark3):
  - v2 credited the held burst rate into a synthetic generation counter AND
    the finished-token diff at completion (double count: 376k synthetic vs
    167k real finished tokens), and kept crediting the held burst during
    prefill-only phases where zero tokens were generated. Removed: counters
    are real now.
  - v2 TPOT estimate summed wall time while requests were in flight (prefill
    included) over synthetic tokens -> 9.7 ms shown vs 15.3 ms real, with
    spikes while the token estimate was frozen. TPOT now uses the engine's
    decode-histogram identity.
  - v2 "live decode tps" was the LAST COMPLETION BURST held indefinitely
    (127.8 t/s shown vs 65.3 t/s real). Now: trailing-window true rate.
  - Histogram families (TTFT, e2e) lost their # HELP/# TYPE lines in v2
    (metadata landed under the bare family key, samples under _bucket/_sum/
    _count keys -> untyped metrics -> histogram_quantile panels dead).
    Metadata is re-attached at output time.
  - request_decode_seconds is no longer discarded in the native profile; it
    is the honest TPOT / decode-rate source.

Configuration (environment variables):
  TF_UPSTREAM       base URL of the TensorFold server (default http://localhost:8002)
  TF_PORT           port for this exporter to listen on   (default 8004)
  TF_BIND           bind address                          (default 0.0.0.0)
  TF_POLL_INTERVAL  upstream poll interval in seconds     (default 2)

Python 3 stdlib only - no external dependencies.
"""

import json
import os
import re
import threading
import time
import uuid
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = os.environ.get("TF_UPSTREAM", "http://localhost:8002").rstrip("/")
PORT = int(os.environ.get("TF_PORT", "8004"))
BIND = os.environ.get("TF_BIND", "0.0.0.0")
POLL = float(os.environ.get("TF_POLL_INTERVAL", "2"))

SAMPLE_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{.*\})?\s+(\S+)(?:\s+\S+)?$")
META_RE = re.compile(r"^# (HELP|TYPE)\s+(\S+)(?:\s(.*))?$")
PREFIXES = ("tensorfold_health:", "tensorfold:")
# Native vLLM exposition namespace (colon-separated). Spark Dashboard's
# VllmAdapter matches UNDERSCORE names, so these are renamed 1:1.
VLLM_PREFIX = "vllm:"
VERSION = "1.1.0"

# Direct base-name renames (after stripping the tensorfold: prefix).
# The "requests_running"/"requests_waiting" mirrors map to the same vLLM
# names as their num_requests_* twins; duplicate samples are deduped.
DIRECT = {
    "num_requests_running": "vllm_num_requests_running",
    "num_requests_waiting": "vllm_num_requests_waiting",
    "requests_running": "vllm_num_requests_running",
    "requests_waiting": "vllm_num_requests_waiting",
    "preemptions_total": "vllm_num_preemptions_total",
    "prompt_tokens_total": "vllm_prompt_tokens_total",
    "generation_tokens_total": "vllm_generation_tokens_total",
    "time_to_first_token_seconds": "vllm_time_to_first_token_seconds",
    "e2e_request_latency_seconds": "vllm_e2e_request_latency_seconds",
    "spec_decode_num_draft_tokens_total": "vllm_spec_decode_num_draft_tokens_total",
    "spec_decode_num_accepted_tokens_total": "vllm_spec_decode_num_accepted_tokens_total",
}

HISTOGRAM_SUFFIXES = ("_bucket", "_sum", "_count")

# Base names that must NOT be passed through:
#  - kv cache gauges carry per-stream labels; they are aggregated separately
#  - the decode-time histograms have per-request (not per-token) semantics;
#    in the native profile the _sum is captured separately as the honest TPOT
#    numerator instead of being synthesized.
SKIP_BASES = {
    "kv_cache_usage_ratio",
    "kv_cache_usage_perc",
    "request_decode_seconds",
    "request_decode_time_seconds",
}

# Native profile: trailing window for the held live-rate estimates.
RATE_WINDOW_S = 600.0   # 10 minutes of finished-counter history
IDLE_POLLS = 3          # consecutive idle polls before held rates decay to 0


def split_base_suffix(name):
    for suf in HISTOGRAM_SUFFIXES:
        if name.endswith(suf):
            return name[: -len(suf)], suf
    return name, ""


def clean(name):
    for p in PREFIXES:
        if name.startswith(p):
            return name[len(p):]
    return name


def _fnum(line):
    """Last whitespace-separated token of a sample line as float, or None."""
    try:
        return float(line.rsplit(" ", 1)[1])
    except (IndexError, ValueError):
        return None


class State:
    """Shared snapshot plus live-rate bookkeeping for both engine profiles."""

    def __init__(self):
        self.lock = threading.Lock()
        self.body = b""
        self.upstream_ok = False
        self.last_ok_ts = 0.0
        self.last_error = ""
        # engine profile: "health" (old) or "native" (no tensorfold_health:*)
        self.profile = "health"
        # live-rate tracking, health profile (live completion counter diffs)
        self.prev_tok = None
        self.prev_ts = None
        self.live_tps = 0.0
        # native profile: trailing window of finished counters, held rates
        self.window = []        # (ts, gen_fin, dec_s, prompt_fin, ttft_s)
        self.rate_decode = 0.0  # held true decode tps (delta gen / delta dec_s)
        self.rate_prefill = 0.0 # held prefill tps estimate (delta prompt / delta ttft)
        self.rate_prefill_hold = None  # last nonzero windowed prefill rate
        self.busy_polls = 0     # consecutive polls with running > 0
        # v4.1: detected upstream metric schema
        self.schema = "unknown"
        # v4: zig-native extras
        self.prev_uptime = None
        self.restarts = 0
        self.last_cached = None
        self.last_stalled = 0
        self.last_fatal = 0
        self.last_inflight = 0
        self.last_rejected = None
        self.cache_ratio = None
        self.prev_cached = None
        self.prev_prompt_r = None
        self.native_raw = b""
        self.health = None
        # exporter-measured latency sums: [sum, count] per family
        self.probe = {"ttft": [0.0, 0], "e2e": [0.0, 0], "itl": [0.0, 0]}
        self.probe_hist = {}
        self.offsets = {}
        self.mem_gib = None

    def set(self, body, ok, err=""):
        with self.lock:
            self.body = body
            self.upstream_ok = ok
            self.last_error = err
            if ok:
                self.last_ok_ts = time.time()


STATE = State()

# /v1/models metadata cache (native profile): context length for the panel.
MODELS_CACHE = {"ts": 0.0, "context_length": None, "model_id": None}


def fetch_upstream(path):
    req = urllib.request.Request(UPSTREAM + path, headers={"Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.status, resp.read(), resp.headers.get("Content-Type", "text/plain")


def update_models_cache():
    """Refresh tfexporter_context_length from /v1/models at most every 5 min."""
    now = time.time()
    if now - MODELS_CACHE["ts"] < 300:
        return
    try:
        _, body, _ = fetch_upstream("/v1/models")
        data = json.loads(body.decode("utf-8", "replace"))
        mlen = None
        mid = None
        for m in data.get("data", []):
            if mid is None and m.get("id"):
                mid = m.get("id")
            mm = m.get("max_model_len")
            if isinstance(mm, (int, float)):
                mlen = mm if mlen is None else max(mlen, mm)
        if mlen is not None:
            MODELS_CACHE["context_length"] = int(mlen)
            MODELS_CACHE["model_id"] = mid
        MODELS_CACHE["ts"] = now
    except (ValueError, KeyError, urllib.error.URLError, OSError):
        MODELS_CACHE["ts"] = now  # retry on the next window, do not hammer


SCHEMA_SOURCES = {
    "zig-v1": {
        "gen": ("tensorfold_completion_tokens_total",),
        "dec": ("tensorfold_decode_seconds_total",),
        "prompt": ("tensorfold_prompt_tokens_total",),
        "pf": ("tensorfold_prefill_seconds_total",),
        "run": ("tensorfold_requests_inflight",),
    },
    "python-v1": {
        "gen": ("tensorfold:generation_tokens_total",),
        "dec": ("tensorfold:request_decode_seconds_sum",),
        "prompt": ("tensorfold:prompt_tokens_total",),
        "pf": ("tensorfold:time_to_first_token_seconds_sum",),
        "run": ("tensorfold:requests_running",),
    },
    "vllm-v1": {
        "gen": ("vllm:generation_tokens_total",),
        "dec": ("vllm:request_decode_time_seconds_sum",),
        "prompt": ("vllm:prompt_tokens_total",),
        "pf": ("vllm:time_to_first_token_seconds_sum",),
        "run": ("vllm:num_requests_running",),
    },
}


def detect_schema(text):
    """Classify the upstream /metrics schema (engine auto-detection).

    vllm-v1:   native vLLM exposition (`vllm:prompt_tokens_total` with
               engine/model_name labels) - e.g. the exlr8 GLM build
    zig-v1:    underscore families with labels (the flash17 Zig engine)
    python-v1: colon families (tensorfold:*) from the Python engine
    unknown:   nothing recognised
    """
    if "vllm:prompt_tokens_total" in text or "vllm:generation_tokens_total" in text:
        return "vllm-v1"
    if "tensorfold_completion_tokens_total" in text or "tensorfold_uptime_seconds" in text:
        return "zig-v1"
    if "tensorfold:generation_tokens_total" in text:
        return "python-v1"
    return "unknown"


def update_native_stats(text, now):
    """Schema-aware collector: one trailing window of finished counters with
    per-schema rate sources. Rates are windowed delta ratios (tokens/seconds
    over the same finished set), held while requests run, 0 when idle.
    A schema switch or counter reset wipes the window (never diff across).
    """
    sources = SCHEMA_SOURCES.get(STATE.schema)
    if not sources:
        return
    vals = {}
    for line in text.splitlines():
        for slot, cands in sources.items():
            if slot not in vals:
                for cand in cands:
                    if line.startswith(cand):
                        vals[slot] = _fnum(line)
                        break
        if len(vals) == len(sources):
            break
    gen = vals.get("gen")
    if gen is None:
        return
    dec = vals.get("dec") or 0.0
    prompt = vals.get("prompt") or 0.0
    pf = vals.get("pf") or 0.0
    running = int(vals.get("run") or 0)
    if STATE.schema == "zig-v1":
        with STATE.lock:
            for line in text.splitlines():
                if line.startswith("tensorfold_cached_tokens_total"):
                    STATE.last_cached = _fnum(line)
                elif line.startswith("tensorfold_requests_stalled"):
                    STATE.last_stalled = int(_fnum(line) or 0)
                elif line.startswith("tensorfold_engine_fatal"):
                    STATE.last_fatal = int(_fnum(line) or 0)
                elif line.startswith("tensorfold_requests_rejected_total"):
                    STATE.last_rejected = _fnum(line)
                elif line.startswith("tensorfold_uptime_seconds"):
                    up = _fnum(line)
                    if STATE.prev_uptime is not None and up < STATE.prev_uptime:
                        STATE.restarts += 1
                        STATE.offsets = {}
                        STATE.probe_hist = {}
                        STATE.probe = {"ttft": [0.0, 0], "e2e": [0.0, 0], "itl": [0.0, 0]}
                        STATE.rate_prefill_hold = None  # new era: no rate history
                    STATE.prev_uptime = up
            STATE.last_inflight = running
    elif STATE.schema == "vllm-v1":
        # Native vLLM: cached prompt tokens are the engine's own prefix-cache
        # hits counter; in-flight requests are its running gauge.
        cached_v = None
        for line in text.splitlines():
            if line.startswith("vllm:prefix_cache_hits_total"):
                cached_v = _fnum(line)  # last match wins
        with STATE.lock:
            if cached_v is not None:
                STATE.last_cached = cached_v
            STATE.last_inflight = running
    with STATE.lock:
        if STATE.window:
            _, p_gen, p_dec, p_prompt, p_pf, p_cached = STATE.window[-1]
            if (gen < p_gen or dec < p_dec or prompt < p_prompt or pf < p_pf
                    or (p_cached is not None and (STATE.last_cached or 0) < p_cached)):
                STATE.window = []
                STATE.rate_decode = 0.0
                STATE.rate_prefill = 0.0
        cached_wv = (STATE.last_cached
                     if STATE.schema in ("zig-v1", "vllm-v1") else None)
        STATE.window.append((now, gen, dec, prompt, pf, cached_wv))
        cutoff = now - RATE_WINDOW_S
        while STATE.window and STATE.window[0][0] < cutoff:
            STATE.window.pop(0)
        if running > 0:
            STATE.busy_polls += 1
        else:
            STATE.busy_polls = 0
        if STATE.busy_polls == 0:
            STATE.rate_decode = 0.0
            STATE.rate_prefill = 0.0
        elif len(STATE.window) >= 2:
            old, new = STATE.window[0], STATE.window[-1]
            dt = new[0] - old[0]
            if dt > 0:
                d_gen, d_dec = new[1] - old[1], new[2] - old[2]
                d_prompt, d_pf = new[3] - old[3], new[4] - old[4]
                if d_dec > 0 and d_gen > 0:
                    STATE.rate_decode = d_gen / d_dec
                if d_pf > 0 and d_prompt > 0:
                    # cached prompts add no prefill work: exclude them,
                    # otherwise the rate explodes on cache-hit traffic
                    d_cached = (new[5] or 0) - (old[5] or 0)
                    STATE.rate_prefill = max(0.0, d_prompt - d_cached) / d_pf
                    if STATE.rate_prefill > 0:
                        STATE.rate_prefill_hold = STATE.rate_prefill
    if STATE.schema in ("zig-v1", "vllm-v1"):
        with STATE.lock:
            cached_now = STATE.last_cached
            if (cached_now is not None and STATE.prev_cached is not None
                    and prompt is not None and STATE.prev_prompt_r is not None):
                d_cached = cached_now - STATE.prev_cached
                d_prompt_r = prompt - STATE.prev_prompt_r
                if d_cached > 0 and d_prompt_r > 0:
                    STATE.cache_ratio = d_cached / d_prompt_r
            STATE.prev_cached = cached_now
            STATE.prev_prompt_r = prompt


def ttft_engine_estimate():
    """Engine-wide TTFT sum/count estimate. The dashboard computes
    Prefill Per-Req Avg as prompt_tokens_total / ttft_seconds_sum,
    so the ttft sum must cover ALL traffic: sum = prompt/rate where
    rate = (prompt - cached) / prefill_seconds (true average prefill
    throughput), count = engine requests_total. Histogram buckets
    remain exporter-probe measurements. Returns (sum, count) or None.
    """
    with STATE.lock:
        offs = {k: dict(v) for k, v in STATE.offsets.items()}
    def emitted(fam):
        off = offs.get(fam)
        return off["emitted"] if off else None
    prompt = emitted("vllm_prompt_tokens_total")
    cached = emitted("vllm_prefix_cache_hits_total")
    pf = emitted("tfexporter_prefill_seconds_total")
    reqs = emitted("tfexporter_requests_total")
    if None in (prompt, cached, pf, reqs) or pf <= 0.0 or reqs <= 0.0:
        return None
    with STATE.lock:
        hold = STATE.rate_prefill_hold
    # windowed rate tracks the current load regime; the cumulative
    # average mixes past eras and lags behind reality for hours
    rate = hold if hold and hold > 0 else max(1.0, (prompt - cached) / pf)
    return (prompt / rate, reqs)


def zig_vllm_bridge(text):
    """v3 contract restoration for the zig-v1 schema: emit the same vllm_*
    series the Python-engine path produced, sourced from zig family names.
    Families without an honest zig source (TTFT/E2E/ITL histograms, KV usage
    percent, spec-decode counters) are omitted rather than faked. Returns the
    Prometheus text bytes to append after rewrite()'s output."""
    vals = {}
    for line in text.splitlines():
        for name in ("tensorfold_prompt_tokens_total",
                     "tensorfold_completion_tokens_total",
                     "tensorfold_requests_inflight",
                     "tensorfold_cached_tokens_total",
                     "tensorfold_decode_seconds_total",
                     "tensorfold_prefill_seconds_total"):
            if name not in vals and line.startswith(name):
                vals[name] = _fnum(line)
    out = []

    def emit(name, value, kind, doc):
        out.append("# HELP " + name + " " + doc + NL10
                   + "# TYPE " + name + " " + kind + NL10
                   + name + " " + repr(float(value)) + NL10)

    NL10 = chr(10)
    pt_raw = vals.get("tensorfold_prompt_tokens_total") or 0.0
    ct_raw = vals.get("tensorfold_completion_tokens_total") or 0.0
    inflight = vals.get("tensorfold_requests_inflight") or 0.0
    cached_raw = vals.get("tensorfold_cached_tokens_total") or 0.0
    dec_raw = vals.get("tensorfold_decode_seconds_total") or 0.0
    pt = persisted_value("vllm_prompt_tokens_total", pt_raw)
    ct = persisted_value("vllm_generation_tokens_total", ct_raw)
    cached = persisted_value("vllm_prefix_cache_hits_total", cached_raw)
    dec_s = persisted_value("vllm_time_per_output_token_seconds_sum", dec_raw)
    persisted_value("vllm_prefix_cache_queries_total", pt_raw)
    persisted_value("vllm_time_per_output_token_seconds_count", ct_raw)
    persisted_value("tfexporter_prefill_seconds_total",
                 vals.get("tensorfold_prefill_seconds_total") or 0.0)
    reqs_raw = 0.0
    for line0 in text.splitlines():
        if line0.startswith("tensorfold_requests_total"):
            reqs_raw = _fnum(line0)  # last match wins
    persisted_value("tfexporter_requests_total", reqs_raw)
    emit("vllm_prompt_tokens_total", pt, "counter",
         "Prompt tokens processed (zig engine).")
    emit("vllm_generation_tokens_total", ct, "counter",
         "Generation tokens produced (zig engine completion counter).")
    emit("vllm_num_requests_running", inflight, "gauge",
         "Requests in flight.")
    emit("vllm_num_requests_waiting", 0.0, "gauge",
         "Queued requests (the zig engine admits directly).")
    emit("vllm_num_requests_swapped", 0.0, "gauge",
         "Swapped requests (not applicable to the zig engine).")
    emit("vllm_num_preemptions_total", 0.0, "counter",
         "Preemptions (not applicable to the zig engine).")
    # Engine-wide per-stream decode average: the adapter prefers this
    # family (request_time_per_output_token) over the probe histogram
    # for Decode Throughput Per-Req Avg, so it reflects all traffic,
    # not only probe requests under their instantaneous load.
    emit("vllm_request_time_per_output_token_seconds_sum", dec_s, "counter",
         "Total decode seconds across all traffic (engine-wide).")
    emit("vllm_request_time_per_output_token_seconds_count", ct, "counter",
         "Completion tokens across all traffic (per-stream TPS denominator).")
    emit("vllm_prefix_cache_hits_total", cached, "counter",
         "Prefix cache token hits.")
    emit("vllm_prefix_cache_queries_total", pt, "counter",
         "Prefix cache queries (prompt tokens served).")
    emit("vllm_request_queue_time_seconds_sum", 0.0, "counter",
         "Queue time (the zig engine admits directly).")
    emit("vllm_request_queue_time_seconds_count", 0.0, "counter",
         "Queued requests observed.")
    with STATE.lock:
        health = dict(STATE.health) if isinstance(STATE.health, dict) else {}
    if health:
        drafted = health.get("drafted_total")
        accepted = health.get("accepted_total")
        rounds = health.get("rounds_total")
        if rounds is not None:
            rounds = persisted_value("vllm_spec_decode_num_drafts_total", rounds)
            emit("vllm_spec_decode_num_drafts_total", rounds, "counter",
                 "Speculative draft rounds (zig /health).")
        if drafted is not None:
            drafted = persisted_value("vllm_spec_decode_num_draft_tokens_total", drafted)
            emit("vllm_spec_decode_num_draft_tokens_total", drafted, "counter",
                 "Speculative draft tokens (zig /health).")
        if accepted is not None:
            accepted = persisted_value("vllm_spec_decode_num_accepted_tokens_total", accepted)
            emit("vllm_spec_decode_num_accepted_tokens_total", accepted, "counter",
                 "Speculative accepted tokens (zig /health).")
    with STATE.lock:
        hists = {k: dict(v) for k, v in STATE.probe_hist.items()}
    fams = (
        ("vllm_time_to_first_token_seconds", "ttft",
         "TTFT histogram: buckets from exporter probes; sum/count estimated from engine counters (prompt/prefill-rate, requests) so prefill per-request math covers all traffic."),
        ("vllm_e2e_request_latency_seconds", "e2e",
         "End-to-end latency histogram, exporter-measured probes."),
        ("vllm_inter_token_latency_seconds", "itl",
         "Inter-token latency histogram per chunk interval, exporter-measured."),
        ("vllm_time_per_output_token_seconds", "tpot",
         "Per-output-token latency, estimated: probe chunk intervals over tokens per chunk; includes current engine load."),
    )
    for base, fam, doc in fams:
        h = hists.get(fam)
        if not h or h.get("+Inf", 0) <= 0:
            continue
        out.append("# HELP " + base + " " + doc + NL10
                   + "# TYPE " + base + " histogram" + NL10)
        for le in PROBE_BUCKETS:
            key = repr(le)
            if key in h:
                out.append(base + '_bucket{le="' + key + '"} '
                           + str(float(h[key])) + NL10)
        out.append(base + '_bucket{le="+Inf"} '
                   + str(float(h["+Inf"])) + NL10)
        s_val = float(h.get("sum", 0.0))
        c_val = float(h["+Inf"])
        if fam == "ttft":
            est = ttft_engine_estimate()
            if est is not None:
                s_val, c_val = est
        out.append(base + "_sum " + repr(s_val) + NL10)
        out.append(base + "_count " + str(c_val) + NL10)


    return "".join(out).encode()


def self_status_lines(profile_value):
    """Common tfexporter self-observability series for any engine profile."""
    with STATE.lock:
        ok_ts = STATE.last_ok_ts
        rate_decode = STATE.rate_decode
        rate_prefill = STATE.rate_prefill
    now = time.time()
    rows = (
        ("tfexporter_upstream_ok", "1", "gauge",
         "1 when the last upstream poll succeeded."),
        ("tfexporter_last_success_timestamp_seconds", str(int(ok_ts)), "gauge",
         "Unix time of the last successful upstream poll."),
        ("tfexporter_upstream_lag_seconds", f"{max(0.0, now - ok_ts):.1f}",
         "gauge", "Seconds since the last successful upstream poll."),
        ("tfexporter_engine_profile", str(profile_value), "gauge",
         "Detected upstream profile: 0 = TensorFold health, 1 = TensorFold "
         "native, 2 = native vLLM passthrough."),
        ("tfexporter_live_decode_tps", f"{rate_decode or 0.0:.1f}", "gauge",
         "Held decode rate (tokens/s): delta(generation tokens) / "
         "delta(decode seconds) over the trailing window; 0 when idle."),
        ("tfexporter_live_prefill_tps", f"{rate_prefill or 0.0:.1f}", "gauge",
         "Held prefill rate (tokens/s): delta(prompt tokens - cached tokens) "
         "/ delta(TTFT seconds) over the trailing window; 0 when idle."),
    )
    out = []
    for name, value, kind, doc in rows:
        out.append("# HELP " + name + " " + doc)
        out.append("# TYPE " + name + " " + kind)
        out.append(name + " " + value)
    return out


def vllm_rewrite(text):
    """Passthrough for a native vLLM upstream (schema vllm-v1).

    The engine already serves honest `vllm:*` families, so nothing is
    synthesized: the colon namespace is renamed 1:1 to the underscore names
    the dashboard adapter matches, labels (engine, model_name,
    finished_reason, le) are preserved verbatim, and non-vLLM families
    (http_*, process_*, python_*) are dropped. Only documented zeros are
    added for families this engine build does not export.
    """
    out = []
    seen = set()        # (mapped name, labels) - no duplicate samples
    sample_names = set()
    meta_seen = set()   # (kind, mapped name) - HELP/TYPE emitted once
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line:
            continue
        if line.startswith("#"):
            m = META_RE.match(line)
            if not m:
                continue
            kind, name, doc = m.group(1), m.group(2), (m.group(3) or "")
            if not name.startswith(VLLM_PREFIX):
                continue
            mapped = name.replace(":", "_")
            if (kind, mapped) in meta_seen:
                continue
            meta_seen.add((kind, mapped))
            out.append("# " + kind + " " + mapped
                       + ((" " + doc) if doc else ""))
            continue
        m = SAMPLE_RE.match(line)
        if not m:
            continue
        name = m.group(1)
        if not name.startswith(VLLM_PREFIX):
            continue
        mapped = name.replace(":", "_")
        labels, value = m.group(2) or "", m.group(3)
        key = (mapped, labels)
        if key in seen:
            continue
        seen.add(key)
        sample_names.add(mapped)
        out.append(mapped + labels + " " + value)

    # Families this vLLM build may not export: documented zero, so the
    # Requests panel sums a complete running/waiting/swapped set.
    if "vllm_num_requests_swapped" not in sample_names:
        out.append("# HELP vllm_num_requests_swapped Swapped requests "
                   "(not exported by this engine build; always 0).")
        out.append("# TYPE vllm_num_requests_swapped gauge")
        out.append("vllm_num_requests_swapped 0")

    out.extend(self_status_lines(2))
    return ("\n".join(out) + "\n").encode()


def rewrite(text):
    """Translate an upstream metrics exposition into vllm_* families."""
    if detect_schema(text) == "vllm-v1":
        # Native vLLM engine: nothing to translate, only the namespace rename.
        return vllm_rewrite(text)
    families = {}  # mapped family key -> [sample lines] (keys may carry _bucket/_sum/_count)
    meta = {}      # mapped BARE family name -> {"help": str|None, "type": str|None}
    kv_stream_max = None
    health = {}        # parsed counter values used for synthesis
    native_src = {}    # native histogram _sum values captured before skipping
    seen = set()
    has_health_family = False

    def fam(base):
        return families.setdefault(base, [])

    def set_meta(name, kind, doc):
        m = meta.setdefault(name, {"help": None, "type": None})
        if kind == "HELP" and m["help"] is None:
            m["help"] = doc
        elif kind == "TYPE" and m["type"] is None:
            m["type"] = doc

    def put_sample(base, labels, value):
        key = (base, labels)
        if key in seen:
            return
        seen.add(key)
        fam(base).append(f"{base}{labels} {value}")

    for raw in text.splitlines():
        line = raw.rstrip()
        if not line:
            continue
        if line.startswith("#"):
            if "tensorfold_health:" in line:
                has_health_family = True
            m = META_RE.match(line)
            if not m:
                continue
            kind, name, doc = m.group(1), clean(m.group(2)), (m.group(3) or "")
            base, _ = split_base_suffix(name)
            if base in SKIP_BASES:
                continue
            if base in DIRECT:
                # Store metadata under the BARE mapped name; at output time it
                # is re-attached to every _bucket/_sum/_count sample family.
                set_meta(DIRECT[base], kind, doc)
            continue

        m = SAMPLE_RE.match(line)
        if not m:
            continue
        base, suffix = split_base_suffix(clean(m.group(1)))
        labels, value = m.group(2) or "", m.group(3)

        if base in SKIP_BASES:
            if base in ("kv_cache_usage_ratio", "kv_cache_usage_perc"):
                try:
                    v = float(value)
                except ValueError:
                    continue
                kv_stream_max = v if kv_stream_max is None else max(kv_stream_max, v)
            elif base == "request_decode_seconds" and suffix == "_sum":
                # Honest decode-seconds total (native TPOT numerator).
                try:
                    native_src["dec_s"] = float(value)
                except ValueError:
                    pass
            continue

        # Health/native counters used for synthesized families.
        if base == "generation_tokens_total":
            try:
                health["generation_finished"] = float(value)
            except ValueError:
                pass
            # fall through to the DIRECT mapping below
        elif base == "completion_tokens_total":
            # LIVE token counter ("running replies' tokens so far included").
            # The dashboard's Live panel computes rate(vllm_generation_tokens_total);
            # the finished-only counter only moves on completions, so feed the
            # dashboard this live counter instead. Deterministically wins over
            # the finished-only mapping regardless of exposition line order.
            try:
                health["completion_tokens_total"] = float(value)
            except ValueError:
                pass
            key = ("vllm_generation_tokens_total", "")
            samples = fam("vllm_generation_tokens_total")
            if key in seen and samples:
                samples[0] = f"vllm_generation_tokens_total {value}"
            else:
                seen.add(key)
                samples.append(f"vllm_generation_tokens_total {value}")
            continue
        elif base == "cached_tokens_total":
            try:
                health["cached_tokens_total"] = float(value)
            except ValueError:
                pass
            continue
        elif base == "decode_seconds_total":
            # Health profile: aggregate decode seconds (TPOT numerator).
            try:
                health["decode_seconds_total"] = float(value)
            except ValueError:
                pass
            continue
        # Keep remaining health counters under tfexporter_ names.
        if base in ("requests_total", "rounds_total", "prefill_seconds_total",
                    "context_length", "streams_max", "pool_tokens",
                    "pool_free_tokens", "kept_prompts"):
            put_sample(f"tfexporter_{base}", labels, value)
            continue
        if base == "streams":
            put_sample("tfexporter_streams", labels, value)
            continue

        if base in DIRECT:
            put_sample(DIRECT[base] + suffix, labels, value)

    prompt_total = None
    pt_samples = families.get("vllm_prompt_tokens_total")
    if pt_samples:
        try:
            prompt_total = float(pt_samples[0].rsplit(" ", 1)[1])
        except (IndexError, ValueError):
            prompt_total = None

    def grab(tf_name):
        samples = families.get(tf_name)
        if not samples:
            return None
        try:
            return float(samples[0].rsplit(" ", 1)[1])
        except (IndexError, ValueError):
            return None

    pool_tokens = grab("tfexporter_pool_tokens")
    pool_free = grab("tfexporter_pool_free_tokens")

    # 0) Generation family metadata per profile.
    if not has_health_family:
        profile = "native"
        with STATE.lock:
            rate_decode = STATE.rate_decode
            rate_prefill = STATE.rate_prefill
        fin = health.get("generation_finished")
        set_meta("vllm_generation_tokens_total", "TYPE", "counter")
        set_meta("vllm_generation_tokens_total", "HELP",
                 "Generated tokens of finished requests (TensorFold native: "
                 "finished-only counter; live rate estimate: "
                 "tfexporter_live_decode_tps).")
        if fin is not None:
            set_meta("tfexporter_generation_finished_tokens_total", "TYPE", "counter")
            set_meta("tfexporter_generation_finished_tokens_total", "HELP",
                     "Mirror of the engine's finished-only generation counter.")
            put_sample("tfexporter_generation_finished_tokens_total", "",
                       str(int(fin)))
        # Real aggregate TPOT from the engine's decode-histogram identity:
        # request_decode_seconds_sum / generation_tokens_total.
        dec_s = native_src.get("dec_s")
        if dec_s is not None and fin is not None and fin > 0:
            for name in ("vllm_time_per_output_token_seconds",
                         "vllm_request_time_per_output_token_seconds"):
                set_meta(name, "TYPE", "histogram")
                set_meta(name, "HELP",
                         "Aggregate decode time per output token = "
                         "request_decode_seconds_sum / generation_tokens_total "
                         "(TensorFold native engine counters).")
                put_sample(name + "_sum", "", repr(dec_s))
                put_sample(name + "_count", "", str(int(fin)))
    else:
        profile = "health"
        rate_decode = None
        rate_prefill = None
        set_meta("vllm_generation_tokens_total", "TYPE", "counter")
        set_meta("vllm_generation_tokens_total", "HELP",
                 "Generated reply tokens, running replies' tokens so far "
                 "included (live; finished-only value: "
                 "tfexporter_generation_finished_tokens_total).")
        fin = health.get("generation_finished")
        if fin is not None:
            set_meta("tfexporter_generation_finished_tokens_total", "TYPE", "counter")
            set_meta("tfexporter_generation_finished_tokens_total", "HELP",
                     "Finished-only generation counter of the engine.")
            put_sample("tfexporter_generation_finished_tokens_total", "",
                       str(int(fin)))

    # 1) KV cache (variant B): the dashboard gauge tracks the busiest ACTIVE
    #    decode stream (0% when idle). The shared pool, which also holds kept
    #    prompts between requests, is exported under tfexporter_pool_usage_perc.
    if kv_stream_max is not None:
        for name in ("vllm_kv_cache_usage_perc", "vllm_gpu_cache_usage_perc"):
            set_meta(name, "TYPE", "gauge")
            set_meta(name, "HELP",
                     "Occupancy of the busiest active decode stream (0 when idle).")
            put_sample(name, "", repr(kv_stream_max))
    pool_usage = None
    if pool_tokens is not None and pool_free is not None and pool_tokens > 0:
        pool_usage = max(0.0, min(1.0, (pool_tokens - pool_free) / pool_tokens))
    if pool_usage is not None:
        set_meta("tfexporter_pool_usage_perc", "TYPE", "gauge")
        set_meta("tfexporter_pool_usage_perc", "HELP",
                 "Fraction of the shared KV pool held by streams and kept prompts.")
        put_sample("tfexporter_pool_usage_perc", "", repr(pool_usage))
    # Fallback: if TF stopped reporting per-stream gauges, show pool usage so
    # the dashboard gauge is never misleadingly pinned at 0.
    if kv_stream_max is None and pool_usage is not None:
        for name in ("vllm_kv_cache_usage_perc", "vllm_gpu_cache_usage_perc"):
            set_meta(name, "TYPE", "gauge")
            set_meta(name, "HELP",
                     "Fraction of the shared KV cache pool in use (fallback).")
            put_sample(name, "", repr(pool_usage))

    # 2) Aggregate TPOT from health counters (health profile only; the native
    #    profile synthesized its own from the decode histogram above).
    if profile == "health":
        dec = health.get("decode_seconds_total")
        comp = health.get("completion_tokens_total")
        if dec is not None and comp is not None and comp > 0:
            for name in ("vllm_time_per_output_token_seconds",
                         "vllm_request_time_per_output_token_seconds"):
                set_meta(name, "TYPE", "histogram")
                set_meta(name, "HELP",
                         "Aggregate decode time per output token (TensorFold "
                         "health counters).")
                put_sample(name + "_sum", "", repr(dec))
                put_sample(name + "_count", "", str(int(comp)))

    # 3) Prefix cache hit rate inputs (best-effort semantics, health profile).
    if profile == "health":
        cached = health.get("cached_tokens_total")
        if cached is not None:
            set_meta("vllm_prefix_cache_hits_total", "TYPE", "counter")
            set_meta("vllm_prefix_cache_hits_total", "HELP",
                     "Cached prompt tokens of finished requests.")
            put_sample("vllm_prefix_cache_hits_total", "", repr(cached))
        if prompt_total is not None:
            set_meta("vllm_prefix_cache_queries_total", "TYPE", "counter")
            set_meta("vllm_prefix_cache_queries_total", "HELP",
                     "Prompt tokens of finished requests (prefix cache queries analog).")
            put_sample("vllm_prefix_cache_queries_total", "", repr(prompt_total))

    # 4) MTP draft acceptance (native profile; real finished counters).
    if profile == "native":
        drafted = grab("vllm_spec_decode_num_draft_tokens_total")
        accepted = grab("vllm_spec_decode_num_accepted_tokens_total")
        if drafted is not None and accepted is not None:
            set_meta("tfexporter_mtp_drafted_total", "TYPE", "counter")
            set_meta("tfexporter_mtp_drafted_total", "HELP",
                     "Draft tokens verified on finished requests.")
            put_sample("tfexporter_mtp_drafted_total", "", str(int(drafted)))
            set_meta("tfexporter_mtp_accepted_total", "TYPE", "counter")
            set_meta("tfexporter_mtp_accepted_total", "HELP",
                     "Draft tokens kept on finished requests.")
            put_sample("tfexporter_mtp_accepted_total", "", str(int(accepted)))
            set_meta("tfexporter_mtp_acceptance_ratio", "TYPE", "gauge")
            set_meta("tfexporter_mtp_acceptance_ratio", "HELP",
                     "Fraction of MTP draft tokens accepted by the model.")
            ratio = (accepted / drafted) if drafted > 0 else 0.0
            put_sample("tfexporter_mtp_acceptance_ratio", "", f"{ratio:.4f}")
        set_meta("vllm_num_requests_swapped", "TYPE", "gauge")
        set_meta("vllm_num_requests_swapped", "HELP",
                 "Always 0: the native engine preempts instead of swapping.")
        put_sample("vllm_num_requests_swapped", "", "0")
        mlen = MODELS_CACHE.get("context_length")
        if mlen is not None:
            set_meta("tfexporter_context_length", "TYPE", "gauge")
            set_meta("tfexporter_context_length", "HELP",
                     "Engine max_model_len from /v1/models.")
            put_sample("tfexporter_context_length", "", str(mlen))

    # 5) Exporter self-status gauges.
    set_meta("tfexporter_upstream_ok", "TYPE", "gauge")
    put_sample("tfexporter_upstream_ok", "", "1")
    with STATE.lock:
        ok_ts = STATE.last_ok_ts
    set_meta("tfexporter_last_success_timestamp_seconds", "TYPE", "gauge")
    put_sample("tfexporter_last_success_timestamp_seconds", "",
               str(int(ok_ts)))
    set_meta("tfexporter_upstream_lag_seconds", "TYPE", "gauge")
    set_meta("tfexporter_upstream_lag_seconds", "HELP",
             "Seconds since the last successful upstream poll.")
    put_sample("tfexporter_upstream_lag_seconds", "",
               f"{max(0.0, time.time() - ok_ts):.1f}")
    set_meta("tfexporter_engine_profile", "TYPE", "gauge")
    set_meta("tfexporter_engine_profile", "HELP",
             "Detected upstream profile: 0 = health (old), 1 = native (no tensorfold_health).")
    put_sample("tfexporter_engine_profile", "", "1" if profile == "native" else "0")

    # 6) Live rate gauges (estimates only; never fed into counters).
    if profile == "native":
        set_meta("tfexporter_live_decode_tps", "TYPE", "gauge")
        set_meta("tfexporter_live_decode_tps", "HELP",
                 "Held true decode rate: delta(generation_tokens_total) / "
                 "delta(request_decode_seconds_sum) over the trailing "
                 f"{int(RATE_WINDOW_S)}s window of finished requests; held "
                 "while requests are in flight, 0 when idle.")
        put_sample("tfexporter_live_decode_tps", "", f"{rate_decode or 0.0:.1f}")
        set_meta("tfexporter_live_prefill_tps", "TYPE", "gauge")
        set_meta("tfexporter_live_prefill_tps", "HELP",
                 "Held prefill rate estimate: delta(prompt_tokens_total) / "
                 "delta(time_to_first_token_seconds_sum) over the trailing "
                 f"{int(RATE_WINDOW_S)}s window; includes queue time; held "
                 "while requests are in flight, 0 when idle.")
        put_sample("tfexporter_live_prefill_tps", "", f"{rate_prefill or 0.0:.1f}")
    else:
        with STATE.lock:
            live = STATE.live_tps
        set_meta("tfexporter_live_decode_tps", "TYPE", "gauge")
        set_meta("tfexporter_live_decode_tps", "HELP",
                 "Live decode rate (tokens/s) from the live completion counter "
                 "diff over the last poll interval; 0 when idle.")
        put_sample("tfexporter_live_decode_tps", "", f"{live:.1f}")

    out = []
    emitted_meta = set()
    for base, samples in families.items():
        if not samples:
            continue
        # Re-attach metadata to histogram sub-families: v2 lost # HELP/# TYPE
        # for _bucket/_sum/_count keys (registered under the bare name only),
        # which left TTFT/e2e untyped and broke histogram_quantile panels.
        # The Prometheus text format declares the type under the BARE family
        # name, emitted once before the first sub-family's samples.
        stripped = base
        for suf in HISTOGRAM_SUFFIXES:
            if stripped.endswith(suf):
                stripped = stripped[: -len(suf)]
                break
        m = meta.get(base)
        meta_name = base
        if m is None and stripped != base:
            m = meta.get(stripped)
            meta_name = stripped
        if m and meta_name not in emitted_meta:
            emitted_meta.add(meta_name)
            if m.get("help"):
                out.append(f"# HELP {meta_name} {m['help']}")
            if m.get("type"):
                out.append(f"# TYPE {meta_name} {m['type']}")
        out.extend(samples)
    text_out = "\n".join(out) + "\n"
    if "tensorfold_completion_tokens_total" in text:
        text_out = text_out.replace("vllm_num_requests_swapped 0.0" + chr(10), "")
        text_out = text_out.replace("vllm_num_requests_swapped 0" + chr(10), "")
        return text_out.encode() + zig_vllm_bridge(text)
    return text_out.encode()


def poll_loop():
    while True:
        try:
            _, body, _ = fetch_upstream("/metrics")
            text = body.decode("utf-8", "replace")
            now = time.time()
            native = "tensorfold_health:" not in text
            schema = detect_schema(text)
            with STATE.lock:
                STATE.profile = "native" if native else "health"
                if schema != STATE.schema:
                    STATE.schema = schema
                    STATE.window = []
                    STATE.rate_decode = 0.0
                    STATE.rate_prefill = 0.0
            if native:
                update_native_stats(text, now)
                update_models_cache()
                try:
                    with STATE.lock:
                        STATE.native_raw = (b"" if schema == "vllm-v1"
                                            else text.encode("utf-8", "replace"))
                except Exception:
                    pass
            try:
                _, hbody, _ = fetch_upstream("/health")
                STATE.health = json.loads(hbody.decode("utf-8", "replace"))
            except (urllib.error.URLError, OSError, ValueError):
                pass
            try:
                with open("/proc/meminfo") as f:
                    for ln in f:
                        if ln.startswith("MemAvailable:"):
                            STATE.mem_gib = float(ln.split()[1]) / (1024.0 * 1024.0)
                            break
            except OSError:
                pass
            else:
                # health profile: live rate from the live completion counter
                tok = None
                for line in text.splitlines():
                    if line.startswith("tensorfold_health:completion_tokens_total "):
                        tok = _fnum(line)
                        break
                live = None
                with STATE.lock:
                    if tok is not None and STATE.prev_tok is not None and STATE.prev_ts is not None:
                        dt = now - STATE.prev_ts
                        if dt > 0:
                            live = max(0.0, (tok - STATE.prev_tok) / dt)
                    STATE.prev_tok, STATE.prev_ts = tok, now
                if live is not None:
                    # smooth slightly: EMA with alpha 0.5 keeps the gauge readable
                    # while still reacting within a couple of polls
                    STATE.live_tps = (live if STATE.live_tps <= 0
                                      else 0.5 * STATE.live_tps + 0.5 * live)
            STATE.set(rewrite(text), True)
            persist_save()
        except (urllib.error.URLError, OSError, ValueError) as exc:
            STATE.set(b"", False, str(exc))
        time.sleep(POLL)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # keep the log quiet
        pass

    def _serve_metrics(self):
        with STATE.lock:
            ok, body, err, ts = (STATE.upstream_ok, STATE.body,
                                 STATE.last_error, STATE.last_ok_ts)
        if not ok:
            # Serve the last good snapshot for a short grace period (10s).
            if body and time.time() - ts < 10:
                self._send(200, body, "text/plain; version=0.0.4; charset=utf-8")
            else:
                self._send(502, f"upstream unavailable: {err}\n".encode(),
                           "text/plain; charset=utf-8")
            return
        extra = self._v4_gauges()
        self._send(200, body + extra, "text/plain; version=0.0.4; charset=utf-8")

    def _v4_gauges(self):
        NL = chr(10)
        with STATE.lock:
            # vllm-v1: the passthrough body already carries the renamed
            # families - never append the raw colon exposition.
            raw = b"" if STATE.schema == "vllm-v1" else STATE.native_raw
            vals = dict(rate_decode=STATE.rate_decode, rate_prefill=STATE.rate_prefill,
                        restarts=STATE.restarts, stalled=STATE.last_stalled,
                        fatal=STATE.last_fatal, inflight=STATE.last_inflight,
                        cached=((STATE.last_cached + _cached_offset_base())
                         if STATE.last_cached is not None else None),
                        rejected=STATE.last_rejected,
                        cache_ratio=STATE.cache_ratio, mem=STATE.mem_gib)
        out = []
        if raw:
            NL10 = chr(10)
            out.append(raw if raw.endswith(NL10.encode()) else raw + NL10.encode())
        model = MODELS_CACHE.get("model_id") or ""
        out.append(("# HELP tfexporter_engine_schema_info Detected upstream metric schema and model." + NL + "# TYPE tfexporter_engine_schema_info gauge" + NL).encode())
        out.append(('tfexporter_engine_schema_info{schema="' + STATE.schema + '",model="' + model + '"} 1' + NL).encode())
        lines = [
                                    ("tfexporter_engine_restarts_total", vals["restarts"], "Engine restarts detected via uptime regression."),
            ("tfexporter_requests_stalled", vals["stalled"], "Engine stalled requests gauge."),
            ("tfexporter_engine_fatal", vals["fatal"], "Engine fatal state gauge."),
            ("tfexporter_requests_inflight", vals["inflight"], "Requests in flight."),
            ("tfexporter_requests_rejected_total", vals["rejected"], "Admission refusals (pool pressure)." if vals["rejected"] is not None else None),
            ("tfexporter_cached_tokens_total", vals["cached"], "Prompt tokens resumed from cache." if vals["cached"] is not None else None),
            ("tfexporter_cache_hit_ratio", vals["cache_ratio"], "cached/prompt tokens ratio." if vals["cache_ratio"] is not None else None),
            ("tfexporter_mem_available_gib", vals["mem"], "Host MemAvailable GiB (the memory floor watch)." if vals["mem"] is not None else None),
        ]
        for name, v, doc in lines:
            if v is None:
                continue
            out.append(("# HELP " + name + " " + doc + NL + "# TYPE " + name + " gauge" + NL + name + " " + str(v) + NL).encode())
        return b"".join(out)

    def _proxy(self):
        try:
            status, body, ctype = fetch_upstream(self.path)
            self._send(status, body, ctype)
        except urllib.error.HTTPError as exc:
            self._send(exc.code, exc.read() if exc.fp else b"", exc.headers.get("Content-Type", "text/plain"))
        except (urllib.error.URLError, OSError) as exc:
            self._send(502, f"upstream unavailable: {exc}\n".encode(), "text/plain; charset=utf-8")

    def _send(self, status, body, ctype):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_GET(self):
        if self.path == "/metrics":
            self._serve_metrics()
        elif self.path in ("/healthz", "/health"):
            with STATE.lock:
                ok = STATE.upstream_ok
            if ok:
                self._send(200, b"ok\n", "text/plain; charset=utf-8")
            else:
                self._proxy()
        else:
            self._proxy()

    def do_POST(self):
        # Transparent proxy for the dashboard's live-rate probe
        # (POST /v1/chat/completions) and any client POSTs.
        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
            body = self.rfile.read(length) if length > 0 else b""
            req = urllib.request.Request(
                UPSTREAM + self.path, data=body,
                headers={"Content-Type": self.headers.get("Content-Type", "application/json"),
                         "Accept": self.headers.get("Accept", "*/*")})
            with urllib.request.urlopen(req, timeout=180) as resp:
                payload = resp.read()
                self._send(resp.status, payload,
                           resp.headers.get("Content-Type", "application/json"))
        except urllib.error.HTTPError as exc:
            try:
                payload = exc.read()
            except OSError:
                payload = b""
            self._send(exc.code, payload, "application/json")
        except (urllib.error.URLError, OSError, ValueError) as exc:
            self._send(502, str(exc).encode(), "text/plain; charset=utf-8")


    def do_HEAD(self):
        self.do_GET()


PROBE_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)


def _hist_add(fam, value):
    """Add one sample to a cumulative Prometheus-style histogram."""
    with STATE.lock:
        h = STATE.probe_hist.setdefault(fam, {})
        for le0 in PROBE_BUCKETS:
            h.setdefault(repr(le0), 0)
        h.setdefault("+Inf", 0)
        for le in PROBE_BUCKETS:
            if value <= le:
                key = repr(le)
                h[key] = h.get(key, 0) + 1
        h["+Inf"] = h.get("+Inf", 0) + 1
        h["sum"] = h.get("sum", 0.0) + value


def _probe_once():
    """One tiny streaming completion; returns (ttft_s, e2e_s, intervals, tokens)."""
    t0 = time.time()
    model = MODELS_CACHE.get("model_id") or "DeepSeek-V4.1-Flash-TF"
    # unique nonce per probe: a repeated prompt would be fully prefix-
    # cached, making TTFT measurements meaningless (~0 ms every time)
    # ~1.4 KB of unique filler: representative prefill load (a bare
    # 15-token prompt prefills in ~2 ms and measures nothing useful)
    nonce = (("ctx " + uuid.uuid4().hex + ". ") * 110
             + "Now say hello briefly, run " + uuid.uuid4().hex[:8] + ".")
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": nonce}],
        "max_tokens": 16, "temperature": 0.0, "stream": True,
        "stream_options": {"include_usage": True},
    }).encode("utf-8", "replace")
    req = urllib.request.Request(
        UPSTREAM + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
    stamps = []
    tokens = None
    with urllib.request.urlopen(req, timeout=30) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except ValueError:
                continue
            if isinstance(obj.get("usage"), dict):
                tokens = obj["usage"].get("completion_tokens", tokens)
            choices = obj.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                # the engine sends an empty {role} stub BEFORE prefill:
                # only stamp chunks carrying a real output token
                if (delta.get("reasoning_content") or delta.get("content")):
                    stamps.append(time.time() - t0)
    e2e = time.time() - t0
    if not stamps:
        return None, None, [], None
    ttft = stamps[0]
    intervals = [stamps[i + 1] - stamps[i] for i in range(len(stamps) - 1)]
    return ttft, e2e, intervals, tokens


def probe_enabled():
    """Probe only engines that expose no latency histograms of their own.

    A native vLLM upstream measures TTFT/E2E/ITL/TPOT itself, so probing it
    would add load and duplicate honest engine series.
    """
    with STATE.lock:
        return STATE.schema != "vllm-v1"


def probe_loop():
    """Measure latencies every 45s; keep rolling sums and bucket histograms."""
    while True:
        time.sleep(45)
        if not probe_enabled():
            continue
        try:
            ttft, e2e, intervals, tokens = _probe_once()
        except (urllib.error.URLError, OSError, ValueError):
            continue
        if ttft is None:
            continue
        with STATE.lock:
            health = dict(STATE.health) if isinstance(STATE.health, dict) else {}
        acc_len = 1.0
        if health.get("rounds_total") and health.get("accepted_total"):
            acc_len = max(1.0, health["accepted_total"] / health["rounds_total"])
        tok_per_chunk = float(tokens) / max(1, len(intervals) + 1) if tokens else acc_len
        _hist_add("ttft", ttft)
        _hist_add("e2e", e2e)
        for iv in intervals:
            _hist_add("itl", iv)
            _hist_add("tpot", iv / max(0.25, tok_per_chunk))
        with STATE.lock:
            STATE.probe["ttft"][0] += ttft
            STATE.probe["ttft"][1] += 1
            STATE.probe["e2e"][0] += e2e
            STATE.probe["e2e"][1] += 1
            STATE.probe["itl"][0] += max(0.0, e2e - ttft)
            STATE.probe["itl"][1] += max(1, len(intervals))


PERSIST_PATH = "/home/admin/llm/tensorfold-metrics-exporter/data/counters.json"


def persist_save():
    """Atomically dump offsets + probe histograms so a restart continues
    the counters instead of resetting them."""
    try:
        os.makedirs(os.path.dirname(PERSIST_PATH), exist_ok=True)
        with STATE.lock:
            blob = {"offsets": dict(STATE.offsets),
                    "probe_hist": {k: dict(v) for k, v in STATE.probe_hist.items()},
                    "probe": {k: list(v) for k, v in STATE.probe.items()}}
        tmp = PERSIST_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(blob, f)
        os.replace(tmp, PERSIST_PATH)
    except (OSError, ValueError, TypeError):
        pass


def persist_load():
    """Restore counters state saved by a previous exporter process."""
    try:
        with open(PERSIST_PATH) as f:
            blob = json.load(f)
        with STATE.lock:
            STATE.offsets = dict(blob.get("offsets") or {})
            STATE.probe_hist = dict(blob.get("probe_hist") or {})
            probe = blob.get("probe") or {}
            for k in ("ttft", "e2e", "itl"):
                if isinstance(probe.get(k), list) and len(probe[k]) == 2:
                    STATE.probe[k] = list(probe[k])
    except (OSError, ValueError, TypeError):
        pass


def persisted_value(fam, raw):
    """emitted = raw + offset; remember both for the next save."""
    with STATE.lock:
        off = STATE.offsets.get(fam)
        base = off["emitted"] - off["engine"] if off else 0.0
        value = raw + base
        STATE.offsets[fam] = {"engine": raw, "emitted": value}
    return value


def offset_base(fam):
    """Read-only offset lookup (for gauges that must not mutate the store)."""
    with STATE.lock:
        off = STATE.offsets.get(fam)
        return off["emitted"] - off["engine"] if off else 0.0


def _cached_offset_base():
    """Offset base for the prefix-cache family. Caller must hold
    STATE.lock (non-reentrant lock: no nested acquisition here)."""
    off = STATE.offsets.get("vllm_prefix_cache_hits_total")
    return off["emitted"] - off["engine"] if off else 0.0


def main():
    persist_load()
    threading.Thread(target=poll_loop, daemon=True).start()
    threading.Thread(target=probe_loop, daemon=True).start()
    server = ThreadingHTTPServer((BIND, PORT), Handler)
    print(f"tensorfold-metrics-exporter v{VERSION} listening on "
          f"{BIND}:{PORT} -> {UPSTREAM}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
