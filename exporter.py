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
PREFIXES = ("tensorfold_health:", "tensorfold:", "velogb10_")
# The velogb10_ strip (v1.5.0) lets the veloGB10 engine land in the same DIRECT /
# health-counter paths as the TensorFold engines: velogb10_prompt_tokens_total
# post-strip reads exactly like the Python engine's prompt_tokens_total. No other
# schema emits a velogb10_-prefixed family, so the strip is inert for them.
# Native vLLM exposition namespace (colon-separated). Spark Dashboard's
# VllmAdapter matches UNDERSCORE names, so these are renamed 1:1.
VLLM_PREFIX = "vllm:"
VERSION = "1.7.0"

# How long an engine /health "live" reading stays believable. The engine keeps its own
# 2 s sliding window (core/lane_host.zig: tokens over window_ns, divided by 2.0), so a
# few seconds of slack covers the poll gap; past it a dead engine must not keep the
# tile warm, and we fall back to the exporter-derived window.
LIVE_FRESH_S = float(os.environ.get("TF_LIVE_FRESH_S", "5"))

# Families the engine already counts that the vLLM namespace never sees. Only schemas we
# have scraped are listed: an invented name would silently produce no series, and a wrong
# one worse - it would look like the engine stopped counting. The flash17 pin H engine
# publishes COLON names (schema python-v1); its underscore twin (zig-v1) is not listed
# because that family set was never verified on a live node.
EXTRA_SOURCES = {
    "python-v1": {
        "rounds": ("tensorfold:decode_rounds_total",),
        "dev_mem": ("tensorfold:device_memory_bytes",),
        "dev_mem_peak": ("tensorfold:device_memory_peak_bytes",),
        "pinned_mem": ("tensorfold:pinned_memory_bytes",),
        "rss": ("tensorfold:process_resident_bytes",),
        "rss_anon": ("tensorfold:process_resident_anon_bytes",),
        "rss_file": ("tensorfold:process_resident_file_bytes",),
        "rss_shmem": ("tensorfold:process_resident_shmem_bytes",),
        "rss_peak": ("tensorfold:process_resident_peak_bytes",),
        "disconnects": ("tensorfold:client_disconnections_total",),
        "gen_running": ("tensorfold:generation_tokens_running",),
    },
    # veloGB10 (TP=1 EXL3 engine). Family names verified against a live
    # v0.7.8 /metrics exposition; an invented name would silently vanish.
    "velo-v1": {
        "rounds": ("velogb10_spec_rounds_total",),
        "rss": ("velogb10_process_rss_bytes",),
        "lanes": ("velogb10_max_batch",),
        "kv_capacity": ("velogb10_kv_positions_capacity",),
        "kv_used": ("velogb10_kv_positions_used",),
        "kv_cached": ("velogb10_kv_positions_cached",),
        "kv_peak": ("velogb10_kv_positions_peak",),
        "kv_alloc": ("velogb10_kv_allocated_bytes",),
        "lane_kv_capacity": ("velogb10_kv_lane_positions_capacity",),
        "lane_kv_used": ("velogb10_kv_lane_positions_used",),
        "lane_kv_cached": ("velogb10_kv_lane_positions_cached",),
        "lane_kv_peak": ("velogb10_kv_lane_positions_peak",),
    },
}

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

# veloGB10-only base names (matched after the velogb10_ strip, only when
# STATE.schema == "velo-v1", so a foreign engine that later grows a same-named
# family cannot be silently re-labelled). request_duration_seconds is the
# engine's end-to-end histogram: its buckets map straight onto the vLLM e2e
# family. prompt_tokens_cached_total is the prefix-cache hit counter.
VELO_DIRECT = {
    "prompt_tokens_cached_total": "vllm_prefix_cache_hits_total",
    "request_duration_seconds": "vllm_e2e_request_latency_seconds",
    "spec_draft_tokens_total": "vllm_spec_decode_num_draft_tokens_total",
    "spec_accepted_tokens_total": "vllm_spec_decode_num_accepted_tokens_total",
}

# veloGB10 liveness/saturation gauges kept under tfexporter_ names: scheduler
# stall detection and lane-reserve pressure, the two things an operator reads
# first when a sweep starts crawling.
VELO_TFEXPORTER_BASES = ("scheduler_busy", "scheduler_last_step_age_seconds",
                         "scheduler_steps_total", "lane_reserve_waiting",
                         "over_cap_arrivals_total")

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
        # v4.2: the engine's own live block, read from /health's "live" object
        self.live_engine = None       # dict, or None when the engine lacks the block
        self.live_engine_ts = 0.0
        self.queue_waiting = None
        self.live_connections = None
        self.prefill_lastjob = None
        # v1.4.0: native families the vLLM namespace has no name for, plus the /health
        # identity block (warming / lanes / version), both parsed per poll.
        self.extras = {}
        self.health_info = {}
        # v1.5.0: veloGB10 version string from velogb10_build_info (one parse).
        self.velo_build = ""
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
    "tandem-v1": {
        "gen": ("qse_generation_tokens_total",),
        "dec": ("qse_time_per_output_token_seconds_sum",),
        "prompt": ("qse_prefill_tokens_forwarded_total",),
        "pf": ("qse_request_prefill_seconds_sum",),
        "run": ("qse_requests_running",),
    },
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
    # veloGB10 has no decode-seconds counter: "dur" (request_duration_seconds_sum,
    # recorded at request completion) minus pf (time_to_first_token_seconds_sum,
    # recorded at first token) is the honest decode-seconds numerator, applied
    # in update_native_stats' velo branch.
    # Flare has no request-seconds counters at all: "dec" is filled by the
    # flare branch with the exporter-measured occupancy integral (time-integral
    # of flare_active_requests on the polling timeline), so rate_decode reads
    # tokens per second PER ACTIVE REQUEST (documented in HELP, not measured
    # per-request seconds). "pf" stays 0: no prefill-seconds source exists, so
    # rate_prefill is honestly 0 and prefill tiles use the counter deltas.
    "flare-v1": {
        "gen": ("flare_generated_tokens_total",),
        "dec": (),
        "prompt": ("flare_prompt_tokens_synthetic",),
        "pf": ("flare_prefill_seconds_synthetic",),
        "run": ("flare_active_requests",),
    },
    "velo-v1": {
        "gen": ("velogb10_generation_tokens_total",),
        "dec": (),
        "dur": ("velogb10_request_duration_seconds_sum",),
        "prompt": ("velogb10_prompt_tokens_total",),
        "pf": ("velogb10_time_to_first_token_seconds_sum",),
        "run": ("velogb10_requests_running",),
    },
}


def detect_schema(text):
    """Classify the upstream /metrics schema (engine auto-detection).

    vllm-v1:   native vLLM exposition (`vllm:prompt_tokens_total` with
               engine/model_name labels) - e.g. the exlr8 GLM build
    zig-v1:    underscore families with labels (the flash17 Zig engine)
    python-v1: colon families (tensorfold:*) from the Python engine
    velo-v1:   the veloGB10 EXL3 engine (`velogb10_prompt_tokens_total`,
               TP=1 single-GB10 Qwen3.8/DeepSeek builds)
    flare-v1:  the Flare (Hyperflare) Rust engine (`flare_prefill_tokens_total`,
               TP=1/2 NVFP4 builds); all vllm_* families come from the
               engine_native_series flare branch because Flare's family names
               share no suffix with the vLLM namespace
    tandem-v1: TandemLLM (qwen38-spark-engine, `qse_requests_running`); the
               engine publishes a real Prometheus /metrics in the `qse_*`
               namespace with genuine TTFT/E2E/TPOT/queue histograms, so the
               dashboard set is a near-1:1 rename; raw qse_* families pass
               through untouched (zero name collisions, verified 2026-11-12)
    unknown:   nothing recognised
    """
    if "vllm:prompt_tokens_total" in text or "vllm:generation_tokens_total" in text:
        return "vllm-v1"
    if "tensorfold_completion_tokens_total" in text or "tensorfold_uptime_seconds" in text:
        return "zig-v1"
    if "tensorfold:generation_tokens_total" in text:
        return "python-v1"
    if "velogb10_prompt_tokens_total" in text:
        return "velo-v1"
    if "flare_prefill_tokens_total" in text:
        return "flare-v1"
    if "qse_requests_running" in text:
        return "tandem-v1"
    return "unknown"


def _label_safe(text):
    """Keep a free-form engine string from breaking the exposition format."""
    return str(text).replace("\\", "").replace('"', "").replace("\n", " ").strip()[:96]


def update_engine_extras(text):
    """Collect the native families listed in EXTRA_SOURCES for the current schema.

    Values are kept as the engine gave them; a family that is absent stays absent so the
    emitter can omit it instead of inventing a zero.
    """
    wanted = EXTRA_SOURCES.get(STATE.schema)
    found = {}
    if wanted:
        for line in text.splitlines():
            if not line or line.startswith("#"):
                continue
            head = line.split(" ", 1)[0]
            base = head.split("{", 1)[0]
            for key, names in wanted.items():
                if base in names and key not in found:
                    try:
                        found[key] = float(line.rsplit(" ", 1)[1])
                    except (IndexError, ValueError):
                        pass
    with STATE.lock:
        STATE.extras = found
    return found


def update_health_info(health):
    """The /health identity and readiness fields: what engine, how many lanes, warming?"""
    info = {}
    if isinstance(health, dict):
        for src, dst in (("warming", "warming"), ("max_batch_size", "lanes"),
                         ("version", "version"), ("backend", "backend"),
                         ("model", "model")):
            if src in health:
                info[dst] = health[src]
    with STATE.lock:
        STATE.health_info = info
    return info


def engine_native_series(existing_names):
    """Native engine families the adapter has no name for, as (name, value, type, doc).

    Emitted from ONE place: the live gauges taught us that a second emitter with its own
    opinion is how a dashboard ends up showing a stale or duplicated number. A name the
    body already carries is skipped, so this can never shadow the engine's own sample.
    """
    with STATE.lock:
        extras = dict(STATE.extras)
        info = dict(STATE.health_info)
        schema = STATE.schema
        velo_build = STATE.velo_build
    rows = []

    def add(name, value, kind, doc):
        if name in existing_names:
            return
        rows.append((name, value, kind, doc))

    if STATE.schema == "tandem-v1":
        t = STATE.tandem or {}

        def tv(name):
            return t.get(name)

        def pair(dst, srcname, help_text):
            s = tv(srcname + "_sum")
            c = tv(srcname + "_count")
            if s is None or c is None:
                return
            add(dst + "_sum", f'{persisted_value("tandem_" + dst + "_sum", s):.6f}',
                "counter", help_text)
            add(dst + "_count", str(int(persisted_value("tandem_" + dst + "_count", c))),
                "counter", help_text)

        run = tv("qse_requests_running")
        if run is not None:
            add("vllm_num_requests_running", str(int(run)), "gauge",
                "Requests generating (qse_requests_running).")
        wait = tv("qse_requests_waiting")
        if wait is not None:
            add("vllm_num_requests_waiting", str(int(wait)), "gauge",
                "Queued requests (qse_requests_waiting).")
        fresh = getattr(STATE, "tandem_fresh", None)
        if fresh is not None:
            add("vllm_prompt_tokens_total",
                str(int(persisted_value("tandem_prompt_tokens_total", fresh))),
                "counter",
                "FRESH prefill tokens, defined as qse_prompt_tokens_total - "
                "qse_prompt_tokens_cached_total. NOT "
                "qse_prefill_tokens_forwarded_total: that counter carries "
                "4096-grid chunk padding and breaks the adapter identity. "
                "The dashboard derives the Prefill tiles from this counter as "
                "COMPUTE throughput (skill rule). SERVED lives in "
                "vllm_prefix_cache_queries_total.")
        served = tv("qse_prompt_tokens_total")
        if served is not None:
            add("vllm_prefix_cache_queries_total",
                str(int(persisted_value("tandem_prefix_cache_queries_total", served))),
                "counter",
                "Served prompt tokens (qse_prompt_tokens_total, fresh + "
                "cached): the hit-rate denominator. Identity: queries - "
                "prompt_tokens = hits holds on the engine's own counters.")
        hits = getattr(STATE, "tandem_hits", None)
        if hits is not None:
            add("vllm_prefix_cache_hits_total",
                str(int(persisted_value("tandem_prefix_cache_hits_total", hits))),
                "counter",
                "Prefix-cache hits in tokens: qse_prompt_tokens_cached_total "
                "clamped to served, falling back to qse_cache_hits_- "
                "total{kind=prefix}; paired with prompt=queries-hits so the "
                "identity is exact.")
        gen_fin = tv("qse_generation_tokens_total")
        live = getattr(STATE, "tandem_live_gen", 0.0)
        if gen_fin is not None or live:
            g = max(gen_fin or 0.0, live)
            add("vllm_generation_tokens_total",
                str(int(persisted_value("tandem_generation_tokens_total", g))),
                "counter",
                "Hybrid LIVE generation counter: max(qse_generation_tokens_- "
                "total [incremented per finished request in on_request()], "
                "exporter-summed kv_length growth of decoding-phase slots). "
                "The engine has no live token counter; this keeps the Live "
                "decode tile non-zero mid-generation, the same class of fix "
                "as the tensorfold health-completion source.")
        succ = tv("qse_request_success_total")
        if succ is not None:
            add("vllm_request_success_total",
                str(int(persisted_value("tandem_request_success_total", succ))),
                "counter", "Finished requests (qse_request_success_total).")
        add("vllm_num_requests_swapped", "0", "counter",
            "Documented zero: TandemLLM has no batching scheduler - server/"
            "metrics.py states 'no preemption to report; those metrics belong "
            "to a batching scheduler and this server has none'.")
        add("vllm_num_preemptions_total", "0", "counter",
            "Documented zero: 8 independent lanes, no preemption path exists "
            "(server/metrics.py, same statement as swapped).")
        kr = getattr(STATE, "tandem_kv_ratio", None)
        if kr is not None:
            add("vllm_kv_cache_usage_perc", f"{kr:.4f}", "gauge",
                "KV usage 0..1 (house semantics): sum of slots[].kv_length "
                "over busy slots / (QSE_PARALLEL x max_len), from /health "
                "parallel.slots; 0 when every lane is idle.")
            add("vllm_gpu_cache_usage_perc", f"{kr:.4f}", "gauge",
                "Twin of kv_cache_usage_perc (adapter fallback chain).")
        pair("vllm_time_to_first_token_seconds", "qse_time_to_first_token_seconds",
             "Native engine histogram renamed from qse_time_to_first_token_- "
             "seconds (per-request TTFT at /v1; probe disabled).")
        pair("vllm_e2e_request_latency_seconds", "qse_e2e_request_latency_seconds",
             "Native per-request end-to-end seconds (qse_e2e_request_latency_- "
             "seconds), renamed.")
        pair("vllm_request_time_per_output_token_seconds", "qse_time_per_output_token_seconds",
             "Native per-request TPOT seconds (qse_time_per_output_token_- "
             "seconds), renamed.")
        pair("vllm_time_per_output_token_seconds", "qse_time_per_output_token_seconds",
             "Alias of the TPOT pair under the tile name without the request_ "
             "prefix; the engine has one measured TPOT population, not two.")
        pair("vllm_inter_token_latency_seconds", "qse_time_per_output_token_seconds",
             "Alias of TPOT: the engine publishes no separate inter-token "
             "histogram; same measured population, documented, not modeled.")
        pair("vllm_request_queue_time_seconds", "qse_request_queue_seconds",
             "Native per-request queue seconds (qse_request_queue_seconds), "
             "renamed _seconds -> _time_seconds for the adapter.")
        for dst, srcname in (
                ("vllm_spec_decode_num_drafts_total", "qse_spec_decode_num_drafts_total"),
                ("vllm_spec_decode_num_draft_tokens_total", "qse_spec_decode_num_draft_tokens_total"),
                ("vllm_spec_decode_num_accepted_tokens_total", "qse_spec_decode_num_accepted_tokens_total")):
            v = tv(srcname)
            if v is not None:
                add(dst, str(int(persisted_value("tandem_" + dst, v))),
                    "counter", f"Renamed from {srcname}.")
        d = tv("qse_spec_decode_num_draft_tokens_total")
        a = tv("qse_spec_decode_num_accepted_tokens_total")
        if d and a is not None:
            add("tfexporter_tandem_spec_accept_ratio", f"{a / d:.4f}", "gauge",
                "accepted/draft tokens (lifetime).")
            add("tfexporter_tandem_first_draft_accept_pct", f"{100.0 * a / d:.2f}", "gauge",
                "Same ratio as a percentage, for the tile footer.")
        vc = tv("qse_verify_seconds_count")
        if vc is not None:
            # adapter Batch tile: tokens per verify step = gen / verify-rounds
            gmax = max(gen_fin or 0.0, live)
            add("vllm_iteration_tokens_total_count",
                str(int(persisted_value("tandem_iteration_count", vc))), "counter",
                "Verify rounds (qse_verify_seconds_count): the denominator "
                "that makes sum/count = tokens per verify step.")
            add("vllm_iteration_tokens_total_sum",
                str(int(persisted_value("tandem_iteration_sum", gmax))), "counter",
                "Generation tokens (same population as vllm_generation_- "
                "tokens_total) paired with verify rounds for the Batch tile.")
        ps = tv("qse_request_prefill_tokens_per_second_sum")
        pc = tv("qse_request_prefill_tokens_per_second_count")
        if ps and pc:
            add("tfexporter_tandem_prefill_tps_mean", f"{ps / pc:.1f}", "gauge",
                "Mean of the engine's measured per-request prefill rate "
                "(qse_request_prefill_tokens_per_second sum/count).")
        ds = tv("qse_request_decode_tokens_per_second_sum")
        dc = tv("qse_request_decode_tokens_per_second_count")
        if ds and dc:
            add("tfexporter_tandem_decode_tps_mean", f"{ds / dc:.1f}", "gauge",
                "Mean of the engine's measured per-request decode rate.")
        add("tfexporter_tandem_live_decode_tps",
            f"{getattr(STATE, 'tandem_live_rate', 0.0):.2f}", "gauge",
            "Tokens/s summed over decoding-phase slots (kv_length growth, "
            "2s polls, EMA half-life one poll).")
        rs = getattr(STATE, "tandem", {}) or {}
        if rs.get("qse_reasoning_tokens_total") is not None:
            add("tfexporter_tandem_reasoning_tokens_total",
                str(int(persisted_value("tandem_reasoning_tokens_total",
                                        rs["qse_reasoning_tokens_total"]))),
                "counter", "Reasoning tokens (qse_reasoning_tokens_total).")

    if schema == "flare-v1":
        pt = getattr(STATE, "flare_prompt_total", None)
        pt0 = getattr(STATE, "flare_e_prompt0", None)
        gt = getattr(STATE, "flare_gen_total", None)
        if pt is not None and pt0 is not None:
            pt = pt - pt0  # era population (same window as the ttft sum)
        if pt is not None:
            add("vllm_prompt_tokens_total", str(int(persisted_value("flare_prompt_tokens_total", pt))),
                "counter",
                "FRESH prompt tokens only (flare_prefill_tokens_total): the "
                "engine excludes prefix-reused tokens from prefill work, and "
                "the dashboard derives Prefill Live/Avg/Per-Req from this "
                "counter as COMPUTE throughput - counting reused tokens there "
                "inflates it by the cache-hit share. Lifetime population.")
        if gt is not None:
            add("vllm_generation_tokens_total", str(int(persisted_value("flare_generation_tokens_total", gt))),
                "counter",
                "Committed generated tokens (flare_generated_tokens_total, "
                "including EOS and stop-withheld tokens per docs/api.md).")
        run = getattr(STATE, "flare_running", None)
        if run is not None:
            add("vllm_num_requests_running", str(int(run)), "gauge",
                "Model slots with a live request (flare_active_requests).")
        wait = getattr(STATE, "flare_waiting", None)
        if wait is not None:
            add("vllm_num_requests_waiting", str(int(wait)), "gauge",
                "Requests waiting in the scheduler (flare_pending_requests).")
        add("vllm_num_requests_swapped", "0", "counter",
            "Documented zero: Flare does not preempt to host KV (preemptions "
            "are counted separately as tfexporter_flare_preemptions_total).")
        kr = getattr(STATE, "flare_kv_ratio", None)
        if kr is not None:
            add("vllm_kv_cache_usage_perc", f"{kr:.4f}", "gauge",
                "KV pool usage as a 0..1 ratio (house semantics, tile "
                "multiplies by 100): (flare_total_kv_tokens - "
                "flare_free_kv_tokens) / flare_total_kv_tokens.")
        rz = getattr(STATE, "flare_reused", None)
        rz0 = getattr(STATE, "flare_e_reused0", None)
        if rz is not None and rz0 is not None:
            rz = rz - rz0
        if rz is not None:
            add("vllm_prefix_cache_hits_total", str(int(persisted_value("flare_prefix_cache_hits_total", rz))),
                "counter",
                "GPU prefix-cache hits in TOKENS (flare_cache_reused_tokens_"
                "total), same unit convention as the velo profile's hits.")
        sv = getattr(STATE, "flare_prompt_served", None)
        sv0 = getattr(STATE, "flare_e_served0", None)
        if sv is not None and sv0 is not None:
            sv = sv - sv0
        if sv is not None:
            add("vllm_prefix_cache_queries_total", str(int(persisted_value("flare_prefix_cache_queries_total", sv))),
                "counter",
                "SERVED prompt tokens (fresh prefill + prefix-reused) as the "
                "queries denominator; hit% = reused / served. Deliberately "
                "larger than vllm_prompt_tokens_total (compute-only).")
        med = getattr(STATE, "flare_ttft_med", None)
        ti = getattr(STATE, "flare_e_ttft_int", None)
        d0 = getattr(STATE, "flare_e_done0", None)
        dc = getattr(STATE, "flare_completed", None)
        if ti is not None and d0 is not None and dc is not None and dc > d0:
            done = dc - d0
            add("vllm_time_to_first_token_seconds_sum", "%.6f" % persisted_value("flare_ttft_int", ti), "counter",
                "ENGINE-SCALE estimator: Flare publishes only a rolling-window "
                "median TTFT (flare_first_response_median_seconds over "
                "window_seconds). sum = window median x lifetime completed "
                "requests, so mean_ms(sum,count) = current window median while "
                "count is the true request total the dashboard shows; the "
                "Prefill Per-Req tile (prompt_total/sum) becomes "
                "mean-prompt-length / window-median-TTFT. The raw per-window "
                "median and its sample/truncation gauges are surfaced as "
                "tfexporter_flare_first_response_*. sum/count = weighted mean "
                "TTFT over the exporter era; prompt_tokens_total is baselined "
                "to the SAME era so the fixed Prefill Per-Req formula reads a "
                "population-consistent rate.")
            add("vllm_time_to_first_token_seconds_count", str(int(done)), "counter",
                "Completed requests over the same era as the sum "
                "(flare_requests_completed_total minus its attach baseline).")
        ai = getattr(STATE, "flare_a_int_last", None)
        g0 = getattr(STATE, "flare_a_gen0", None)
        if ai is not None and gt is not None and g0 is not None and gt >= g0:
            # Population-consistent window: request-seconds and tokens both
            # measured since the integral era opened (exporter attach /
            # engine restart); each side persists its own offset so an
            # exporter restart shifts both and the ratio stays continuous.
            ai_p = persisted_value("flare_active_integral", ai)
            tok_p = persisted_value("flare_tpot_tokens", gt - g0)
            add("vllm_request_time_per_output_token_seconds_sum", "%.3f" % ai_p, "counter",
                "DERIVED, not per-request measured: integral of "
                "flare_active_requests dt on the exporter polling timeline "
                "(request-seconds in the engine), opened at exporter attach.")
            add("vllm_request_time_per_output_token_seconds_count", str(int(tok_p)), "counter",
                "Completion tokens over the SAME window as the integral "
                "(flare_generated_tokens_total minus its attach baseline), "
                "so mean_ms(sum,count) = mean TPOT across active requests.")
            add("vllm_inter_token_latency_seconds_sum", "%.3f" % ai_p, "counter",
                "ITL alias of the occupancy-derived TPOT pair above (same "
                "seconds/token population; stated, not fabricated).")
            add("vllm_inter_token_latency_seconds_count", str(int(tok_p)), "counter",
                "Token intervals (completion tokens) over the same window.")
        qi = getattr(STATE, "flare_q_int", None)
        qb = getattr(STATE, "flare_q_base", None)
        qd = getattr(STATE, "flare_q_done", None)
        if qi is not None and qb is not None and qd is not None and qd > qb:
            add("vllm_request_queue_time_seconds_sum", "%.1f" % persisted_value("flare_queue_integral", qi), "counter",
                "Little's-law estimate (Flare has no queue-seconds counter): "
                "trapezoidal integral of flare_pending_requests over the "
                "exporter 2 s polling timeline, rebased on engine restart.")
            add("vllm_request_queue_time_seconds_count", str(int(qd - qb)), "counter",
                "Requests completed in the same window as the queue integral.")
        steps = getattr(STATE, "flare_steps", None)
        if steps and gt is not None:
            add("vllm_iteration_tokens_total_sum", str(int(persisted_value("flare_iteration_tokens_sum", gt))), "counter",
                "Sum = completion tokens; count = scheduler executions "
                "(sum of flare_scheduler_executions_total over reason labels). "
                "sum/count = tokens per step = effective decode batch.")
            add("vllm_iteration_tokens_total_count", str(int(persisted_value("flare_iteration_count", steps))), "counter",
                "Physical scheduler executions (docs/api.md scheduler table).")
        off = getattr(STATE, "flare_off", None)
        acc = getattr(STATE, "flare_acc", None)
        if off is not None:
            add("vllm_spec_decode_num_draft_tokens_total", str(int(persisted_value("flare_spec_offered", off))), "counter",
                "Speculative draft positions offered (flare_speculative_"
                "offered_drafts_total).")
        if acc is not None:
            add("vllm_spec_decode_num_accepted_tokens_total", str(int(persisted_value("flare_spec_accepted", acc))), "counter",
                "Accepted draft tokens (flare_speculative_accepted_drafts_"
                "total, excluding bonus token).")
        vd = getattr(STATE, "flare_ver_drafts", None)
        vr = getattr(STATE, "flare_ver_rounds", None)
        if vd is not None and vr:
            add("tfexporter_flare_verification_depth", "%.2f" % (vd / vr), "gauge",
                "Mean draft-prefix length per progressing round: "
                "flare_verification_draft_tokens_total / "
                "flare_verification_request_rounds_total (docs/api.md DEPTH).")
        ar = getattr(STATE, "flare_accept_ratio", None)
        if ar is not None:
            add("tfexporter_flare_spec_accept_ratio", f"{ar:.4f}", "gauge",
                "accepted/offered draft tokens (lifetime since engine start).")
        pre = getattr(STATE, "flare_preempt", None)
        if pre is not None:
            pre_p = persisted_value("flare_preemptions_total", pre)
            add("vllm_num_preemptions_total", str(int(pre_p)), "counter",
                "Requests preempted by the scheduler (flare_preemptions_total), "
                "under the name the VllmAdapter reads.")
            add("tfexporter_flare_preemptions_total", str(int(pre_p)), "counter",
                "Same series, tfexporter-namespaced twin.")
        rounds = getattr(STATE, "flare_rounds", None)
        if rounds is not None:
            add("vllm_spec_decode_num_drafts_total", str(int(persisted_value("flare_spec_rounds", rounds))), "counter",
                "Speculative verification rounds (flare_speculative_rounds_"
                "total summed over batch/width labels). Mean acceptance length "
                "= accepted / this; plausibility gate: Batch tile ~= this "
                "ratio + 1 (MTP).")
        if med is not None:
            add("tfexporter_flare_first_response_median_seconds", f"{med:.6f}", "gauge",
                "Engine's own rolling-window TTFT median, raw.")
        ns = getattr(STATE, "flare_ttft_n", None)
        if ns is not None:
            add("tfexporter_flare_first_response_samples", str(int(ns)), "gauge",
                "Samples inside the current first-response window; median is "
                "trustworthy when this is healthy and truncated == 0.")
        kr2 = getattr(STATE, "flare_kv_ratio", None)
        if kr2 is not None:
            add("vllm_gpu_cache_usage_perc", f"{kr2:.4f}", "gauge",
                "Twin of vllm_kv_cache_usage_perc (the adapter reads either).")
        trunc = getattr(STATE, "flare_ttft_trunc", None)
        if trunc is not None:
            add("tfexporter_flare_first_response_truncated", str(int(trunc)), "gauge",
                "Samples truncated at window end (0 = trustworthy median).")
        gl = getattr(STATE, "flare_gen_last_min", None)
        if gl is not None:
            add("tfexporter_flare_generated_tokens_last_minute", str(int(gl)), "gauge",
                "Engine-reported generated tokens in the last 60 s (native live rate).")
        fs = getattr(STATE, "flare_free_slots", None)
        if fs is not None:
            add("tfexporter_flare_free_request_slots", str(int(fs)), "gauge",
                "Free model slots (flare_free_request_slots).")
    if schema == "velo-v1":
        ts = getattr(STATE, "velo_tpot_sum", None)
        tc = getattr(STATE, "velo_tpot_count", None)
        if ts is not None and tc is not None:
            add("vllm_request_time_per_output_token_seconds_sum", f"{ts:.3f}",
                "counter",
                "Total decode seconds over finished requests: "
                "request_duration_seconds_sum minus time_to_first_token_seconds_sum "
                "(veloGB10 exposes no decode-seconds counter). Feeds the dashboard "
                "Decode Throughput Per-Req Avg as count/sum.")
            add("vllm_request_time_per_output_token_seconds_count", str(int(tc)),
                "counter",
                "Completion tokens across all traffic (per-stream TPS denominator, "
                "same convention as the zig profile).")
        # ITL: same finished-request population as the TPOT pair above,
        # aliased under vLLM's name (pinH convention: HELP states the alias).
        if ts is not None and tc is not None:
            add("vllm_inter_token_latency_seconds_sum", f"{ts:.3f}", "counter",
                "Inter-token latency, same seconds/token population as the "
                "synthesized TPOT pair (decode seconds over finished requests). "
                "vLLM-named alias for the dashboard ITL tile.")
            add("vllm_inter_token_latency_seconds_count", str(int(tc)),
                "counter",
                "Token intervals (completion tokens), vLLM-named alias of the "
                "TPOT denominator.")
            # Avg batch: tokens per scheduler decode step (gen/steps). With
            # MTP this is the effective decode-batch width per step.
            steps = getattr(STATE, "velo_steps", None)
            if steps:
                add("vllm_iteration_tokens_total_sum", str(int(tc)), "counter",
                    "Tokens produced across scheduler decode steps "
                    "(generation_tokens_total); pair with _count for avg batch.")
                add("vllm_iteration_tokens_total_count", str(int(steps)),
                    "counter",
                    "Scheduler decode steps (velogb10_scheduler_steps_total); "
                    "sum/count = tokens per step = effective decode batch.")
        tpb = getattr(STATE, "velo_tps_buckets", None)
        tot = getattr(STATE, "velo_tps_total", None)
        if tpb and tot:
            # ITL/TPOT SLO buckets: the engine publishes a PER-REQUEST
            # decode-rate histogram (tokens/s); per-request TPOT/ITL is its
            # exact reciprocal, so these buckets are a lossless change of
            # variable of a measured distribution, not a modeled one.
            # Requests sitting exactly on a bucket edge count as SLO misses
            # (conservative; float measure-zero in practice).
            for base, doc in (
                ("vllm_request_time_per_output_token_seconds",
                 "Per-request time per output token: derived as 1/tps from the "
                 "engine's velogb10_decode_tokens_per_second_bucket histogram "
                 "(exact reciprocal of a per-request measurement)."),
                ("vllm_inter_token_latency_seconds",
                 "Per-request inter-token latency: same reciprocal derivation "
                 "as the TPOT histogram; same population as the ITL sum/count "
                 "alias this exporter emits."),
            ):
                if base + "_bucket" in existing_names:
                    continue
                rows.append(("__meta__", base, "histogram", doc))
                for le_s, cc in sorted((1.0 / le_t, tot - cum_t)
                                       for le_t, cum_t in tpb if le_t > 0):
                    rows.append(("__sample__", '%s_bucket{le="%g"} %d'
                                 % (base, le_s, int(cc)), None, None))
                rows.append(("__sample__", '%s_bucket{le="+Inf"} %d'
                             % (base, int(tot)), None, None))
        qi = getattr(STATE, "velo_q_int", None)
        qb = getattr(STATE, "velo_q_base", None)
        qd = getattr(STATE, "velo_q_done", None)
        if qi is not None and qb is not None and qd is not None and qd > qb:
            add("vllm_request_queue_time_seconds_sum", "%.1f" % qi, "counter",
                "Queue wait seconds, Little's-law estimate: trapezoidal "
                "time-integral of velogb10_requests_waiting sampled per "
                "exporter poll since the exporter attached (veloGB10 has no "
                "native queue-seconds counter).")
            add("vllm_request_queue_time_seconds_count", str(int(qd - qb)),
                "counter",
                "Requests completed in the same window as the queue-time "
                "integral (the Little's-law denominator).")
        pq = getattr(STATE, "velo_prefix_queries", None)
        if pq is not None:
            # The dashboard's Cache block computes hit rate only when queries
            # is present and >0; without this the tiles stay blank forever
            # (v1.5.3). Queries = prompt tokens served, zig convention.
            add("vllm_prefix_cache_queries_total", str(int(pq)), "counter",
                "Prompt tokens served (prefix-cache denominator, same "
                "convention as the zig profile: every prompt token is a "
                "cache query).")

    rounds = extras.get("rounds")
    if rounds is not None:
        # The engine counts decode rounds; the adapter's draft-count family is the only
        # place a dashboard will show them. This replaces the old read of /health
        # "rounds_total", a field the pin H engine has never published.
        add("vllm_spec_decode_num_drafts_total", str(int(rounds)), "counter",
            "Decode rounds the engine has run (tensorfold:decode_rounds_total); each "
            "round advances every live stream by one drafted/verified step.")

    mem = [("active", "dev_mem"), ("peak", "dev_mem_peak"), ("pinned", "pinned_mem")]
    got_mem = [(kind, extras[key]) for kind, key in mem if extras.get(key) is not None]
    if got_mem and "tfexporter_engine_device_memory_bytes" not in existing_names:
        rows.append(("__meta__", "tfexporter_engine_device_memory_bytes", "gauge",
                     "Device memory the engine's allocator reports, in bytes: active = "
                     "in use, peak = high-water since engine start (reset only by an "
                     "explicit /health?reset_peak=1, which this exporter never calls), "
                     "pinned = host memory pinned for the device."))
        for kind, v in got_mem:
            rows.append(("__sample__", ('tfexporter_engine_device_memory_bytes{kind="%s"}'
                                        % kind) + " " + str(int(v)), None, None))

    rss = [("rss", "rss"), ("anon", "rss_anon"), ("file", "rss_file"),
           ("shmem", "rss_shmem"), ("peak", "rss_peak")]
    got = [(kind, extras[key]) for kind, key in rss if extras.get(key) is not None]
    if got and "tfexporter_engine_process_resident_bytes" not in existing_names:
        rows.append(("__meta__", "tfexporter_engine_process_resident_bytes", "gauge",
                     "Engine process RSS breakdown in bytes (anon/file/shmem, total and "
                     "high-water peak), straight from the engine's own counters."))
        for kind, v in got:
            rows.append(("__sample__", ('tfexporter_engine_process_resident_bytes{kind="%s"}'
                                        % kind) + " " + str(int(v)), None, None))

    d = extras.get("disconnects")
    if d is not None:
        add("tfexporter_client_disconnections_total", str(int(d)), "counter",
            "Client connections dropped by the engine mid-stream (aborts, dropped SSE "
            "subscribers); pairs with tfexporter_prefilling when a stream dies early.")
    g = extras.get("gen_running")
    if g is not None:
        add("tfexporter_generation_tokens_running", str(int(g)), "gauge",
            "Tokens generated so far by requests still in flight (excludes prompts).")

    if "warming" in info:
        add("tfexporter_engine_warming",
            "1" if info["warming"] in (True, 1, "true") else "0", "gauge",
            "1 while the engine is still warming its graphs/pools and would not serve a "
            "prompt at speed; 0 once ready.")
    if isinstance(info.get("lanes"), (int, float)) and not isinstance(info.get("lanes"), bool):
        add("tfexporter_engine_lanes", str(int(info["lanes"])), "gauge",
            "Engine lanes = the hard cap on concurrent requests (/health max_batch_size). "
            "Compare with vllm_num_requests_running: pinned at this number means the "
            "engine is saturated, not slow.")
    elif extras.get("lanes") is not None:
        # veloGB10 /health carries no max_batch_size; the engine publishes its
        # lane cap as velogb10_max_batch on /metrics instead (it is runtime-
        # adjustable, so a static config value would lie).
        add("tfexporter_engine_lanes", str(int(extras["lanes"])), "gauge",
            "Engine lanes = the hard cap on concurrent requests (veloGB10 "
            "velogb10_max_batch). Pinned at this number means saturated, not slow.")

    # veloGB10 KV story: the pre-allocating lane pool, from the engine's own
    # position counters. used/capacity is the honest pool utilization the vLLM
    # gauge would show on a paged engine; peak/cached distinguish 'full of live
    # context' from 'full of kept prompts' — the question a deep sweep raises.
    kv = [("pool", "capacity", extras.get("kv_capacity")),
          ("pool", "used", extras.get("kv_used")),
          ("pool", "cached", extras.get("kv_cached")),
          ("pool", "peak", extras.get("kv_peak")),
          ("lane", "capacity", extras.get("lane_kv_capacity")),
          ("lane", "used", extras.get("lane_kv_used")),
          ("lane", "cached", extras.get("lane_kv_cached")),
          ("lane", "peak", extras.get("lane_kv_peak"))]
    got_kv = [(scope, state, v) for scope, state, v in kv if v is not None]
    if got_kv and "tfexporter_kv_positions" not in existing_names:
        rows.append(("__meta__", "tfexporter_kv_positions", "gauge",
                     "KV pool positions (tokens): scope=pool is the whole engine "
                     "pool, scope=lane the per-lane reserve; state=capacity/used/"
                     "cached/peak. From velogb10_kv_*_positions_* counters."))
        for scope, state, v in got_kv:
            rows.append(("__sample__",
                         ('tfexporter_kv_positions{scope="%s",state="%s"} %d'
                          % (scope, state, int(v))), None, None))
        alloc = extras.get("kv_alloc")
        if alloc is not None:
            add("tfexporter_kv_allocated_bytes", str(int(alloc)), "gauge",
                "Bytes the engine has allocated for its KV pools "
                "(velogb10_kv_allocated_bytes).")
        cap = extras.get("kv_capacity")
        used = extras.get("kv_used")
        if cap and used is not None:
            # vLLM v1 semantics: this gauge is a 0..1 RATIO; the dashboard
            # multiplies by 100 itself. Emitting a percent here showed 391%
            # on the tile (fixed in v1.5.1).
            ratio = used / cap
            add("vllm_kv_cache_usage_perc", f"{ratio:.4f}", "gauge",
                "KV pool utilization ratio (0..1, vLLM v1 semantics): velogb10 positions used/capacity. "
                "used counts live context positions only; retained cached prompt "
                "blocks are separate (see tfexporter_kv_positions{state=\"cached\"}).")

    version = _label_safe(info.get("version", ""))
    backend = _label_safe(info.get("backend", ""))
    if schema == "velo-v1":
        # veloGB10 /health is {status, telemetry} only — no version/backend fields.
        version = version or _label_safe(velo_build)
        backend = backend or "velogb10"
    model = _label_safe(info.get("model", "") or MODELS_CACHE.get("model_id") or "")
    if (version or backend) and "tfexporter_engine_info" not in existing_names:
        # One sample only, the labelled one: a bare tfexporter_engine_info 1 next to the
        # labelled form made the series read as two different engines (caught live, not
        # in the unit suite, which only inspected the rows in isolation).
        rows.append(("__meta__", "tfexporter_engine_info", "gauge",
                     "Constant 1; the labels carry what the engine reported about itself "
                     "over /health, so a fleet can be read without ssh."))
        rows.append(("__sample__", 'tfexporter_engine_info{version="%s",backend="%s",'
                     'model="%s"} 1' % (version, backend, model), None, None))
    return rows


def update_engine_live(health, now):
    """Take the engine's own live rates out of /health's "live" block.

    The engine already keeps a 2 s sliding window of decoded tokens, so the exporter
    must not re-derive and then re-smooth what it is being told. The derived window
    stays as the fallback for engines without the block (old health-profile builds,
    native vLLM) and for the seconds around a restart.

    prefill_tokens_per_second is deliberately not used as a rate: the engine writes it
    once per finished prompt pass, timed from admission, so it carries queue time and
    is one global last-writer-wins slot. We publish it as an informational series.
    """
    live = health.get("live") if isinstance(health, dict) else None
    if not isinstance(live, dict):
        with STATE.lock:
            STATE.live_engine = None
            STATE.queue_waiting = None
            STATE.live_connections = None
            STATE.prefill_lastjob = None
        return False

    def _num(key):
        v = live.get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None
        return float(v)

    dec = _num("decode_tokens_per_second")
    if dec is None:
        with STATE.lock:
            STATE.live_engine = None
            STATE.queue_waiting = None
            STATE.live_connections = None
            STATE.prefill_lastjob = None
        return False
    with STATE.lock:
        STATE.live_engine = {"decode_tokens_per_second": dec,
                             "prefill_tokens_per_second": _num("prefill_tokens_per_second"),
                             "waiting": _num("waiting"),
                             "connections": _num("connections")}
        STATE.live_engine_ts = now
        STATE.queue_waiting = STATE.live_engine["waiting"]
        STATE.live_connections = STATE.live_engine["connections"]
        STATE.prefill_lastjob = STATE.live_engine["prefill_tokens_per_second"]
    return True


def live_decode_choice(now=None):
    """Decide which decode rate the tile shows, and what to publish beside it.

    One place on purpose: the gauge is emitted both by self_status_lines() (native vLLM
    passthrough) and by rewrite() (both TensorFold profiles), and two emitters with two
    opinions is how a dashboard ends up showing a stale number. The engine's own 2 s
    window wins while it is fresh - including a fresh 0, which honestly means "nothing
    decoded in the last two seconds" - otherwise the exporter's derived window speaks.
    """
    now = time.time() if now is None else now
    with STATE.lock:
        derived = STATE.rate_decode or 0.0
        live = dict(STATE.live_engine) if isinstance(STATE.live_engine, dict) else None
        age = now - STATE.live_engine_ts
        waiting = STATE.queue_waiting
        conns = STATE.live_connections
    fresh = bool(live) and age <= LIVE_FRESH_S
    if fresh:
        return {"value": live["decode_tokens_per_second"], "source": 1,
                "derived": derived, "live": live, "fresh": True,
                "waiting": waiting, "conns": conns}
    return {"value": derived, "source": 0, "derived": derived, "live": None,
            "fresh": False, "waiting": waiting, "conns": conns}


def engine_live_series(choice):
    """The extra series a fresh engine live block buys us, as (name, value, kind, doc)."""
    rows = []
    live = choice.get("live")
    if not live:
        return rows
    pj = live.get("prefill_tokens_per_second")
    if pj is not None:
        rows.append(("tfexporter_engine_prefill_lastjob_tps", f"{pj:.1f}", "gauge",
                     "The engine's prefill figure for its most recent finished prompt "
                     "pass: uncached prompt tokens over the time since ADMISSION, held "
                     "for the engine's window. Informational only - it carries queue "
                     "time; use tfexporter_live_prefill_tps for a rate."))
        rows.append(("tfexporter_prefilling", "1" if pj > 0 else "0", "gauge",
                     "1 while the engine has published a recent prompt pass."))
    if live.get("waiting") is not None:
        rows.append(("tfexporter_queue_waiting", str(int(live["waiting"])), "gauge",
                     "Requests waiting for a lane, from the same /health snapshot as "
                     "the live decode rate."))
    if live.get("connections") is not None:
        rows.append(("tfexporter_live_connections", str(int(live["connections"])),
                     "gauge", "Live connections held by the engine, from /health."))
    return rows


def update_tandem_health(now):
    """TandemLLM: pull /health for the slot view (KV ratio + the live decode
    window). The engine's generation counter is finished-only (on_request()
    increments it when the request ends), so the LIVE token feed is the
    kv_length growth of slots in the decoding phase; vllm_generation_tokens_-
    total publishes max(finished, live) so short requests that finish between
    two polls still land."""
    try:
        _, body, _ = fetch_upstream("/health")
        h = json.loads(body.decode("utf-8", "replace"))
    except Exception:
        return
    par = h.get("parallel") or {}
    slots = par.get("slots") or []
    n_lanes = par.get("n") or len(slots) or 1
    max_len = 0.0
    busy_kv = 0.0
    dec = {}
    for i, s in enumerate(slots):
        ml = s.get("max_len")
        if ml:
            max_len = max(max_len, float(ml))
        kv = s.get("kv_length")
        if kv:
            busy_kv += float(kv)
        # engine's own decoding set (server/slots.py:1367): phase is
        # "decode"|"runner" while a request is in the block loop. "" means
        # finished-resident (kv kept for the prefix cache) - must NOT count.
        if s.get("phase") in ("decode", "runner", "solo") and kv:
            dec[i] = float(kv)
    ratio = None
    if max_len > 0:
        ratio = min(1.0, busy_kv / (n_lanes * max_len))
    with STATE.lock:
        STATE.tandem_health = h
        STATE.tandem_kv_ratio = ratio
        # per-slot deltas: a slot SEEDS on its first decoding sample (its
        # kv_length there is prompt + generated-so-far; counting the jump
        # would bill prompt tokens as generation), and only later samples
        # of the same decoding slot contribute positive growth.
        prev = getattr(STATE, "tandem_prev_dec_kv", None) or {}
        d = 0.0
        for i, kv in dec.items():
            if i in prev:
                d += max(0.0, kv - prev[i])
        STATE.tandem_prev_dec_kv = dec
        if d > 0:
            STATE.tandem_live_gen = getattr(STATE, "tandem_live_gen", 0.0) + d
        dt = now - (getattr(STATE, "tandem_prev_ts", None) or now)
        if dt > 0:
            inst = max(0.0, d / dt)
            STATE.tandem_live_rate = (inst if getattr(STATE, "tandem_live_rate", 0.0) <= 0
                                      else 0.5 * STATE.tandem_live_rate + 0.5 * inst)
        STATE.tandem_prev_ts = now


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
    if STATE.schema == "velo-v1":
        # No decode-seconds counter exists on this engine: total completed-request
        # time minus TTFT time is the honest decode numerator (both are recorded
        # per finished request, so the difference shares the same denominator).
        vals["dec"] = max(0.0, (vals.get("dur") or 0.0) - (vals.get("pf") or 0.0))
    if STATE.schema == "tandem-v1":
        # TandemLLM (qwen38-spark-engine). qse_* Prometheus families; family
        # inventory verified against a live /metrics scrape 2026-11-12 (282
        # series, 93 families) and the tile audit against the spark-dashboard
        # binary. The engine separates FRESH prefill (forwarded) from SERVED
        # prompt and CACHED tokens as its own counters, and publishes real
        # per-request histograms for TTFT / E2E / TPOT / queue - all lifetime
        # sums, so pairs pass straight through with exporter-restart offsets.
        t = {}
        hits_prefix = 0.0
        for line in text.splitlines():
            if line.startswith("qse_"):
                base = line.split(" ", 1)[0]
                name = base.split("{", 1)[0]
                v = _fnum(line)
                if v is None:
                    continue
                if name == "qse_cache_hits_total" and 'kind="prefix"' in base:
                    hits_prefix += v
                elif "{" in base:
                    t[name] = t.get(name, 0.0) + v
                else:
                    t[name] = v
        STATE.tandem = t
        # FRESH is DEFINED as served - cached, never as qse_prefill_tokens_-
        # forwarded_total: the forwarded counter includes chunk-boundary
        # padding on the 4096 grid (measured 2026-11-12: +20 800 tokens over
        # 43 requests, which pushed queries - prompt negative). Defining
        # fresh this way makes the adapter identity queries - prompt = hits
        # exact on every traffic mix. forwarded stays as a diagnostic only.
        served_v = t.get("qse_prompt_tokens_total")
        cached_v = t.get("qse_prompt_tokens_cached_total")
        if cached_v is None:
            cached_v = hits_prefix
        if served_v is not None:
            STATE.tandem_hits = min(cached_v or 0.0, served_v)
            STATE.tandem_fresh = max(0.0, served_v - STATE.tandem_hits)
        else:
            STATE.tandem_hits = None
            STATE.tandem_fresh = None
        # prefer the token-level counter; fall back to the prefix kind of
        # qse_cache_hits_total (chunk lookups) if the engine lacks it
        STATE.tandem_cache_prefix_hits = t.get("qse_prompt_tokens_cached_total")
        if STATE.tandem_cache_prefix_hits is None:
            STATE.tandem_cache_prefix_hits = hits_prefix
        STATE.tandem_up = t.get("qse_requests_total")
    if STATE.schema == "flare-v1":
        # Flare (Hyperflare) Rust engine. Family inventory verified against a
        # live /metrics scrape 2026-11-12 (docs/api.md "Serving terminal" table).
        # No per-request seconds exist anywhere on this engine: TPOT/ITL and the
        # decode rate ride on the occupancy integral below, queue time on
        # Little's law - both documented in their HELP, neither invented.
        f = {}
        sched_sum = None
        for line in text.splitlines():
            if line.startswith("flare_"):
                base = line.split(" ", 1)[0]
                name = base.split("{", 1)[0]
                v = _fnum(line)
                if v is None:
                    continue
                if name == "flare_scheduler_executions_total":
                    sched_sum = (sched_sum or 0.0) + v
                elif "{" in base:
                    # labeled family: sum across labels (rounds by batch/width, ...)
                    f[name] = f.get(name, 0.0) + v
                else:
                    # unlabeled: LAST match wins (fixture/dup hygiene)
                    f[name] = v
        gen_v = f.get("flare_generated_tokens_total")
        prefill = f.get("flare_prefill_tokens_total") or 0.0
        reused = f.get("flare_cache_reused_tokens_total") or 0.0
        # vLLM prompt_tokens semantics include prefix-reused tokens: served
        # prompt = fresh prefill + reused. (rate_prefill stays 0: no
        # prefill-seconds source; prompt tiles are counter-driven.)
        vals["prompt"] = prefill + reused
        cached_v = reused if reused else None
        up = f.get("flare_uptime_seconds")
        active = f.get("flare_active_requests")
        # hoisted: era init and the TTFT accumulator both need it
        done = f.get("flare_requests_completed_total")
        with STATE.lock:
            if cached_v is not None:
                STATE.last_cached = cached_v
            # Occupancy integral (trapezoidal over the 2 s polling timeline):
            # integral of active-requests dt = request-seconds spent in the
            # engine. Monotonic within an engine era; rebased on restart.
            if active is not None and gen_v is not None:
                if getattr(STATE, "flare_a_t", None) is None or up is not None and                         getattr(STATE, "flare_a_up", None) is not None and up < STATE.flare_a_up:
                    STATE.flare_a_int = 0.0
                    STATE.flare_a_t = now
                    STATE.flare_a_a = active
                    STATE.flare_a_up = up
                    # same era baseline for the token side of the TPOT pair:
                    # sum and count must cover the SAME window or
                    # mean_ms(sum,count) divides an attach-time integral by
                    # lifetime tokens (v1.6.1 audit caught 8401 t/s nonsense)
                    STATE.flare_a_gen0 = gen_v
                    # ...and for the Prefill Per-Req pair (v1.6.2 audit: the
                    # adapter's fixed formula prompt_total/ttft_sum mixed a
                    # lifetime numerator with a current-window denominator:
                    # 6355 t/s during tool-eval against a measured 2.3k cold
                    # plateau). All prefill-side counters baseline together.
                    STATE.flare_e_prompt0 = prefill or 0.0
                    STATE.flare_e_served0 = vals.get("prompt") or 0.0
                    STATE.flare_e_reused0 = reused
                    STATE.flare_e_done0 = done
                    STATE.flare_e_done_last = done
                    STATE.flare_e_ttft_int = 0.0
                else:
                    dt_a = now - STATE.flare_a_t
                    if 0 < dt_a < 300:
                        STATE.flare_a_int += (STATE.flare_a_a + active) * 0.5 * dt_a
                    STATE.flare_a_t = now
                    STATE.flare_a_a = active
                STATE.flare_a_int_last = STATE.flare_a_int
            vals["dec"] = getattr(STATE, "flare_a_int_last", 0.0)
            # Little's-law queue integral (same discipline as velo v1.5.5):
            # waiting-seconds / completions over the same polling window.
            pend = f.get("flare_pending_requests")
            done = f.get("flare_requests_completed_total")
            if pend is not None and done is not None:
                if getattr(STATE, "flare_q_t", None) is None or done < (STATE.flare_q_base or 0):
                    STATE.flare_q_int = 0.0
                    STATE.flare_q_t = now
                    STATE.flare_q_w = pend
                    STATE.flare_q_base = done
                else:
                    dt_q = now - STATE.flare_q_t
                    if 0 < dt_q < 300:
                        STATE.flare_q_int += (STATE.flare_q_w + pend) * 0.5 * dt_q
                    STATE.flare_q_t = now
                    STATE.flare_q_w = pend
                STATE.flare_q_done = done
            if getattr(STATE, "flare_e_done_last", None) is not None and done is not None:
                med_now = f.get("flare_first_response_median_seconds")
                dn = done - STATE.flare_e_done_last
                if dn > 0 and med_now is not None:
                    # true sum of per-request TTFTs approximated with the
                    # engine's window median at the moment of completion
                    STATE.flare_e_ttft_int += dn * med_now
                    STATE.flare_e_done_last = done
                elif dn > 0 and med_now is None:
                    pass  # hold: absorbed by the next poll, never invented
            STATE.flare_preempt = f.get("flare_preemptions_total")
            STATE.flare_failed = f.get("flare_requests_failed_total")
            STATE.flare_cancelled = f.get("flare_requests_cancelled_total")
            STATE.flare_waiting = pend
            STATE.flare_running = active
            tot = f.get("flare_total_kv_tokens")
            free = f.get("flare_free_kv_tokens")
            if tot and free is not None and tot > 0:
                STATE.flare_kv_ratio = max(0.0, min(1.0, (tot - free) / tot))
                STATE.flare_kv_used = tot - free
                STATE.flare_kv_total = tot
            STATE.flare_completed = f.get("flare_requests_completed_total")
            STATE.flare_rounds = f.get("flare_speculative_rounds_total")
            STATE.flare_ttft_med = f.get("flare_first_response_median_seconds")
            STATE.flare_ttft_n = f.get("flare_first_response_samples")
            STATE.flare_ttft_win = f.get("flare_first_response_window_seconds")
            STATE.flare_ttft_trunc = f.get("flare_first_response_truncated")
            if sched_sum is not None:
                STATE.flare_steps = sched_sum
            STATE.flare_ver_drafts = f.get("flare_verification_draft_tokens_total")
            STATE.flare_ver_rounds = f.get("flare_verification_request_rounds_total")
            STATE.flare_acc = f.get("flare_speculative_accepted_drafts_total")
            STATE.flare_off = f.get("flare_speculative_offered_drafts_total")
            STATE.flare_free_slots = f.get("flare_free_request_slots")
            STATE.flare_cache_hits = f.get("flare_cache_hits_total")
            STATE.flare_cache_lookups = f.get("flare_cache_lookups_total")
            STATE.flare_gen_last_min = f.get("flare_generated_tokens_last_minute")
            if STATE.flare_acc is not None and STATE.flare_off:
                STATE.flare_accept_ratio = STATE.flare_acc / STATE.flare_off
            if up is not None:
                if STATE.prev_uptime is not None and up < STATE.prev_uptime:
                    STATE.restarts += 1
                    STATE.offsets = {}
                    STATE.probe_hist = {}
                    STATE.probe = {"ttft": [0.0, 0], "e2e": [0.0, 0], "itl": [0.0, 0]}
                    STATE.rate_prefill_hold = None
                    STATE.flare_a_t = None       # integrals are era-bound too
                    STATE.flare_q_t = None
                STATE.prev_uptime = up
            # Tile-semantics split (v1.6.2): the dashboard's Prefill tiles
            # (Live/Avg = delta of prompt_tokens_total; Per-Req = prompt_total
            # / ttft_sum) measure COMPUTE, so the numerator must exclude
            # prefix-reused tokens - they cost no prefill time and inflated
            # Per-Req to 6.9k+ t/s against a measured 2.3k plateau. Served
            # (fresh+reused) stays only as the cache-queries denominator.
            STATE.flare_prompt_total = prefill if prefill else 0.0
            STATE.flare_prompt_served = vals.get("prompt")
            STATE.flare_gen_total = gen_v
            STATE.flare_reused = reused
            STATE.last_inflight = int(active or 0)
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
    elif STATE.schema == "velo-v1":
        # veloGB10: its own prefix-cache hit counter, admission refusals, and an
        # uptime regression that means the engine was restarted (same wipe
        # discipline as zig: rates, offsets and probe history are era-bound).
        # mem_available_bytes is the engine's reading of the serving host - use it
        # instead of the exporter-local /proc/meminfo, which is the wrong machine
        # whenever the exporter runs remotely (e.g. .21 watching .25).
        cached_v = rejected_v = up = None
        mem_v = None
        steps_v = None
        build = None
        tps_b = []
        wait_v = None
        done_v = None
        done_bare = None
        for line in text.splitlines():
            if line.startswith("velogb10_prompt_tokens_cached_total"):
                cached_v = _fnum(line)
            elif line.startswith("velogb10_requests_rejected_total"):
                rejected_v = _fnum(line)
            elif line.startswith("velogb10_uptime_seconds"):
                up = _fnum(line)
            elif line.startswith("velogb10_mem_available_bytes"):
                mem_v = _fnum(line)
            elif line.startswith("velogb10_build_info"):
                build = line
            elif line.startswith("velogb10_scheduler_steps_total"):
                steps_v = _fnum(line)
            elif line.startswith("velogb10_decode_tokens_per_second_bucket"):
                k1 = line.find('le="')
                if k1 >= 0:
                    k2 = line.find('"', k1 + 4)
                    try:
                        tps_b.append((float(line[k1 + 4:k2]),
                                      float(line.rsplit(" ", 1)[1])))
                    except ValueError:
                        pass
            elif line.startswith("velogb10_requests_waiting"):
                wait_v = _fnum(line)
            elif line.startswith("velogb10_requests_total{"):
                v_fin = _fnum(line)
                if v_fin is not None:
                    done_v = (done_v or 0.0) + v_fin
            elif line.startswith("velogb10_requests_total "):
                done_bare = _fnum(line)
        with STATE.lock:
            if cached_v is not None:
                STATE.last_cached = cached_v
            if rejected_v is not None:
                STATE.last_rejected = rejected_v
            # Cumulative TPOT-shape pair for the dashboard's Decode Throughput
            # "Per-Req Avg": same house convention as the zig emitter
            # (sum = finished decode seconds, count = completion tokens).
            STATE.velo_tpot_sum = dec
            STATE.velo_tpot_count = gen
            STATE.velo_prefix_queries = prompt
            if mem_v is not None:
                STATE.mem_gib = mem_v / (1024.0 ** 3)
            if steps_v is not None:
                STATE.velo_steps = steps_v
            if tps_b:
                inf_cum = None
                finite = []
                for le_t, cum_t in tps_b:
                    if le_t == float("inf"):
                        inf_cum = cum_t
                    else:
                        finite.append((le_t, cum_t))
                STATE.velo_tps_buckets = sorted(finite)
                STATE.velo_tps_total = inf_cum if inf_cum is not None else 0
            # Queue time: veloGB10 publishes no queue-seconds counter, so the
            # exporter measures it with Little's law on its own polling
            # timeline: mean queue wait = (time-integral of the waiting gauge)
            # / completions over the SAME window (L = lambda * W). Trapezoidal
            # integration of the gauge sampled once per poll; the window opens
            # when the exporter first sees the engine, and rebases on any
            # counter reset (engine restart). Both emitted series share that
            # window, so sum/count is a coherent average.
            if done_v is None and done_bare is not None:
                done_v = done_bare   # bare form (some builds): no finish labels
            if wait_v is not None and done_v is not None:
                if getattr(STATE, "velo_q_t", None) is None or done_v < STATE.velo_q_base:
                    STATE.velo_q_int = 0.0
                    STATE.velo_q_t = now
                    STATE.velo_q_w = wait_v
                    STATE.velo_q_base = done_v
                else:
                    dt_q = now - STATE.velo_q_t
                    if 0 < dt_q < 300:
                        STATE.velo_q_int += (STATE.velo_q_w + wait_v) * 0.5 * dt_q
                    STATE.velo_q_t = now
                    STATE.velo_q_w = wait_v
                STATE.velo_q_done = done_v
            if build is not None and STATE.velo_build == "":
                i = build.find('version="')
                if i >= 0:
                    j = build.find('"', i + 9)
                    if j > i:
                        STATE.velo_build = build[i + 9:j]
            if up is not None:
                if STATE.prev_uptime is not None and up < STATE.prev_uptime:
                    STATE.restarts += 1
                    STATE.offsets = {}
                    STATE.probe_hist = {}
                    STATE.probe = {"ttft": [0.0, 0], "e2e": [0.0, 0], "itl": [0.0, 0]}
                    STATE.rate_prefill_hold = None  # new era: no rate history
                STATE.prev_uptime = up
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
                     if STATE.schema in ("zig-v1", "vllm-v1", "velo-v1", "flare-v1") else None)
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
    if STATE.schema in ("zig-v1", "vllm-v1", "velo-v1", "flare-v1"):
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
    """Common tfexporter self-observability series for any engine profile.

    The live decode tile prefers the engine's own 2 s window; the exporter-derived
    window is always emitted next to it so a divergence is visible rather than hidden.
    """
    now = time.time()
    with STATE.lock:
        ok_ts = STATE.last_ok_ts
        rate_prefill = STATE.rate_prefill
    choice = live_decode_choice(now)
    decode_val, decode_src = choice["value"], choice["source"]
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
        ("tfexporter_live_decode_tps", f"{decode_val:.1f}", "gauge",
         "Live decode rate (tokens/s). From the engine's /health live window when "
         "that reading is fresh (see tfexporter_live_decode_tps_source), else the "
         "exporter's held window: delta(generation tokens)/delta(decode seconds)."),
        ("tfexporter_live_decode_tps_source", str(decode_src), "gauge",
         "Provenance of tfexporter_live_decode_tps: 1 = the engine's own live "
         "window, 0 = exporter-derived over finished counters."),
        ("tfexporter_live_decode_tps_derived", f"{choice['derived']:.1f}", "gauge",
         "The exporter-derived decode rate (tokens/s), kept even when the engine "
         "supplies one, so the two can be compared."),
        ("tfexporter_live_prefill_tps", f"{rate_prefill or 0.0:.1f}", "gauge",
         "Held prefill rate (tokens/s): delta(prompt tokens - cached tokens) "
         "/ delta(TTFT seconds) over the trailing window; 0 when idle. Always "
         "exporter-derived: the engine's prefill figure is per-job and includes "
         "queue time, so it is not a rate."),
    )
    rows += tuple(engine_live_series(choice))
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
    with STATE.lock:
        schema = STATE.schema
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
            elif schema == "velo-v1" and base in VELO_DIRECT:
                set_meta(VELO_DIRECT[base], kind, doc)
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
        if schema == "velo-v1" and base in VELO_TFEXPORTER_BASES:
            put_sample(f"tfexporter_{base}", labels, value)
            continue

        if base in DIRECT:
            put_sample(DIRECT[base] + suffix, labels, value)
        elif schema == "velo-v1" and base in VELO_DIRECT:
            put_sample(VELO_DIRECT[base] + suffix, labels, value)

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
    #    The engine's own /health live window is preferred when fresh; the derived
    #    window stays published next to it so a divergence is visible, not hidden.
    # Which denominator the derived prefill rate really used: the zig engine exposes a
    # prefill-seconds counter (pure compute), the python engine and the health profile
    # fall back to TTFT sums (queue included). Say so instead of guessing in the docs.
    with STATE.lock:
        prefill_doc_seconds = (profile == "native" and STATE.schema == "zig-v1")
    choice = live_decode_choice()
    if profile != "native":
        # Old health-profile engine: the exporter's own live figure there is the live
        # completion-counter diff, not the native finished-request window, so that diff
        # is what the derived series has to report - and what the tile shows until the
        # engine starts publishing a live block of its own.
        with STATE.lock:
            health_live = STATE.live_tps or 0.0
        choice = dict(choice, derived=health_live)
        if not choice["fresh"]:
            choice = dict(choice, value=health_live, source=0)
    for name, value, kind, doc in (
            ("tfexporter_live_decode_tps", f"{choice['value']:.1f}", "gauge",
             "Live decode rate (tokens/s). From the engine's /health live window "
             "when that reading is fresh, else from the exporter: for the native "
             f"profile delta(generation_tokens_total)/delta(request_decode_seconds_sum) "
             f"over the trailing {int(RATE_WINDOW_S)}s window of finished requests, "
             "held while requests are in flight; for the health profile the live "
             "completion counter diff. 0 when idle."),
            ("tfexporter_live_decode_tps_source", str(choice["source"]), "gauge",
             "Provenance of tfexporter_live_decode_tps: 1 = the engine's own live "
             "window from /health, 0 = exporter-derived."),
            ("tfexporter_live_decode_tps_derived", f"{choice['derived']:.1f}", "gauge",
             "The exporter-derived decode rate (tokens/s), always emitted so the "
             "engine-supplied tile can be compared against it."),
            ("tfexporter_live_prefill_tps", f"{rate_prefill or 0.0:.1f}", "gauge",
             "Held prefill rate estimate: delta(prompt tokens - cached tokens) / "
             + ("delta(tensorfold_prefill_seconds_total), engine compute only, no "
                "queue time" if prefill_doc_seconds else
                "delta(time_to_first_token_seconds_sum), which includes queue time")
             + f", over the trailing {int(RATE_WINDOW_S)}s window; held while requests "
             "are in flight, 0 when idle. An aggregate over the window, not a per-job "
             "figure: concurrent requests and prefix-cache hits can push it well above "
             "any single request's rate. Always exporter-derived, because the engine's "
             "own prefill figure is per-job and timed from admission, so it is not a "
             "rate (see tfexporter_engine_prefill_lastjob_tps)."),
    ):
        set_meta(name, "TYPE", kind)
        set_meta(name, "HELP", doc)
        put_sample(name, "", value)
    for name, value, kind, doc in engine_live_series(choice):
        set_meta(name, "TYPE", kind)
        set_meta(name, "HELP", doc)
        put_sample(name, "", value)

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
            note_engine_latency_hist(text)
            with STATE.lock:
                STATE.profile = "native" if native else "health"
                if schema != STATE.schema:
                    STATE.schema = schema
                    STATE.window = []
                    STATE.rate_decode = 0.0
                    STATE.rate_prefill = 0.0
            # Collected for every schema: a schema change must clear the previous
            # engine's numbers rather than keep serving them as if they were current.
            update_engine_extras(text)
            if native:
                update_native_stats(text, now)
                if STATE.schema == "tandem-v1":
                    update_tandem_health(now)
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
                update_engine_live(STATE.health, now)
                update_health_info(STATE.health)
            except (urllib.error.URLError, OSError, ValueError):
                pass
            try:
                # velo-v1 skips the exporter-local reading: update_native_stats
                # already set mem_gib from velogb10_mem_available_bytes, which is
                # the SERVING host's value (the exporter itself may run elsewhere).
                if schema != "velo-v1":
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
        extra = self._v4_gauges(body)
        self._send(200, body + extra, "text/plain; version=0.0.4; charset=utf-8")

    def _v4_gauges(self, body=b""):
        NL = chr(10)
        # Names the rewritten body already carries: a native sample from the engine always
        # wins over our rename, so the series below must never shadow or duplicate it.
        existing = set()
        for line in body.decode("utf-8", "replace").splitlines():
            if line and not line.startswith("#"):
                existing.add(line.split("{", 1)[0].split(" ", 1)[0])
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
        for row in engine_native_series(existing):
            head, a, b, c = row
            if head == "__meta__":
                out.append(("# HELP " + a + " " + c + NL + "# TYPE " + a + " " + b + NL).encode())
            elif head == "__sample__":
                out.append((a + NL).encode())
            else:
                out.append(("# HELP " + head + " " + c + NL + "# TYPE " + head + " " + b +
                            NL + head + " " + a + NL).encode())
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


# The synthetic probe exists only for engines that cannot measure their own
# latency (see probe_enabled). TensorFold pin H and newer expose real
# TTFT/E2E/ITL/TPOT histograms over ALL traffic, so probing them is wasted
# prefill that also pollutes the honest histograms with a fixed-size artificial
# request. Detect it from the upstream body only, never from our own output.
HIST_MARKERS = ("time_to_first_token_seconds",
                "request_time_per_output_token_seconds")
ENGINE_HAS_LATENCY_HIST = False


def note_engine_latency_hist(text):
    """Remember whether the upstream already publishes latency histograms."""
    global ENGINE_HAS_LATENCY_HIST
    seen = any(marker in text for marker in HIST_MARKERS)
    if seen != ENGINE_HAS_LATENCY_HIST:
        ENGINE_HAS_LATENCY_HIST = seen
        print("tf-exporter: upstream latency histograms "
              + ("present -> synthetic probe disabled" if seen
                 else "absent -> synthetic probe enabled"), flush=True)


def probe_enabled():
    """Probe only engines that expose no latency histograms of their own.

    A native vLLM upstream measures TTFT/E2E/ITL/TPOT itself, so probing it
    would add load and duplicate honest engine series. The same reasoning
    applies to a TensorFold engine that publishes the same families.
    """
    with STATE.lock:
        if STATE.schema in ("vllm-v1", "flare-v1", "tandem-v1"):
            return False
        return not ENGINE_HAS_LATENCY_HIST


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
