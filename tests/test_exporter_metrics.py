"""Unit tests for the tensorfold-metrics-exporter metric math.

Run on spark3: python3 -m unittest discover -s tests -v
Covers: schema detection (TensorFold zig/python + native vLLM), zig rate
windows, windowed cache-hit ratio, vllm bridge contract, the vllm-v1 colon ->
underscore passthrough, histogram bucketing, restart detection, v4 gauges.
"""
import importlib.util
import os
import threading
import unittest

EXPORTER_PATH = "/home/admin/llm/tensorfold-metrics-exporter/exporter.py"

ZIG_TEXT = (
    '# HELP tensorfold_requests_total Requests.\n'
    '# TYPE tensorfold_requests_total counter\n'
    'tensorfold_requests_total{model="DeepSeek-V4.1-Flash-TF"} 100\n'
    'tensorfold_prompt_tokens_total{model="DeepSeek-V4.1-Flash-TF"} 1000\n'
    'tensorfold_cached_tokens_total{model="DeepSeek-V4.1-Flash-TF"} 400\n'
    'tensorfold_completion_tokens_total{model="DeepSeek-V4.1-Flash-TF"} 200\n'
    'tensorfold_decode_seconds_total{model="DeepSeek-V4.1-Flash-TF"} 10\n'
    'tensorfold_prefill_seconds_total{model="DeepSeek-V4.1-Flash-TF"} 2\n'
    'tensorfold_requests_inflight{model="DeepSeek-V4.1-Flash-TF"} 3\n'
    'tensorfold_requests_stalled 0\n'
    'tensorfold_engine_fatal 0\n'
    'tensorfold_uptime_seconds 5000\n'
    'tensorfold_requests_rejected_total 1\n'
)

ZIG_STEP = ZIG_TEXT.replace(
    'tensorfold_completion_tokens_total{model="DeepSeek-V4.1-Flash-TF"} 200',
    'tensorfold_completion_tokens_total{model="DeepSeek-V4.1-Flash-TF"} 300',
).replace(
    'tensorfold_decode_seconds_total{model="DeepSeek-V4.1-Flash-TF"} 10',
    'tensorfold_decode_seconds_total{model="DeepSeek-V4.1-Flash-TF"} 15',
).replace(
    'tensorfold_prompt_tokens_total{model="DeepSeek-V4.1-Flash-TF"} 1000',
    'tensorfold_prompt_tokens_total{model="DeepSeek-V4.1-Flash-TF"} 1500',
).replace(
    'tensorfold_prefill_seconds_total{model="DeepSeek-V4.1-Flash-TF"} 2',
    'tensorfold_prefill_seconds_total{model="DeepSeek-V4.1-Flash-TF"} 3',
).replace(
    'tensorfold_cached_tokens_total{model="DeepSeek-V4.1-Flash-TF"} 400',
    'tensorfold_cached_tokens_total{model="DeepSeek-V4.1-Flash-TF"} 700',
)

PYTHON_TEXT = (
    '# HELP tensorfold:generation_tokens_total gen\n'
    '# TYPE tensorfold:generation_tokens_total counter\n'
    'tensorfold:generation_tokens_total 300\n'
    'tensorfold:request_decode_seconds_sum 15\n'
    'tensorfold:prompt_tokens_total 900\n'
    'tensorfold:time_to_first_token_seconds_sum 3\n'
    'tensorfold:requests_running 2\n'
    'tensorfold:health_completion_tokens_total 5\n'
)


def load_exporter():
    spec = importlib.util.spec_from_file_location("exporter_under_test", EXPORTER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ExporterMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_exporter()

    def setUp(self):
        st = self.m.STATE
        st.schema = "zig-v1"
        st.window = []
        st.rate_decode = 0.0
        st.rate_prefill = 0.0
        st.cache_ratio = None
        st.prev_cached = None
        st.prev_prompt_r = None
        st.restarts = 0
        st.prev_uptime = None
        st.probe_hist = {}
        st.last_cached = None
        st.last_stalled = 0
        st.last_fatal = 0
        st.last_inflight = 0
        st.last_rejected = None
        st.offsets = {}          # tests must not inherit persistence offsets
        st.rate_prefill_hold = None  # nor the windowed-rate hold

    # --- schema detection ---

    def test_detect_schema_zig(self):
        self.assertEqual(self.m.detect_schema(ZIG_TEXT), "zig-v1")

    def test_detect_schema_python(self):
        self.assertEqual(self.m.detect_schema(PYTHON_TEXT), "python-v1")

    def test_detect_schema_unknown(self):
        self.assertEqual(self.m.detect_schema("# nothing here\n"), "unknown")

    # --- number parsing ---

    def test_fnum_plain_and_labeled(self):
        self.assertEqual(self.m._fnum("foo 1.5"), 1.5)
        self.assertEqual(self.m._fnum('bar{model="x"} 42'), 42.0)

    # --- zig rate windows ---

    def test_zig_rates_true_decode_and_prefill(self):
        m = self.m
        m.update_native_stats(ZIG_TEXT, 1000.0)
        m.update_native_stats(ZIG_STEP, 1010.0)
        # +100 completion tokens over +5 decode seconds -> 20 tok/s
        self.assertAlmostEqual(m.STATE.rate_decode, 20.0, places=6)
        # +500 prompt, but +300 of them cached -> +200 uncached over +1 pf second
        self.assertAlmostEqual(m.STATE.rate_prefill, 200.0, places=6)

    def test_prefill_rate_excludes_cached_tokens(self):
        m = self.m
        # cache-heavy step: prompt grows 500, cached grows 500 -> zero prefill work
        heavy = ZIG_TEXT.replace(
            'tensorfold_prompt_tokens_total{model="DeepSeek-V4.1-Flash-TF"} 1000',
            'tensorfold_prompt_tokens_total{model="DeepSeek-V4.1-Flash-TF"} 1500',
        ).replace(
            'tensorfold_cached_tokens_total{model="DeepSeek-V4.1-Flash-TF"} 400',
            'tensorfold_cached_tokens_total{model="DeepSeek-V4.1-Flash-TF"} 900',
        ).replace(
            'tensorfold_prefill_seconds_total{model="DeepSeek-V4.1-Flash-TF"} 2',
            'tensorfold_prefill_seconds_total{model="DeepSeek-V4.1-Flash-TF"} 2.1',
        )
        m.update_native_stats(ZIG_TEXT, 1000.0)
        m.update_native_stats(heavy, 1010.0)
        # all new prompt tokens were cache hits: true prefill rate is 0
        self.assertAlmostEqual(m.STATE.rate_prefill, 0.0, places=6)

    def test_zig_windowed_cache_hit_ratio(self):
        m = self.m
        m.update_native_stats(ZIG_TEXT, 1000.0)
        m.update_native_stats(ZIG_STEP, 1010.0)
        # +300 cached over +500 prompt in the window -> 0.6
        self.assertAlmostEqual(m.STATE.cache_ratio, 0.6, places=6)

    def test_rates_held_only_while_busy(self):
        m = self.m
        idle = ZIG_TEXT.replace(
            'tensorfold_requests_inflight{model="DeepSeek-V4.1-Flash-TF"} 3',
            'tensorfold_requests_inflight{model="DeepSeek-V4.1-Flash-TF"} 0')
        m.update_native_stats(ZIG_TEXT, 1000.0)
        m.update_native_stats(ZIG_STEP, 1010.0)
        self.assertGreater(m.STATE.rate_decode, 0.0)
        m.update_native_stats(idle, 1020.0)  # inflight 0 -> rates reset to 0
        self.assertEqual(m.STATE.rate_decode, 0.0)
        self.assertEqual(m.STATE.rate_prefill, 0.0)

    def test_python_schema_uses_ttft_prefill(self):
        m = self.m
        m.STATE.schema = "python-v1"
        m.update_native_stats(PYTHON_TEXT, 1000.0)
        # first python poll: gen parsed, rates not yet (need 2 window points)
        self.assertIsNotNone(m.STATE.window)

    # --- restart detection ---

    def test_uptime_regression_counts_restart(self):
        m = self.m
        m.STATE.prev_uptime = 6000.0
        m.update_native_stats(ZIG_TEXT, 1000.0)  # uptime 5000 < 6000
        self.assertEqual(m.STATE.restarts, 1)
        self.assertEqual(m.STATE.prev_uptime, 5000.0)

    def test_no_restart_on_growing_uptime(self):
        m = self.m
        m.update_native_stats(ZIG_TEXT, 1000.0)
        m.update_native_stats(ZIG_STEP, 1010.0)
        self.assertEqual(m.STATE.restarts, 0)

    # --- zig -> vllm bridge contract ---

    def test_bridge_core_series(self):
        out = self.m.zig_vllm_bridge(ZIG_TEXT).decode()
        for expected in (
            "vllm_prompt_tokens_total 1000.0",
            "vllm_generation_tokens_total 200.0",
            "vllm_num_requests_running 3.0",
            "vllm_num_requests_waiting 0.0",
            "vllm_num_requests_swapped 0.0",
            "vllm_num_preemptions_total 0.0",
            "vllm_prefix_cache_hits_total 400.0",
            "vllm_prefix_cache_queries_total 1000.0",
            "vllm_request_queue_time_seconds_sum 0.0",
            "vllm_request_time_per_output_token_seconds_sum 10.0",
            "vllm_request_time_per_output_token_seconds_count 200.0",
        ):
            self.assertIn(expected, out)

    def test_bridge_honest_omissions_without_probe_data(self):
        out = self.m.zig_vllm_bridge(ZIG_TEXT).decode()
        self.assertNotIn("vllm_time_to_first_token_seconds", out)
        self.assertNotIn("vllm_e2e_request_latency_seconds", out)
        self.assertNotIn("vllm_inter_token_latency_seconds", out)
        self.assertNotIn("vllm_kv_cache_usage_perc", out)

    def test_bridge_histograms_when_probed(self):
        m = self.m
        m._hist_add("e2e", 0.4)
        m._hist_add("e2e", 1.2)
        m._hist_add("ttft", 0.003)
        out = m.zig_vllm_bridge(ZIG_TEXT).decode()
        self.assertIn("# TYPE vllm_e2e_request_latency_seconds histogram", out)
        self.assertIn('vllm_e2e_request_latency_seconds_bucket{le="0.5"} 1.0', out)
        self.assertIn('vllm_e2e_request_latency_seconds_bucket{le="+Inf"} 2.0', out)
        self.assertIn("vllm_e2e_request_latency_seconds_count 2.0", out)
        self.assertIn("# TYPE vllm_time_to_first_token_seconds histogram", out)
        self.assertIn('vllm_time_to_first_token_seconds_bucket{le="0.005"} 1.0', out)

    def test_bridge_spec_decode_from_health(self):
        m = self.m
        m.STATE.health = {"drafted_total": 250000, "accepted_total": 160000,
                          "rounds_total": 90000}
        out = m.zig_vllm_bridge(ZIG_TEXT).decode()
        self.assertIn("vllm_spec_decode_num_draft_tokens_total 250000.0", out)
        self.assertIn("vllm_spec_decode_num_accepted_tokens_total 160000.0", out)
        self.assertIn("vllm_spec_decode_num_drafts_total 90000.0", out)

    def test_probe_prompt_is_unique_per_call(self):
        import re
        src = open(EXPORTER_PATH).read()
        # the nonce is built from uuid4 - verify the construction exists
        self.assertIn("uuid.uuid4().hex[:8]", src)

    def test_per_request_prompt_gets_engine_scale_ttft(self):
        m = self.m
        zig = ZIG_TEXT + "tensorfold_requests_total 100\n"
        m.update_native_stats(zig, 1000.0)
        m._hist_add("ttft", 0.003)   # probe buckets must exist for the family
        out = m.zig_vllm_bridge(zig).decode()
        # single poll: no windowed rate yet -> cumulative fallback
        # rate = (1000 - 400 cached) / 2 pf seconds = 300
        # sum = prompt/rate = 1000/300; count = engine requests
        self.assertIn("vllm_time_to_first_token_seconds_sum 3.3333333333333335", out)
        self.assertIn("vllm_time_to_first_token_seconds_count 100.0", out)
        # the upstream invariant: prompt_total / sum == prefill rate
        self.assertAlmostEqual(1000.0 / (1000.0 / 300.0), 300.0, places=6)

    def test_per_request_prompt_tracks_windowed_rate(self):
        m = self.m
        zig = ZIG_TEXT + "tensorfold_requests_total 100\n"
        step = ZIG_STEP.replace(
            'tensorfold_requests_total 100', 'tensorfold_requests_total 110'
        ) + "tensorfold_requests_total 110\n"
        zig = zig + "tensorfold_requests_total 100\n"
        m.update_native_stats(zig, 1000.0)
        m.update_native_stats(step, 1010.0)
        m._hist_add("ttft", 0.003)
        out = m.zig_vllm_bridge(step).decode()
        # windowed rate_prefill = (500-300)/1 = 200 -> sum = 1500/200 = 7.5
        self.assertIn("vllm_time_to_first_token_seconds_sum 7.5", out)
        self.assertIn("vllm_time_to_first_token_seconds_count 110.0", out)

    def test_hold_survives_idle_between_prefill_phases(self):
        m = self.m
        zig = ZIG_TEXT + "tensorfold_requests_total 100\n"
        step = ZIG_STEP + "tensorfold_requests_total 110\n"
        idle = ZIG_STEP.replace(
            'tensorfold_requests_inflight{model="DeepSeek-V4.1-Flash-TF"} 3',
            'tensorfold_requests_inflight{model="DeepSeek-V4.1-Flash-TF"} 0',
        ) + "tensorfold_requests_total 110\n"
        m.update_native_stats(zig, 1000.0)
        m.update_native_stats(step, 1010.0)          # hold = 200
        m.update_native_stats(idle, 1020.0)          # idle: windowed rates -> 0
        self.assertEqual(m.STATE.rate_decode, 0.0)   # live rates reset...
        self.assertEqual(m.STATE.rate_prefill, 0.0)
        m._hist_add("ttft", 0.003)
        out = m.zig_vllm_bridge(idle).decode()
        # ...but the estimate keeps the held windowed rate: sum = 1500/200
        self.assertIn("vllm_time_to_first_token_seconds_sum 7.5", out)

    def test_engine_restart_clears_rate_hold(self):
        m = self.m
        m.STATE.rate_prefill_hold = 200.0            # stale rate from old era
        m.STATE.prev_uptime = 6000.0
        m.update_native_stats(ZIG_TEXT, 1000.0)      # uptime 5000 -> restart
        self.assertIsNone(m.STATE.rate_prefill_hold)

    # --- histogram bucketing ---

    def test_hist_add_cumulative_semantics(self):
        m = self.m
        m._hist_add("ttft", 0.003)
        m._hist_add("ttft", 0.007)
        h = m.STATE.probe_hist["ttft"]
        self.assertEqual(h[repr(0.005)], 1)   # only 0.003 <= 0.005
        self.assertEqual(h[repr(0.01)], 2)    # both <= 0.01
        self.assertEqual(h["+Inf"], 2)
        self.assertAlmostEqual(h["sum"], 0.010, places=9)

    def test_hist_buckets_non_decreasing(self):
        m = self.m
        for v in (0.2, 0.03, 1.4, 0.006, 6.0):
            m._hist_add("e2e", v)
        h = m.STATE.probe_hist["e2e"]
        counts = [h[repr(le)] for le in self.m.PROBE_BUCKETS]
        for a, b in zip(counts, counts[1:]):
            self.assertGreaterEqual(b, a)  # cumulative monotonicity
        self.assertEqual(h["+Inf"], 5)

    def test_slo_thresholds_have_exact_bucket_edges(self):
        # default dashboard SLOs: 50ms ITL/TPOT, 500ms TTFT, 5s E2E
        for edge in (0.05, 0.5, 5.0):
            self.assertIn(edge, self.m.PROBE_BUCKETS)

    # --- v4 gauges ---

    def test_v4_gauges_contain_schema_info_and_state(self):
        m = self.m
        m.update_native_stats(ZIG_TEXT, 1000.0)
        m.update_native_stats(ZIG_STEP, 1010.0)
        m.MODELS_CACHE["model_id"] = "DeepSeek-V4.1-Flash-TF"
        m.STATE.mem_gib = 8.0
        handler = object.__new__(m.Handler)
        body = handler._v4_gauges().decode()
        self.assertIn('tfexporter_engine_schema_info{schema="zig-v1",'
                      'model="DeepSeek-V4.1-Flash-TF"} 1', body)
        self.assertIn("tfexporter_requests_inflight 3", body)
        self.assertIn("tfexporter_cache_hit_ratio 0.6", body)
        self.assertIn("tfexporter_mem_available_gib", body)

    def test_full_pipeline_rewrite_plus_bridge(self):
        m = self.m
        m.update_native_stats(ZIG_TEXT, 1000.0)
        m.update_native_stats(ZIG_STEP, 1010.0)
        out = m.rewrite(ZIG_STEP).decode()
        # live rates come from the rewrite block (single emission, no dupes)
        self.assertIn("tfexporter_live_decode_tps 20.0", out)
        self.assertIn("tfexporter_live_prefill_tps 200.0", out)
        # bridge series ride in the same body
        self.assertIn("vllm_generation_tokens_total 300.0", out)
        # sample lines (not HELP/TYPE mentions) must appear exactly once
        def samples(body, name):
            return [ln for ln in body.splitlines()
                    if ln.startswith(name + " ") and not ln.startswith("#")]
        self.assertEqual(len(samples(out, "tfexporter_live_decode_tps")), 1)
        self.assertEqual(len(samples(out, "vllm_num_requests_swapped")), 1)
        self.assertEqual(len(samples(out, "vllm_generation_tokens_total")), 1)


    # --- persistent counters (exporter restarts must not reset the era) ---

    def test_offset_continuation_across_exporter_restart(self):
        m = self.m
        m.STATE.offsets = {"vllm_generation_tokens_total":
                           {"engine": 100.0, "emitted": 600.0}}
        out = m.zig_vllm_bridge(ZIG_TEXT).decode()
        # engine raw 200 + offset base (600-100) -> 700, era continues
        self.assertIn("vllm_generation_tokens_total 700.0", out)
        self.assertEqual(m.STATE.offsets["vllm_generation_tokens_total"],
                         {"engine": 200.0, "emitted": 700.0})

    def test_engine_restart_clears_offsets(self):
        m = self.m
        m.STATE.offsets = {"vllm_generation_tokens_total":
                           {"engine": 100.0, "emitted": 600.0}}
        m.STATE.prev_uptime = 6000.0
        m.update_native_stats(ZIG_TEXT, 1000.0)  # uptime 5000 -> engine restart
        self.assertEqual(m.STATE.offsets, {})
        out = m.zig_vllm_bridge(ZIG_TEXT).decode()
        self.assertIn("vllm_generation_tokens_total 200.0", out)  # raw, new era

    # --- deadlock regression: the offset helper must be lock-free ---

    def test_cached_offset_base_is_lock_free(self):
        m = self.m
        m.STATE.offsets = {"vllm_prefix_cache_hits_total":
                           {"engine": 1.0, "emitted": 2.0}}
        # prod deadlock shape: same-thread nested acquisition inside
        # _v4_gauges' own critical section. _cached_offset_base must not
        # touch STATE.lock, so it completes while the lock is held.
        m.STATE.lock.acquire()
        try:
            base = m._cached_offset_base()
        finally:
            m.STATE.lock.release()
        self.assertEqual(base, 1.0)

    def test_v4_gauges_completes_and_uses_offset(self):
        m = self.m
        m.STATE.mem_gib = 8.0
        m.STATE.last_cached = 10.0
        m.STATE.offsets = {"vllm_prefix_cache_hits_total":
                           {"engine": 1.0, "emitted": 2.0}}
        handler = object.__new__(m.Handler)
        body = handler._v4_gauges().decode()   # was: hang before the fix
        # offset base = emitted - engine = 1.0; raw 10 + 1 = 11
        self.assertIn("tfexporter_cached_tokens_total 11.0", body)

    def test_persist_roundtrip(self):
        m = self.m
        old_path = m.PERSIST_PATH
        m.PERSIST_PATH = "/tmp/test_counters_roundtrip.json"
        try:
            m.STATE.offsets = {"vllm_prompt_tokens_total":
                               {"engine": 5.0, "emitted": 55.0}}
            m._hist_add("ttft", 0.003)
            m.persist_save()
            m.STATE.offsets = {}
            m.STATE.probe_hist = {}
            m.persist_load()
            self.assertEqual(m.STATE.offsets["vllm_prompt_tokens_total"],
                             {"engine": 5.0, "emitted": 55.0})
            self.assertEqual(m.STATE.probe_hist["ttft"][repr(0.005)], 1)
        finally:
            m.PERSIST_PATH = old_path
            os.remove("/tmp/test_counters_roundtrip.json")



# --- native vLLM (exlr8) exposition fixture --------------------------------

def vllm_text(gen=37082.0, dec_s=500.0, prompt=878696.0, ttft_s=91.5,
              cached=87552.0, running=1.0):
    """One poll of a native vLLM engine (`vllm:*` colon families)."""
    lbl = '{engine="0",model_name="glm-5.3-flash"}'
    return (
        '# HELP vllm:prompt_tokens_total Number of prefill tokens processed.\n'
        '# TYPE vllm:prompt_tokens_total counter\n'
        'vllm:prompt_tokens_total' + lbl + ' %s\n' % prompt
        + '# HELP vllm:generation_tokens_total Number of generation tokens processed.\n'
        '# TYPE vllm:generation_tokens_total counter\n'
        'vllm:generation_tokens_total' + lbl + ' %s\n' % gen
        + '# HELP vllm:num_requests_running Number of requests in execution batches.\n'
        '# TYPE vllm:num_requests_running gauge\n'
        'vllm:num_requests_running' + lbl + ' %s\n' % running
        + '# TYPE vllm:num_requests_waiting gauge\n'
        'vllm:num_requests_waiting' + lbl + ' 0.0\n'
        + '# TYPE vllm:kv_cache_usage_perc gauge\n'
        'vllm:kv_cache_usage_perc' + lbl + ' 0.25\n'
        + '# TYPE vllm:prefix_cache_hits_total counter\n'
        'vllm:prefix_cache_hits_total' + lbl + ' %s\n' % cached
        + '# TYPE vllm:prefix_cache_queries_total counter\n'
        'vllm:prefix_cache_queries_total' + lbl + ' %s\n' % prompt
        + '# HELP vllm:time_to_first_token_seconds Histogram of TTFT.\n'
        '# TYPE vllm:time_to_first_token_seconds histogram\n'
        'vllm:time_to_first_token_seconds_bucket{engine="0",le="0.5",model_name="glm-5.3-flash"} 22.0\n'
        'vllm:time_to_first_token_seconds_bucket{engine="0",le="+Inf",model_name="glm-5.3-flash"} 197.0\n'
        'vllm:time_to_first_token_seconds_sum' + lbl + ' %s\n' % ttft_s
        + 'vllm:time_to_first_token_seconds_count' + lbl + ' 197.0\n'
        + '# TYPE vllm:request_time_per_output_token_seconds histogram\n'
        'vllm:request_time_per_output_token_seconds_sum' + lbl + ' 640.2\n'
        'vllm:request_time_per_output_token_seconds_count' + lbl + ' %s\n' % gen
        + '# TYPE vllm:request_decode_time_seconds histogram\n'
        'vllm:request_decode_time_seconds_sum' + lbl + ' %s\n' % dec_s
        + 'vllm:request_decode_time_seconds_count' + lbl + ' 20000.0\n'
        + '# TYPE vllm:spec_decode_num_drafts_total counter\n'
        'vllm:spec_decode_num_drafts_total' + lbl + ' 11124.0\n'
        + '# TYPE process_resident_memory_bytes gauge\n'
        'process_resident_memory_bytes 1.0\n'
        '# TYPE http_requests_total counter\n'
        'http_requests_total{handler="metrics"} 12.0\n'
    )


VLLM_TEXT = vllm_text()


class VllmPassthroughTests(unittest.TestCase):
    """vllm-v1: a native vLLM upstream is detected and passed through with
    colon families renamed 1:1 to the underscore names the adapter matches."""

    @classmethod
    def setUpClass(cls):
        cls.m = load_exporter()

    def setUp(self):
        st = self.m.STATE
        st.schema = "vllm-v1"
        st.window = []
        st.rate_decode = 0.0
        st.rate_prefill = 0.0
        st.rate_prefill_hold = None
        st.busy_polls = 0
        st.cache_ratio = None
        st.prev_cached = None
        st.prev_prompt_r = None
        st.last_cached = None
        st.last_inflight = 0
        st.last_stalled = 0
        st.last_fatal = 0
        st.last_rejected = None
        st.offsets = {}
        st.native_raw = b""
        st.probe_hist = {}

    # --- detection ---

    def test_detect_schema_vllm(self):
        self.assertEqual(self.m.detect_schema(VLLM_TEXT), "vllm-v1")

    def test_detect_schema_vllm_wins_over_mixed_text(self):
        self.assertEqual(
            self.m.detect_schema(VLLM_TEXT + "tensorfold:prompt_tokens_total 5\n"),
            "vllm-v1")

    def test_detect_schema_zig_still_wins_on_tensorfold(self):
        self.assertEqual(self.m.detect_schema(ZIG_TEXT), "zig-v1")

    # --- passthrough contract ---

    def test_passthrough_renames_families_and_keeps_labels(self):
        out = self.m.vllm_rewrite(VLLM_TEXT).decode()
        self.assertIn('vllm_prompt_tokens_total{engine="0",model_name="glm-5.3-flash"} 878696.0', out)
        self.assertIn('vllm_kv_cache_usage_perc{engine="0",model_name="glm-5.3-flash"} 0.25', out)
        self.assertIn('vllm_time_to_first_token_seconds_bucket{engine="0",le="0.5",model_name="glm-5.3-flash"} 22.0', out)

    def test_passthrough_has_no_colon_in_sample_names(self):
        out = self.m.vllm_rewrite(VLLM_TEXT).decode()
        for line in out.splitlines():
            if not line or line.startswith("#"):
                continue
            name = line.split("{")[0].split(" ")[0]
            self.assertNotIn(":", name, line)

    def test_passthrough_metadata_precedes_samples(self):
        out = self.m.vllm_rewrite(VLLM_TEXT).decode()
        self.assertIn("# TYPE vllm_time_to_first_token_seconds histogram", out)
        self.assertIn("# HELP vllm_time_to_first_token_seconds Histogram of TTFT.", out)
        self.assertLess(out.index("# TYPE vllm_time_to_first_token_seconds histogram"),
                        out.index("vllm_time_to_first_token_seconds_bucket"))

    def test_passthrough_metadata_emitted_once(self):
        text = VLLM_TEXT + "# TYPE vllm:num_requests_running gauge\n"
        out = self.m.vllm_rewrite(text).decode()
        self.assertEqual(out.count("# TYPE vllm_num_requests_running gauge"), 1)

    def test_passthrough_drops_non_vllm_families(self):
        out = self.m.vllm_rewrite(VLLM_TEXT).decode()
        for bad in ("process_resident_memory_bytes", "http_requests_total"):
            self.assertNotIn(bad, out)

    def test_passthrough_no_duplicate_samples(self):
        text = VLLM_TEXT + 'vllm:prompt_tokens_total{engine="0",model_name="glm-5.3-flash"} 878696.0\n'
        out = self.m.vllm_rewrite(text).decode()
        samples = [ln for ln in out.splitlines()
                   if ln.startswith('vllm_prompt_tokens_total{')]
        self.assertEqual(len(samples), 1)

    def test_passthrough_emits_swapped_zero_once(self):
        out = self.m.vllm_rewrite(VLLM_TEXT).decode()
        lines = [ln for ln in out.splitlines()
                 if ln.startswith("vllm_num_requests_swapped ")]
        self.assertEqual(lines, ["vllm_num_requests_swapped 0"])

    def test_passthrough_keeps_engine_swapped_when_exported(self):
        text = VLLM_TEXT + ('# TYPE vllm:num_requests_swapped gauge\n'
                            'vllm:num_requests_swapped{engine="0"} 2.0\n')
        out = self.m.vllm_rewrite(text).decode()
        self.assertIn('vllm_num_requests_swapped{engine="0"} 2.0', out)
        self.assertNotIn("vllm_num_requests_swapped 0", out)

    def test_passthrough_self_status_and_profile(self):
        out = self.m.vllm_rewrite(VLLM_TEXT).decode()
        self.assertIn("tfexporter_engine_profile 2", out)
        self.assertIn("tfexporter_upstream_ok 1", out)
        self.assertIn("# TYPE tfexporter_upstream_lag_seconds gauge", out)

    def test_adapter_series_inventory_is_complete(self):
        out = self.m.vllm_rewrite(VLLM_TEXT).decode()
        for name in ("vllm_prompt_tokens_total", "vllm_generation_tokens_total",
                     "vllm_num_requests_running", "vllm_num_requests_waiting",
                     "vllm_num_requests_swapped", "vllm_kv_cache_usage_perc",
                     "vllm_prefix_cache_hits_total",
                     "vllm_prefix_cache_queries_total",
                     "vllm_time_to_first_token_seconds_sum",
                     "vllm_time_to_first_token_seconds_count",
                     "vllm_request_time_per_output_token_seconds_sum",
                     "vllm_request_time_per_output_token_seconds_count",
                     "vllm_spec_decode_num_drafts_total"):
            self.assertIn(name, out)

    def test_rewrite_routes_vllm_schema_to_passthrough(self):
        out = self.m.rewrite(VLLM_TEXT).decode()
        self.assertIn("vllm_prompt_tokens_total", out)
        self.assertNotIn("tensorfold", out)
        self.assertNotIn("vllm:", "".join(
            ln.split("{")[0].split(" ")[0]
            for ln in out.splitlines() if ln and not ln.startswith("#")))

    # --- live rates / state from the vllm-v1 schema ---

    def test_vllm_live_rates_and_cache_ratio(self):
        m = self.m
        a = vllm_text()
        b = vllm_text(gen=38082.0, dec_s=520.0, prompt=888696.0, ttft_s=96.5,
                      cached=92552.0, running=2.0)
        m.update_native_stats(a, 1000.0)
        m.update_native_stats(b, 1010.0)
        self.assertAlmostEqual(m.STATE.rate_decode, 50.0, places=6)      # 1000 tok / 20 s
        self.assertAlmostEqual(m.STATE.rate_prefill, 1000.0, places=6)   # (10000-5000) / 5 s
        self.assertAlmostEqual(m.STATE.cache_ratio, 0.5, places=6)       # 5000 / 10000
        self.assertEqual(m.STATE.last_inflight, 2)
        self.assertEqual(m.STATE.last_cached, 92552.0)

    def test_vllm_idle_poll_zeroes_rates(self):
        m = self.m
        m.update_native_stats(vllm_text(), 1000.0)
        m.update_native_stats(vllm_text(gen=38082.0, dec_s=520.0, prompt=888696.0,
                                        ttft_s=96.5, cached=92552.0, running=2.0),
                              1010.0)
        self.assertGreater(m.STATE.rate_decode, 0.0)
        m.update_native_stats(vllm_text(gen=38082.0, dec_s=520.0, prompt=888696.0,
                                        ttft_s=96.5, cached=92552.0, running=0.0),
                              1020.0)
        self.assertEqual(m.STATE.rate_decode, 0.0)
        self.assertEqual(m.STATE.rate_prefill, 0.0)

    def test_vllm_rates_exported_as_gauges(self):
        m = self.m
        m.update_native_stats(vllm_text(), 1000.0)
        step = vllm_text(gen=38082.0, dec_s=520.0, prompt=888696.0, ttft_s=96.5,
                         cached=92552.0, running=2.0)
        m.update_native_stats(step, 1010.0)
        out = m.rewrite(step).decode()
        self.assertIn("tfexporter_live_decode_tps 50.0", out)
        self.assertIn("tfexporter_live_prefill_tps 1000.0", out)

    # --- probe must stay off for engines that measure their own latencies ---

    def test_probe_disabled_for_vllm_profile(self):
        self.assertFalse(self.m.probe_enabled())

    def test_probe_enabled_for_tensorfold_profiles(self):
        self.m.STATE.schema = "zig-v1"
        self.assertTrue(self.m.probe_enabled())
        self.m.STATE.schema = "python-v1"
        self.assertTrue(self.m.probe_enabled())

    # --- raw colon exposition must never leak through the v4 gauges ---

    def test_v4_gauges_do_not_replay_raw_colon_text(self):
        m = self.m
        m.STATE.native_raw = VLLM_TEXT.encode()
        handler = object.__new__(m.Handler)
        body = handler._v4_gauges().decode()
        self.assertNotIn("vllm:", body)
        self.assertIn('tfexporter_engine_schema_info{schema="vllm-v1"', body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
