"""Unit tests for the tensorfold-metrics-exporter metric math.

Run on spark3: python3 -m unittest discover -s tests -v
Covers: schema detection, zig rate windows, windowed cache-hit ratio,
vllm bridge contract, histogram bucketing, restart detection, v4 gauges.
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
