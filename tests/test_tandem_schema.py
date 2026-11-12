"""tandem-v1 schema tests (v1.7.0): TandemLLM qse_* -> dashboard vllm_* set.

Hand-computed fixture:
  fresh 1000 | served 5000 | cached 4000 (identity 5000-1000=4000)
  gen finished 500 | live slot growth 60 -> emitted max(500,60)=500
  second health step pushes live to 900 -> emitted 900 (live > finished)
  kv ratio = (1000+2000)/(8 x 262144) = 0.0014
  ttft 20.0/40 | e2e 100.0/40 | tpot 2.5/500 | queue 8.0/40
  spec accept 720/1200 = 0.6 | verify steps 120 | prefill mean 1100 | decode mean 50
"""
import importlib.util
import json
import os
import unittest

EXPORTER_PATH = os.environ.get("TF_EXPORTER_PATH",
                               "/home/admin/llm/tensorfold-metrics-exporter/exporter.py")


def load_exporter():
    spec = importlib.util.spec_from_file_location("tf_exporter_under_test", EXPORTER_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def qse(text_lines):
    return "\n".join(text_lines) + "\n"


def qse_text(with_cached_total=True):
    lines = [
        "qse_requests_running 2",
        "qse_requests_waiting 1",
        "qse_prefill_tokens_forwarded_total 1000",
        "qse_prompt_tokens_total 5000",
        "qse_generation_tokens_total 500",
        "qse_request_success_total 40",
        "qse_time_to_first_token_seconds_sum 20.0",
        "qse_time_to_first_token_seconds_count 40",
        "qse_e2e_request_latency_seconds_sum 100.0",
        "qse_e2e_request_latency_seconds_count 40",
        "qse_time_per_output_token_seconds_sum 2.5",
        "qse_time_per_output_token_seconds_count 500",
        "qse_request_queue_seconds_sum 8.0",
        "qse_request_queue_seconds_count 40",
        "qse_spec_decode_num_drafts_total 100",
        "qse_spec_decode_num_draft_tokens_total 1200",
        "qse_spec_decode_num_accepted_tokens_total 720",
        "qse_verify_seconds_sum 30.0",
        "qse_verify_seconds_count 120",
        "qse_request_prefill_tokens_per_second_sum 44000.0",
        "qse_request_prefill_tokens_per_second_count 40",
        "qse_request_decode_tokens_per_second_sum 2000.0",
        "qse_request_decode_tokens_per_second_count 40",
        "qse_reasoning_tokens_total 220",
        'qse_cache_hits_total{kind="prefix"} 12',
        'qse_cache_hits_total{kind="session"} 5',
    ]
    if with_cached_total:
        lines.insert(3, "qse_prompt_tokens_cached_total 4000")
    return qse(lines)


def health(slots):
    """slots: list of (phase, kv_length) for 8 lanes; missing -> idle."""
    full = [{"phase": "idle", "kv_length": 0, "max_len": 262144} for _ in range(8)]
    for i, (ph, kv) in enumerate(slots):
        full[i] = {"phase": ph, "kv_length": kv, "max_len": 262144}
    return json.dumps({"parallel": {"n": 8, "slots": full}}).encode()


TANDEM_ATTRS = ("tandem", "tandem_cache_prefix_hits", "tandem_up",
                "tandem_health", "tandem_kv_ratio", "tandem_live_gen",
                "tandem_live_rate", "tandem_prev_slot_kv", "tandem_prev_dec_kv",
                "tandem_prev_ts", "tandem_fresh",
                "tandem_hits")


class TandemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_exporter()

    def setUp(self):
        st = self.m.STATE
        st.schema = "tandem-v1"
        st.window = []
        st.rate_decode = 0.0
        st.rate_prefill = 0.0
        st.cache_ratio = None
        st.prev_cached = None
        st.prev_prompt_r = None
        st.restarts = 0
        st.prev_uptime = None
        st.probe_hist = {}
        st.probe = {"ttft": [0.0, 0], "e2e": [0.0, 0], "itl": [0.0, 0]}
        st.rate_prefill_hold = None
        st.last_cached = None
        st.offsets = {}
        st.extras = {}
        st.health_info = {}
        for a in TANDEM_ATTRS:
            setattr(st, a, None)
        st.tandem_live_gen = 0.0
        st.tandem_live_rate = 0.0

    def scrape(self):
        rows = self.m.engine_native_series(set())
        out = []
        for head, a, b, c in rows:
            if head == "__sample__":
                out.append(a)
            elif head != "__meta__":
                out.append("%s %s" % (head, a))
        return out

    def value(self, lines, name):
        for l in lines:
            if l.split(" ", 1)[0].split("{", 1)[0] == name:
                return l.rsplit(" ", 1)[1]
        return None

    def collect(self, with_cached_total=True):
        self.m.update_native_stats(qse_text(with_cached_total), now=1000.0)

    def feed_health(self, snaps):
        """Drive update_tandem_health with canned /health payloads, 10 s apart."""
        seq = list(snaps)
        idx = {"i": 0}

        def fake_fetch(path):
            body = seq[idx["i"]]
            idx["i"] += 1
            return 200, body, "application/json"

        old = self.m.fetch_upstream
        self.m.fetch_upstream = fake_fetch
        try:
            t = 1000.0
            for _ in seq:
                self.m.update_tandem_health(t)
                t += 10.0
        finally:
            self.m.fetch_upstream = old

    # ------------------------------------------------------------------ tests
    def test_detection_and_probe_denied(self):
        self.assertEqual(self.m.detect_schema(qse_text()), "tandem-v1")
        self.m.STATE.schema = "tandem-v1"
        self.assertFalse(self.m.probe_enabled())

    def test_collection_and_identity(self):
        self.collect()
        st = self.m.STATE
        self.assertEqual(st.tandem["qse_prefill_tokens_forwarded_total"], 1000.0)
        self.assertEqual(st.tandem_cache_prefix_hits, 4000.0)  # token counter wins
        lines = self.scrape()
        q = float(self.value(lines, "vllm_prefix_cache_queries_total"))
        p = float(self.value(lines, "vllm_prompt_tokens_total"))
        h = float(self.value(lines, "vllm_prefix_cache_hits_total"))
        self.assertEqual((p, q, h), (1000.0, 5000.0, 4000.0))
        self.assertEqual(q - p, h)  # skill identity: queries - prompt = hits

    def test_hits_fallback_prefix_only(self):
        self.collect(with_cached_total=False)
        lines = self.scrape()
        self.assertEqual(self.value(lines, "vllm_prefix_cache_hits_total"), "12")

    def test_fresh_clamped_to_served(self):
        # forwarded (6000, with padding) must NEVER leak into the prompt
        # counter: cold traffic (cached 0) emits prompt = served = 5000
        text = qse_text().replace("qse_prompt_tokens_cached_total 4000",
                                  "qse_prompt_tokens_cached_total 0").replace(
            "qse_prefill_tokens_forwarded_total 1000",
            "qse_prefill_tokens_forwarded_total 6000")
        self.m.update_native_stats(text, now=1000.0)
        lines = self.scrape()
        self.assertEqual(self.value(lines, "vllm_prompt_tokens_total"), "5000")
        self.assertEqual(self.value(lines, "vllm_prefix_cache_queries_total"), "5000")
        self.assertEqual(self.value(lines, "vllm_prefix_cache_hits_total"), "0")

    def test_generation_hybrid_finished_then_live(self):
        self.collect()
        # no live feed yet -> finished counter (500) is what shows
        lines = self.scrape()
        self.assertEqual(self.value(lines, "vllm_generation_tokens_total"), "500")
        # one decoding slot grows 60 tokens across two polls: live=60 < 500
        self.feed_health([health([("decode", 100)]), health([("decode", 160)])])
        lines = self.scrape()
        self.assertEqual(self.value(lines, "vllm_generation_tokens_total"), "500")
        self.assertEqual(self.m.STATE.tandem_live_gen, 60.0)
        # push live past finished: emitted must follow the live sum
        # (deltas: 160->1000 = 840, live total 60+840 = 900 > finished 500)
        self.feed_health([health([("decode", 160)]), health([("decode", 1000)])])
        lines = self.scrape()
        self.assertEqual(self.value(lines, "vllm_generation_tokens_total"), "900")

    def test_kv_ratio_uses_all_lanes(self):
        self.collect()
        self.feed_health([health([("decode", 1000), ("decode", 2000)]),
                          health([("decode", 1000), ("decode", 2000)])])
        lines = self.scrape()
        expected = 3000.0 / (8 * 262144)
        self.assertEqual(self.value(lines, "vllm_kv_cache_usage_perc"), f"{expected:.4f}")
        self.assertEqual(self.value(lines, "vllm_gpu_cache_usage_perc"), f"{expected:.4f}")
        v = float(self.value(lines, "vllm_kv_cache_usage_perc"))
        self.assertTrue(0.0 < v < 1.0, "must be a 0..1 ratio, not a percent")

    def test_documented_zeros(self):
        self.collect()
        lines = self.scrape()
        self.assertEqual(self.value(lines, "vllm_num_requests_swapped"), "0")
        self.assertEqual(self.value(lines, "vllm_num_preemptions_total"), "0")

    def test_gauges_and_histogram_pairs(self):
        self.collect()
        lines = self.scrape()
        self.assertEqual(self.value(lines, "vllm_num_requests_running"), "2")
        self.assertEqual(self.value(lines, "vllm_num_requests_waiting"), "1")
        self.assertEqual(self.value(lines, "vllm_request_success_total"), "40")
        self.assertEqual(self.value(lines, "vllm_time_to_first_token_seconds_sum"), "20.000000")
        self.assertEqual(self.value(lines, "vllm_time_to_first_token_seconds_count"), "40")
        self.assertEqual(self.value(lines, "vllm_e2e_request_latency_seconds_count"), "40")
        self.assertEqual(self.value(lines, "vllm_request_queue_time_seconds_sum"), "8.000000")
        self.assertEqual(self.value(lines, "vllm_request_queue_time_seconds_count"), "40")
        # three TPOT aliases over the SAME measured population
        for n in ("vllm_request_time_per_output_token_seconds",
                  "vllm_time_per_output_token_seconds",
                  "vllm_inter_token_latency_seconds"):
            self.assertEqual(self.value(lines, n + "_sum"), "2.500000", n)
            self.assertEqual(self.value(lines, n + "_count"), "500", n)

    def test_spec_and_batch_tile(self):
        self.collect()
        lines = self.scrape()
        self.assertEqual(self.value(lines, "vllm_spec_decode_num_drafts_total"), "100")
        self.assertEqual(self.value(lines, "vllm_spec_decode_num_draft_tokens_total"), "1200")
        self.assertEqual(self.value(lines, "vllm_spec_decode_num_accepted_tokens_total"), "720")
        self.assertEqual(self.value(lines, "tfexporter_tandem_spec_accept_ratio"), "0.6000")
        self.assertEqual(self.value(lines, "tfexporter_tandem_first_draft_accept_pct"), "60.00")
        # Batch tile: sum = generation population, count = verify rounds
        self.assertEqual(self.value(lines, "vllm_iteration_tokens_total_count"), "120")
        self.assertEqual(self.value(lines, "vllm_iteration_tokens_total_sum"), "500")

    def test_diagnostic_gauges(self):
        self.collect()
        self.feed_health([health([("decode", 100)]), health([("decode", 160)])])
        lines = self.scrape()
        self.assertEqual(self.value(lines, "tfexporter_tandem_prefill_tps_mean"), "1100.0")
        self.assertEqual(self.value(lines, "tfexporter_tandem_decode_tps_mean"), "50.0")
        self.assertEqual(self.value(lines, "tfexporter_tandem_reasoning_tokens_total"), "220")
        self.assertGreater(float(self.value(lines, "tfexporter_tandem_live_decode_tps")), 0.0)

    def test_persisted_value_stable_across_repeat_scrapes(self):
        self.collect()
        a = self.value(self.scrape(), "vllm_prompt_tokens_total")
        b = self.value(self.scrape(), "vllm_prompt_tokens_total")
        c = self.value(self.scrape(), "vllm_prompt_tokens_total")
        self.assertEqual((a, b, c), ("1000", "1000", "1000"))

    def test_other_schemas_unaffected(self):
        # a flare exposition must not be re-labelled by the tandem branch
        self.m.STATE.schema = "flare-v1"
        self.m.STATE.tandem = {"qse_prompt_tokens_total": 999999.0}
        lines = self.scrape()
        self.assertIsNone(self.value(lines, "vllm_prefix_cache_queries_total"))


if __name__ == "__main__":
    unittest.main()
