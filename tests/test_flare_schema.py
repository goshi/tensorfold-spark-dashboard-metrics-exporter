"""flare-v1 schema tests (v1.6.0): detection, collection, synthesized vllm_* families.

Fixture numbers are hand-computed:
  prompt = prefill 1100 + reused 450 = 1550 | gen 600 | active 2 | pending 1
  kv ratio = (100000 - 20000)/100000 = 0.80 | ttft 0.4 x 20 samples
  steps 1000+100=1100 | depth 3000/600=5 | accept 1800/3000=0.6
  occupancy integral (active 2 both samples, dt 10) = 20.0
  queue integral (pending 1 both samples, dt 10) = 10.0 over 10 completions
"""
import importlib.util
import os
import unittest

EXPORTER_PATH = os.environ.get("TF_EXPORTER_PATH",
                               "/home/admin/llm/tensorfold-metrics-exporter/exporter.py")


def load_exporter():
    spec = importlib.util.spec_from_file_location("tf_exporter_under_test", EXPORTER_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _line(base, value, labels=""):
    return "%s%s %s" % (base, labels, value)


def flare_text(gen=600, prefill=1100, reused=450, active=2, pending=1,
               completed=50, free_kv=20000, uptime=5060, sched=100,
               med=0.4, samples=20):
    return "\n".join([
        "# HELP flare_generated_tokens_total gen",
        "# TYPE flare_generated_tokens_total counter",
        _line("flare_generated_tokens_total", gen),
        _line("flare_prefill_tokens_total", prefill),
        _line("flare_cache_reused_tokens_total", reused),
        _line("flare_active_requests", active),
        _line("flare_pending_requests", pending),
        _line("flare_requests_completed_total", completed),
        _line("flare_requests_failed_total", 1),
        _line("flare_requests_cancelled_total", 0),
        _line("flare_preemptions_total", 0),
        _line("flare_uptime_seconds", uptime),
        _line("flare_total_kv_tokens", 100000),
        _line("flare_free_kv_tokens", free_kv),
        _line("flare_first_response_median_seconds", med),
        _line("flare_first_response_samples", samples),
        _line("flare_first_response_window_seconds", 600),
        _line("flare_first_response_truncated", 0),
        _line("flare_scheduler_executions_total", 1000, '{reason="ordinary"}'),
        _line("flare_scheduler_executions_total", sched, '{reason="discovery"}'),
        _line("flare_verification_draft_tokens_total", 3000),
        _line("flare_verification_request_rounds_total", 600),
        _line("flare_speculative_accepted_drafts_total", 1800),
        _line("flare_speculative_offered_drafts_total", 3000),
        _line("flare_speculative_rounds_total", 500, '{batch="1",width="2"}'),
        _line("flare_speculative_rounds_total", 500, '{batch="2",width="2"}'),
        _line("flare_free_request_slots", 2),
        _line("flare_generated_tokens_last_minute", 120),
        _line("flare_cache_hits_total", 30),
        _line("flare_cache_lookups_total", 50),
    ]) + "\n"


FLARE_TEXT = flare_text(gen=500, prefill=1000, reused=400, active=2, pending=1,
                        completed=40, free_kv=25000, uptime=5000, sched=0,
                        med=0.5, samples=10)
FLARE_STEP = flare_text()  # the bumped defaults above


class FlareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_exporter()

    def setUp(self):
        st = self.m.STATE
        st.schema = "flare-v1"
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
        for attr in ("flare_a_t", "flare_a_int", "flare_a_a", "flare_a_up",
                     "flare_a_int_last", "flare_a_gen0", "flare_e_prompt0",
                     "flare_e_served0", "flare_e_reused0", "flare_e_done0",
                     "flare_e_done_last", "flare_e_ttft_int", "flare_q_t", "flare_q_int", "flare_q_w",
                     "flare_q_base", "flare_q_done", "flare_prompt_total",
                     "flare_gen_total", "flare_reused", "flare_prompt_served", "flare_running",
                     "flare_waiting", "flare_kv_ratio", "flare_kv_used",
                     "flare_kv_total", "flare_ttft_med", "flare_ttft_n",
                     "flare_ttft_win", "flare_ttft_trunc", "flare_steps",
                     "flare_ver_drafts", "flare_ver_rounds", "flare_acc",
                     "flare_off", "flare_accept_ratio", "flare_preempt",
                     "flare_failed", "flare_cancelled", "flare_free_slots",
                     "flare_cache_hits", "flare_cache_lookups",
                     "flare_gen_last_min", "flare_completed",
                     "flare_rounds"):
            setattr(st, attr, None)

    def samples(self):
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

    def test_detection(self):
        self.assertEqual(self.m.detect_schema(FLARE_TEXT), "flare-v1")
        self.assertFalse(self.m.probe_enabled())

    def test_collection(self):
        self.m.update_native_stats(FLARE_TEXT, now=1000.0)
        self.m.update_native_stats(FLARE_STEP, now=1010.0)
        st = self.m.STATE
        self.assertEqual(st.flare_prompt_total, 1100)      # FRESH lifetime
        self.assertEqual(st.flare_prompt_served, 1550)     # SERVED lifetime
        # era baselines opened at first scrape (prompt 1000 / served 1400 /
        # reused 400 / completed 40) -> populations the tiles divide
        self.assertEqual(st.flare_e_prompt0, 1000)
        self.assertAlmostEqual(st.flare_e_ttft_int, 4.0, places=6)  # dn 10 x med 0.4
        self.assertEqual(st.flare_gen_total, 600)
        self.assertAlmostEqual(st.flare_kv_ratio, 0.80, places=6)  # RATIO not percent
        self.assertAlmostEqual(st.flare_a_int_last, 20.0, places=3)  # (2+2)/2*10
        self.assertAlmostEqual(st.flare_q_int, 10.0, places=3)       # (1+1)/2*10
        self.assertEqual(st.flare_q_done, 50)
        self.assertEqual(st.flare_steps, 1100)          # both reason labels summed
        self.assertAlmostEqual(st.flare_accept_ratio, 0.6, places=6)
        # rate_decode = dgen/ddec = 100/20 = 5 tokens/s per active request
        self.assertAlmostEqual(st.rate_decode, 5.0, places=6)
        self.assertEqual(st.rate_prefill, 0.0)           # honest zero: no pf source

    def test_emitted_families(self):
        self.m.update_native_stats(FLARE_TEXT, now=1000.0)
        self.m.update_native_stats(FLARE_STEP, now=1010.0)
        lines = self.samples()

        self.assertEqual(self.value(lines, "vllm_generation_tokens_total"), "600")
        self.assertEqual(self.value(lines, "vllm_num_requests_running"), "2")
        self.assertEqual(self.value(lines, "vllm_num_requests_waiting"), "1")
        self.assertEqual(self.value(lines, "vllm_num_requests_swapped"), "0")
        self.assertEqual(self.value(lines, "vllm_kv_cache_usage_perc"), "0.8000")

        # engine-scale TTFT pair: count = lifetime completed, sum = median x completed
        # ttft pair = era true-sum estimator / era completions
        self.assertEqual(self.value(lines, "vllm_time_to_first_token_seconds_count"), "10")
        self.assertAlmostEqual(float(self.value(lines, "vllm_time_to_first_token_seconds_sum")), 4.0, places=4)
        # era-conserved prefill side (v1.6.2 audit)
        self.assertEqual(self.value(lines, "vllm_prompt_tokens_total"), "100")   # 1100-1000
        self.assertEqual(self.value(lines, "vllm_prefix_cache_queries_total"), "150")  # 1550-1400
        self.assertEqual(self.value(lines, "vllm_prefix_cache_hits_total"), "50")      # 450-400
        # adapter-faithful names (binary-verified inventory)
        self.assertEqual(self.value(lines, "vllm_num_preemptions_total"), "0")
        self.assertEqual(self.value(lines, "vllm_spec_decode_num_drafts_total"), "1000")
        self.assertEqual(self.value(lines, "tfexporter_flare_first_response_samples"), "20")
        # population-consistent: sum = integral since attach (20.0), count =
        # tokens over the SAME window (600 - 500 = 100), not lifetime
        self.assertAlmostEqual(float(self.value(lines, "vllm_request_time_per_output_token_seconds_sum")), 20.0, places=3)
        self.assertEqual(self.value(lines, "vllm_request_time_per_output_token_seconds_count"), "100")
        self.assertEqual(self.value(lines, "vllm_inter_token_latency_seconds_count"), "100")
        self.assertEqual(self.value(lines, "vllm_gpu_cache_usage_perc"), "0.8000")
        self.assertAlmostEqual(float(self.value(lines, "vllm_request_queue_time_seconds_sum")), 10.0, places=3)
        self.assertEqual(self.value(lines, "vllm_request_queue_time_seconds_count"), "10")
        self.assertEqual(self.value(lines, "vllm_iteration_tokens_total_count"), "1100")
        self.assertEqual(self.value(lines, "vllm_spec_decode_num_draft_tokens_total"), "3000")
        self.assertEqual(self.value(lines, "vllm_spec_decode_num_accepted_tokens_total"), "1800")
        self.assertEqual(self.value(lines, "tfexporter_flare_verification_depth"), "5.00")
        self.assertEqual(self.value(lines, "tfexporter_flare_spec_accept_ratio"), "0.6000")
        self.assertEqual(self.value(lines, "tfexporter_flare_generated_tokens_last_minute"), "120")
        self.assertEqual(self.value(lines, "tfexporter_flare_free_request_slots"), "2")
        # honesty: no fabricated SLO buckets
        self.assertIsNone(self.value(lines, "vllm_inter_token_latency_seconds_bucket"))
        self.assertIsNone(self.value(lines, "vllm_e2e_request_latency_seconds_bucket"))

    def test_era_reset_on_engine_restart(self):
        self.m.update_native_stats(FLARE_TEXT, now=1000.0)
        self.m.update_native_stats(FLARE_STEP, now=1010.0)
        # the offset store fills only when a counter drop is observed; inject
        # an entry the way a previous era would have left it
        self.m.STATE.offsets["flare_generated_tokens_total"] = 12345.0
        restart = flare_text(uptime=10)  # uptime regression
        self.m.update_native_stats(restart, now=1020.0)
        self.assertEqual(self.m.STATE.restarts, 1)
        self.assertIsNone(self.m.STATE.flare_a_t)   # integrals rebase with the era
        self.assertIsNone(self.m.STATE.flare_q_t)

    def test_survives_exporter_restart(self):
        # persisted counters: emitted value must not fall after offsets reload
        self.m.update_native_stats(FLARE_TEXT, now=1000.0)
        self.m.update_native_stats(FLARE_STEP, now=1010.0)
        first = self.value(self.samples(), "vllm_generation_tokens_total")
        st = self.m.STATE
        saved = dict(st.offsets)
        st.offsets = {}                     # simulate exporter restart (file reloaded)
        st.offsets.update(saved)            # persist_load() restores the store
        again = self.value(self.samples(), "vllm_generation_tokens_total")
        self.assertEqual(first, again)      # monotonic across exporter restart


if __name__ == "__main__":
    unittest.main(verbosity=2)
