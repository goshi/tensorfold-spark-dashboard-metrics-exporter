#!/usr/bin/env python3
"""Smoke test for exporter.rewrite() - run with: python3 tests/test_rewrite.py

Feeds a synthetic python-v1 TensorFold exposition through the rewriter and
asserts the expected vLLM / tfexporter families come out the other side.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import exporter  # noqa: E402

SAMPLE = """# HELP tensorfold:num_requests_running Number of requests in running state
# TYPE tensorfold:num_requests_running gauge
tensorfold:num_requests_running 2
# TYPE tensorfold:num_requests_waiting gauge
tensorfold:num_requests_waiting 5
# TYPE tensorfold:preemptions_total counter
tensorfold:preemptions_total 0
# TYPE tensorfold:prompt_tokens_total counter
tensorfold:prompt_tokens_total 12345
# TYPE tensorfold:generation_tokens_total counter
tensorfold:generation_tokens_total 6789
# TYPE tensorfold:time_to_first_token_seconds histogram
tensorfold:time_to_first_token_seconds_count 10
tensorfold:time_to_first_token_seconds_sum 1.5
# TYPE tensorfold:e2e_request_latency_seconds histogram
tensorfold:e2e_request_latency_seconds_count 10
tensorfold:e2e_request_latency_seconds_sum 4.2
# TYPE tensorfold:spec_decode_num_draft_tokens_total counter
tensorfold:spec_decode_num_draft_tokens_total 100
# TYPE tensorfold:spec_decode_num_accepted_tokens_total counter
tensorfold:spec_decode_num_accepted_tokens_total 80
# TYPE tensorfold_health:decode_seconds_total counter
tensorfold_health:decode_seconds_total 12.5
# TYPE tensorfold_health:completion_tokens_total counter
tensorfold_health:completion_tokens_total 5000
# TYPE tensorfold_health:cached_tokens_total counter
tensorfold_health:cached_tokens_total 1234
# TYPE tensorfold_health:pool_tokens gauge
tensorfold_health:pool_tokens 10000
# TYPE tensorfold_health:pool_free_tokens gauge
tensorfold_health:pool_free_tokens 4000
# TYPE tensorfold_health:streams gauge
tensorfold_health:streams{state="active"} 2
# TYPE tensorfold:kv_cache_usage_ratio gauge
tensorfold:kv_cache_usage_ratio{pool="0"} 0.61
tensorfold:kv_cache_usage_perc{stream="3"} 0.55
tensorfold:kv_cache_usage_perc{stream="7"} 0.72
"""

# Deterministic pass-throughs (exact values) ...
EXPECTED = [
    "vllm_num_requests_running 2",
    "vllm_num_requests_waiting 5",
    "vllm_num_preemptions_total 0",
    "vllm_prompt_tokens_total 12345",
    "vllm_time_to_first_token_seconds_count 10",
    "vllm_time_to_first_token_seconds_sum 1.5",
    "vllm_e2e_request_latency_seconds_sum 4.2",
    "vllm_spec_decode_num_accepted_tokens_total 80",
    "vllm_spec_decode_num_draft_tokens_total 100",
    "vllm_time_per_output_token_seconds_sum 12.5",
    "vllm_time_per_output_token_seconds_count 5000",
    "vllm_prefix_cache_hits_total 1234",
    "vllm_prefix_cache_queries_total 12345",
    "vllm_kv_cache_usage_perc 0.72",
    "tfexporter_pool_usage_perc 0.6",
    "tfexporter_streams{state=",
    "tfexporter_upstream_ok 1",
    "tfexporter_generation_finished_tokens_total 6789",
]

# ... and families whose value now comes from exporter state (presence only).
EXPECTED_FAMILIES = [
    "vllm_generation_tokens_total",
    "vllm_request_time_per_output_token_seconds_sum",
    "vllm_request_time_per_output_token_seconds_count",
]


def main():
    out = exporter.rewrite(SAMPLE).decode()
    missing = [exp for exp in EXPECTED if exp not in out]
    assert not missing, f"missing families: {missing}"
    missing = [name for name in EXPECTED_FAMILIES if name not in out]
    assert not missing, f"missing state-fed families: {missing}"
    print("test_rewrite: OK -", len(EXPECTED) + len(EXPECTED_FAMILIES),
          "families verified")


if __name__ == "__main__":
    main()
