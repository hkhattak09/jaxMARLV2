#!/usr/bin/env python
"""Run all test suites for the stateful CTM + recurrent critic implementation.

Usage:
    cd MARL-LLM/marl_llm
    python tests/run_all_tests.py            # run all tests
    python tests/run_all_tests.py -v         # verbose
    python tests/run_all_tests.py -k burn    # filter by keyword

All arguments are forwarded to pytest.
"""

import sys
from pathlib import Path

# Ensure MARL-LLM/marl_llm is on sys.path (same as individual test files)
_THIS_DIR = Path(__file__).resolve().parent
_MARL_LLM = _THIS_DIR.parent
if str(_MARL_LLM) not in sys.path:
    sys.path.insert(0, str(_MARL_LLM))

import pytest

# Ordered by implementation phase
TEST_MODULES = [
    "tests/test_ctm_implementation.py",      # Pre-existing CTM tests
    "tests/test_episode_buffer.py",           # Phase 1: Episode-sequence buffer
    "tests/test_recurrent_critic.py",         # Phase 2: LSTM in AggregatingCritic
    # Phase 3: Config — no dedicated tests (argparse declarations)
    "tests/test_sequence_update.py",          # Phase 4: update_sequence with burn-in
    "tests/test_agent_lstm_param.py",         # Phase 5: lstm_hidden_dim threading
    "tests/test_stateful_rollout.py",         # Phase 6: Stateful training rollout
    "tests/test_stateful_eval.py",            # Phase 7: Stateful eval rollout
    "tests/test_gradient_hotpaths.py",        # Gradient flow & critical hot paths
]


def main():
    # Resolve paths relative to MARL-LLM/marl_llm
    test_paths = [str(_MARL_LLM / mod) for mod in TEST_MODULES]

    # Forward any CLI args (e.g. -v, -k, --tb=short) to pytest
    extra_args = sys.argv[1:]
    if not any(a.startswith("-v") for a in extra_args):
        extra_args.append("-v")

    exit_code = pytest.main(test_paths + extra_args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
