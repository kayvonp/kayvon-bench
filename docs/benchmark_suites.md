# Benchmark Suites

Benchmark suites let you save a whole benchmark campaign under one reusable name.

Each suite lives in `configs/suites/*.yaml` and can snapshot:

- the LLM profiles and endpoint strings
- the benchmark profiles and corpora
- the exact run order for a session

## Commands

List saved suites:

```bash
/Users/kpirestani/.venvs/kayvon-bench/bin/python benchmarker.py --list-suites
```

Describe a suite without running it:

```bash
/Users/kpirestani/.venvs/kayvon-bench/bin/python benchmarker.py --describe-suite march_2026_agentic_model_sweep_for_airweave
```

Run a full suite:

```bash
/Users/kpirestani/.venvs/kayvon-bench/bin/python benchmarker.py --suite march_2026_agentic_model_sweep_for_airweave
```

## Current Saved Suite

- `march_2026_agentic_model_sweep_for_airweave`
  - Label: `March 2026 agentic model sweep for Airweave`
  - Purpose: reusable Together sweep covering simple probe, grounded extraction, and Track A retrieval workloads
