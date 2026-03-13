# Benchmark Workflow

This repo now supports a repeatable benchmark workflow for saved benchmark suites, Google Sheets export, plotting, and a browser-friendly report.

## Core Pieces

- Suite runner: `benchmarker.py`
- Saved suites: `configs/suites/*.yaml`
- Latest example suite: `configs/suites/march_2026_agentic_model_sweep_for_airweave.yaml`
- Plot generator: `scripts/plot_airweave_track_a_suite.py`
- Latest report outputs:
  - `docs/airweave_track_a_model_report_2026_03.md`
  - `docs/airweave_track_a_model_report_2026_03.html`

## Reusable Workflow

### 1. Inspect saved suites

```bash
/Users/kpirestani/.venvs/kayvon-bench/bin/python benchmarker.py --list-suites
```

### 2. Preview a suite before running

```bash
/Users/kpirestani/.venvs/kayvon-bench/bin/python benchmarker.py --describe-suite <suite_slug>
```

### 3. Run the suite and export results to Google Sheets

```bash
/Users/kpirestani/.venvs/kayvon-bench/bin/python benchmarker.py --suite <suite_slug>
```

Notes:

- `GOOGLE_SERVICE_ACCOUNT_JSON` must be present in `.env`.
- Successful export appends rows to worksheet `Summary`.

### 4. Generate comparison plots

For the current Airweave Track A workflow:

```bash
/Users/kpirestani/.venvs/kayvon-bench/bin/python scripts/plot_airweave_track_a_suite.py
```

Outputs are written to:

- `figures/airweave_track_a_suite/`

### 5. View the report

Preferred:

- open `docs/*.html` in Safari or Chrome

Fallback:

- use the Markdown version in editor preview

## When Starting a New Benchmark Campaign

1. Copy an existing suite YAML in `configs/suites/`
2. Update:
   - model endpoints
   - benchmark profile names
   - run list
   - sample counts / concurrency
3. Run the suite
4. Build plots from the new summary rows
5. Create a matching HTML report in `docs/`

## Notes For Future Codex Sessions

- The latest benchmark reporting flow is already encoded in code and docs; future sessions should prefer reusing the suite + plot-script pattern instead of rebuilding ad hoc notebooks.
- If pricing is needed for a new report, verify Together public serverless pricing once and record the assumption in the report.
- If a model is benchmarked on dedicated infrastructure but needs a public-pricing comparison, note that explicitly in the report.
