# Together Dedicated Endpoints

This project sometimes needs a Together dedicated endpoint when a serverless model is blocked, rate-limited, or too inconsistent for benchmarking.

This note captures the working path we used so the next dedicated deployment is faster and less guessy.

## Prerequisites

- Use an external Python environment, not a project-local venv.
- Recommended benchmark environment for this repo:

```bash
/Users/kayvon/.venvs/kayvon-bench/bin/python
```

- Install the runtime and export dependencies:

```bash
/Users/kayvon/.venvs/kayvon-bench/bin/pip install openai PyYAML gspread google-auth together
```

- Ensure `.env` contains:
  - `TOGETHER_API_KEY`
  - `GOOGLE_SERVICE_ACCOUNT_JSON`

## Key Lesson

The dedicated endpoint creation flow is:

1. Check whether the model supports dedicated hardware.
2. Check which hardware sizes are available.
3. Create the endpoint.
4. Wait for the endpoint state to become `STARTED`.
5. Use the returned endpoint `name` as the benchmark model ID.
6. Add an `llm_profile` and benchmark profile in `benchmarker_config.yaml`.
7. Run the benchmark with the export-capable Python environment so spreadsheet logging works.

The important detail is that the benchmark should use the endpoint `name`, not the base public model string.

## SDK Surface

The Together Python SDK exposes endpoint management under:

```python
client = together.Together(api_key=...)
client.endpoints.list_hardware(model="moonshotai/Kimi-K2.5")
client.endpoints.create(...)
client.endpoints.retrieve(endpoint_id)
client.endpoints.list(mine=True, type="dedicated")
```

The `create()` call expects:

```python
client.endpoints.create(
    model="moonshotai/Kimi-K2.5",
    hardware="8x_nvidia_h200_140gb_sxm",
    autoscaling={"min_replicas": 1, "max_replicas": 1},
    state="STARTED",
    display_name="kpirestani_e423/moonshotai/Kimi-K2.5",
)
```

For this project, `min_replicas=1` and `max_replicas=1` is a sensible default for low-load latency benchmarking.

## Hardware Discovery Example

Example query:

```python
client.endpoints.list_hardware(model="moonshotai/Kimi-K2.5")
```

For Kimi K2.5, Together reported:

- hardware: `8x_nvidia_h200_140gb_sxm`
- availability: `available`

## Provisioning Caveat

A successful `create()` call does not mean the endpoint is ready.

You must poll until:

```text
state == STARTED
```

It can remain in `PENDING` for a while.

Example polling loop:

```python
while True:
    ep = client.endpoints.retrieve(endpoint_id)
    if ep.state == "STARTED":
        break
    time.sleep(15)
```

## Wiring Into This Repo

Once the endpoint exists, add a dedicated llm profile in `benchmarker_config.yaml`:

```yaml
together_kimi_k2_5_dedicated:
  provider: together
  model: together_sso/moonshotai/Kimi-K2.5-<endpoint-suffix>
  reasoning_effort: low
  streaming_enabled: true
  temperature: 0.6
  empty_visible_stream_retries: 1
```

Then add a benchmark profile pointing to that llm profile.

The benchmark should use the dedicated endpoint `name`, for example:

```yaml
model: together_sso/moonshotai/Kimi-K2.5-1b38c3da
```

## Current Kimi Dedicated Endpoint

Current endpoint created during investigation:

- endpoint ID: `endpoint-1baa3f15-23d0-44c8-8e63-3f0ca4702515`
- endpoint name: `together_sso/moonshotai/Kimi-K2.5-1b38c3da`
- display name: `kpirestani_e423/moonshotai/Kimi-K2.5`
- hardware: `8x_nvidia_h200_140gb_sxm`

This is useful as a checkpoint, but do not assume it will always be the endpoint we want long-term.

## Running Benchmarks

Use the export-capable interpreter so the benchmark both runs and appends to Google Sheets:

```bash
/Users/kayvon/.venvs/kayvon-bench/bin/python benchmarker.py --benchmark-profile together_kimi_k2_5_dedicated_track_a_retrieval_smoke
```

## Spreadsheet Logging Reminder

If `gspread` is missing in the interpreter you use, the benchmark will run but the results will not be appended to the sheet.

Successful export ends with:

```text
Google Sheets export: appended 1 row(s) to 'Summary' ...
```

## Recommendation

For future dedicated bring-up requests:

- check existing endpoints first with `client.endpoints.list(mine=True, type="dedicated")`
- reuse an existing endpoint only if it is clearly ours and appropriate for the benchmark
- otherwise create a fresh dedicated endpoint, wait for `STARTED`, then wire its returned `name` into config
