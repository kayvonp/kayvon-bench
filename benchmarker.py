import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any

import yaml
from openai import OpenAI


DEFAULT_CONFIG_PATH = "benchmarker_config.yaml"
DEFAULT_PROVIDERS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "api_style": "responses",
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "api_key_env": "TOGETHER_API_KEY",
        "api_style": "chat_completions",
    },
}


def load_local_env(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def load_yaml_config(path: str) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise SystemExit(f"Config file not found: {path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"Config file '{path}' must be a top-level mapping.")
    return data


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


def print_table(headers: list[str], rows: list[list[str]], indent: str = "") -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def fmt(row: list[str]) -> str:
        return indent + " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))

    print(fmt(headers))
    print(indent + "-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt(row))


def extract_usage_tokens(response: Any) -> tuple[int, int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0, 0

    prompt_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or 0)

    if prompt_tokens == 0:
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    if completion_tokens == 0:
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens

    return prompt_tokens, completion_tokens, total_tokens


def extract_usage_detail_int(usage: Any, parent_attr: str, child_attr: str) -> int:
    parent = getattr(usage, parent_attr, None)
    if parent is None:
        return 0
    value = getattr(parent, child_attr, 0)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def extract_response_text(response: Any, api_style: str) -> str:
    if api_style == "chat_completions":
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                text = getattr(item, "text", None) if not isinstance(item, dict) else item.get("text")
                if text:
                    parts.append(str(text))
            return "\n".join(parts)
        return str(content or "")

    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text:
        return output_text

    output = getattr(response, "output", None) or []
    parts: list[str] = []
    for item in output:
        content = getattr(item, "content", None)
        if content is None and isinstance(item, dict):
            content = item.get("content")
        if not content:
            continue
        for c in content:
            c_type = getattr(c, "type", None) if not isinstance(c, dict) else c.get("type")
            if c_type != "output_text":
                continue
            text = getattr(c, "text", None) if not isinstance(c, dict) else c.get("text")
            if text:
                parts.append(str(text))
    return "\n".join(parts)


def extract_reasoning_text(response: Any, api_style: str) -> str:
    if api_style == "chat_completions":
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        reasoning = getattr(message, "reasoning", "")
        if isinstance(reasoning, str):
            return reasoning
        if isinstance(reasoning, list):
            return "\n".join(str(x) for x in reasoning)
        return str(reasoning or "")
    return ""


def estimate_visible_tokens(text: str) -> int:
    if not text:
        return 0
    # Simple tokenizer-free estimate that tracks BPE-ish sizes reasonably for diagnostics.
    return max(1, round(len(text) / 4))


def get_int_field(value: Any) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def resolve_output_token_target(sample: dict[str, Any]) -> int | None:
    for key in (
        "max_tokens",
        "output_tokens",
        "max_output_tokens",
        "target_output_tokens",
        "output_tokens_requested",
    ):
        value = get_int_field(sample.get(key))
        if value is not None:
            return value
    return None


def resolve_max_output_tokens_policy(profile: dict[str, Any]) -> tuple[str, int | None]:
    """Resolve max-output-token policy for a benchmark profile.

    Modes:
    - disabled: do not send any max token cap.
    - from_corpus: use per-sample token target fields from dataset rows.
    - fixed: always use max_output_tokens_fixed from profile.
    """
    mode_raw = profile.get("max_output_tokens_mode")
    if mode_raw is None:
        # Backward-compatible behavior.
        disable_max_completion_tokens = bool(profile.get("disable_max_completion_tokens", False))
        return ("disabled", None) if disable_max_completion_tokens else ("from_corpus", None)

    mode = str(mode_raw).strip().lower()
    valid_modes = {"disabled", "from_corpus", "fixed"}
    if mode not in valid_modes:
        raise SystemExit(
            f"Unsupported max_output_tokens_mode: {mode!r}. "
            f"Expected one of: {', '.join(sorted(valid_modes))}."
        )

    if mode == "fixed":
        fixed_value = get_int_field(profile.get("max_output_tokens_fixed"))
        if fixed_value is None:
            raise SystemExit(
                "max_output_tokens_mode='fixed' requires a positive integer max_output_tokens_fixed."
            )
        return mode, fixed_value

    return mode, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run quick LLM endpoint saturation benchmark.")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to YAML config (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--benchmark-profile",
        help="Benchmark profile name in config. Defaults to active.benchmark_profile (or benchmark.active_profile).",
    )
    return parser.parse_args()


def compute_requests_for_level(concurrency: int, profile: dict[str, Any]) -> int:
    request_sizing = profile.get("request_sizing")
    if request_sizing:
        mode = str(request_sizing.get("mode", "concurrency_scaled"))
        if mode != "concurrency_scaled":
            raise SystemExit(f"Unsupported request_sizing.mode: {mode}")

        waves_per_level = int(request_sizing.get("waves_per_level", 6))
        min_requests = int(request_sizing.get("min_requests", concurrency))
        max_requests = int(request_sizing.get("max_requests", 0))
        if waves_per_level <= 0:
            raise SystemExit("request_sizing.waves_per_level must be > 0.")
        if min_requests <= 0:
            raise SystemExit("request_sizing.min_requests must be > 0.")

        requested = max(min_requests, concurrency * waves_per_level)
        if max_requests > 0:
            requested = min(requested, max_requests)
        return requested

    static_requests = int(profile.get("requests_per_level", 10))
    if static_requests <= 0:
        raise SystemExit("requests_per_level must be > 0.")
    return static_requests


def resolve_llm_settings(config: dict[str, Any], llm_profile_name: str) -> dict[str, str]:
    providers = {**DEFAULT_PROVIDERS, **(config.get("providers") or {})}
    profiles = config.get("llm_profiles") or config.get("profiles") or {}

    profile = profiles.get(llm_profile_name)
    if not profile:
        raise SystemExit(f"LLM profile '{llm_profile_name}' not found in config.")

    provider_name = profile.get("provider")
    if not provider_name:
        raise SystemExit(f"LLM profile '{llm_profile_name}' is missing provider.")
    provider_defaults = providers.get(provider_name) or {}

    model = profile.get("model")
    base_url = profile.get("base_url") or provider_defaults.get("base_url")
    api_key_env = profile.get("api_key_env") or provider_defaults.get("api_key_env")
    api_style = profile.get("api_style") or provider_defaults.get("api_style") or "chat_completions"

    missing = []
    if not model:
        missing.append("model")
    if not base_url:
        missing.append("base_url")
    if not api_key_env:
        missing.append("api_key_env")
    if missing:
        raise SystemExit(
            f"LLM profile '{llm_profile_name}' is missing required setting(s): {', '.join(missing)}."
        )

    return {
        "profile_name": llm_profile_name,
        "provider_name": provider_name,
        "model": model,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "api_style": api_style,
        "reasoning_effort": profile.get("reasoning_effort"),
    }


def read_corpus(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append(obj)
    if not records:
        raise SystemExit(f"No records found in dataset: {path}")
    return records


def build_chat_messages(sample: dict[str, Any]) -> list[dict[str, str]]:
    raw_messages = sample.get("messages") or []
    messages: list[dict[str, str]] = []
    for msg in raw_messages:
        role = str(msg.get("role", "user"))
        content = str(msg.get("content", ""))
        if role in {"system", "user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    if not messages:
        messages = [{"role": "user", "content": str(sample)}]
    return messages


def build_responses_input(sample: dict[str, Any]) -> str:
    raw_messages = sample.get("messages") or []
    parts: list[str] = []
    for msg in raw_messages:
        role = str(msg.get("role", "user"))
        content = str(msg.get("content", ""))
        if content:
            parts.append(f"{role}: {content}")
    if not parts:
        return str(sample)
    return "\n\n".join(parts)


def single_request(
    client: OpenAI,
    model: str,
    api_style: str,
    sample: dict[str, Any],
    output_tokens: int | None,
    timeout_s: float,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        request_kwargs: dict[str, Any] = {
            "model": model,
            "timeout": timeout_s,
        }

        if api_style == "responses":
            request_kwargs["input"] = build_responses_input(sample)
        elif api_style == "chat_completions":
            request_kwargs["messages"] = build_chat_messages(sample)
        else:
            raise ValueError(f"Unsupported api_style: {api_style}")

        if reasoning_effort:
            request_kwargs["reasoning_effort"] = reasoning_effort

        if output_tokens is not None:
            if api_style == "responses":
                request_kwargs["max_output_tokens"] = output_tokens
                try:
                    response = client.responses.create(**request_kwargs)
                except TypeError:
                    request_kwargs.pop("reasoning_effort", None)
                    request_kwargs.pop("max_output_tokens", None)
                    request_kwargs["max_tokens"] = output_tokens
                    response = client.responses.create(**request_kwargs)
            else:
                request_kwargs["max_completion_tokens"] = output_tokens
                try:
                    response = client.chat.completions.create(**request_kwargs)
                except TypeError:
                    request_kwargs.pop("reasoning_effort", None)
                    request_kwargs.pop("max_completion_tokens", None)
                    request_kwargs["max_tokens"] = output_tokens
                    response = client.chat.completions.create(**request_kwargs)
        else:
            if api_style == "responses":
                try:
                    response = client.responses.create(**request_kwargs)
                except TypeError:
                    request_kwargs.pop("reasoning_effort", None)
                    response = client.responses.create(**request_kwargs)
            else:
                try:
                    response = client.chat.completions.create(**request_kwargs)
                except TypeError:
                    request_kwargs.pop("reasoning_effort", None)
                    response = client.chat.completions.create(**request_kwargs)

        prompt_tokens, completion_tokens, total_tokens = extract_usage_tokens(response)
        usage = getattr(response, "usage", None)
        reasoning_tokens = extract_usage_detail_int(usage, "completion_tokens_details", "reasoning_tokens")
        if reasoning_tokens == 0:
            reasoning_tokens = extract_usage_detail_int(usage, "output_tokens_details", "reasoning_tokens")
        accepted_prediction_tokens = extract_usage_detail_int(
            usage, "completion_tokens_details", "accepted_prediction_tokens"
        )
        rejected_prediction_tokens = extract_usage_detail_int(
            usage, "completion_tokens_details", "rejected_prediction_tokens"
        )
        visible_text = extract_response_text(response, api_style)
        reasoning_text = extract_reasoning_text(response, api_style)
        visible_chars = len(visible_text)
        visible_words = len(visible_text.split()) if visible_text else 0
        visible_token_est = estimate_visible_tokens(visible_text)
        reasoning_chars = len(reasoning_text)
        reasoning_token_est = estimate_visible_tokens(reasoning_text)
        finish_reason = ""
        if api_style == "chat_completions":
            choices = getattr(response, "choices", None) or []
            if choices:
                finish_reason = str(getattr(choices[0], "finish_reason", "") or "")
        latency = time.perf_counter() - start
        req_output_tps = completion_tokens / latency if latency > 0 else 0.0
        return {
            "ok": True,
            "latency_s": latency,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "req_output_tps": req_output_tps,
            "reasoning_tokens": reasoning_tokens,
            "accepted_prediction_tokens": accepted_prediction_tokens,
            "rejected_prediction_tokens": rejected_prediction_tokens,
            "visible_chars": visible_chars,
            "visible_words": visible_words,
            "visible_token_est": visible_token_est,
            "reasoning_chars": reasoning_chars,
            "reasoning_token_est": reasoning_token_est,
            "finish_reason": finish_reason,
            "visible_preview": visible_text[:240].replace("\n", " "),
            "reasoning_preview": reasoning_text[:240].replace("\n", " "),
        }
    except Exception as e:  # noqa: BLE001
        latency = time.perf_counter() - start
        return {
            "ok": False,
            "latency_s": latency,
            "error": str(e),
        }


def summarize_level(concurrency: int, elapsed_s: float, results: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [r for r in results if r["ok"]]
    errors = [r for r in results if not r["ok"]]
    latencies = [r["latency_s"] for r in successes]
    req_output_tps_values = [r["req_output_tps"] for r in successes]
    prompt_tokens = sum(int(r.get("prompt_tokens", 0)) for r in successes)
    completion_tokens = sum(int(r.get("completion_tokens", 0)) for r in successes)
    reasoning_tokens = sum(int(r.get("reasoning_tokens", 0)) for r in successes)
    visible_token_est = sum(int(r.get("visible_token_est", 0)) for r in successes)
    visible_chars = sum(int(r.get("visible_chars", 0)) for r in successes)
    reasoning_token_est = sum(int(r.get("reasoning_token_est", 0)) for r in successes)
    reasoning_chars = sum(int(r.get("reasoning_chars", 0)) for r in successes)
    target_output_tokens = sum(int(r.get("target_output_tokens", 0) or 0) for r in successes)
    total_tokens = sum(int(r.get("total_tokens", 0)) for r in successes)

    completed = len(results)
    success_count = len(successes)
    error_count = len(errors)

    return {
        "concurrency": concurrency,
        "elapsed_s": elapsed_s,
        "completed": completed,
        "successes": success_count,
        "errors": error_count,
        "success_rate": (success_count / completed) if completed else 0.0,
        "attempts_per_sec": (completed / elapsed_s) if elapsed_s > 0 else 0.0,
        "requests_per_sec": (success_count / elapsed_s) if elapsed_s > 0 else 0.0,
        "input_tokens_per_sec": (prompt_tokens / elapsed_s) if elapsed_s > 0 else 0.0,
        "output_tokens_per_sec": (completion_tokens / elapsed_s) if elapsed_s > 0 else 0.0,
        "reasoning_tokens_per_sec": (reasoning_tokens / elapsed_s) if elapsed_s > 0 else 0.0,
        "visible_token_est_per_sec": (visible_token_est / elapsed_s) if elapsed_s > 0 else 0.0,
        "target_output_tokens": target_output_tokens,
        "avg_tokens_per_request": (total_tokens / success_count) if success_count else 0.0,
        "avg_prompt_tokens_per_request": (prompt_tokens / success_count) if success_count else 0.0,
        "avg_completion_tokens_per_request": (completion_tokens / success_count) if success_count else 0.0,
        "avg_reasoning_tokens_per_request": (reasoning_tokens / success_count) if success_count else 0.0,
        "avg_visible_token_est_per_request": (visible_token_est / success_count) if success_count else 0.0,
        "avg_visible_chars_per_request": (visible_chars / success_count) if success_count else 0.0,
        "avg_reasoning_token_est_per_request": (reasoning_token_est / success_count) if success_count else 0.0,
        "avg_reasoning_chars_per_request": (reasoning_chars / success_count) if success_count else 0.0,
        "avg_req_output_tokens_per_sec": mean(req_output_tps_values) if req_output_tps_values else 0.0,
        "lat_p50": percentile(latencies, 0.50),
        "lat_p95": percentile(latencies, 0.95),
        "req_output_tps_p50": percentile(req_output_tps_values, 0.50),
        "req_output_tps_p95": percentile(req_output_tps_values, 0.95),
        "error_examples": [e.get("error", "") for e in errors[:3]],
    }


def detect_saturation(
    history: list[dict[str, Any]],
    current: dict[str, Any],
    sat_cfg: dict[str, Any],
) -> tuple[bool, str]:
    if not history:
        return False, ""

    prev = history[-1]
    best_tps = max([h["output_tokens_per_sec"] for h in history] + [current["output_tokens_per_sec"]])

    plateau_pct = float(sat_cfg.get("throughput_plateau_pct", 0.03))
    decline_pct = float(sat_cfg.get("throughput_decline_pct", 0.07))
    latency_mult = float(sat_cfg.get("latency_spike_multiplier", 1.35))

    plateau_or_decline = current["output_tokens_per_sec"] <= (prev["output_tokens_per_sec"] * (1.0 + plateau_pct))
    sharp_decline = current["output_tokens_per_sec"] <= (best_tps * (1.0 - decline_pct))
    latency_spike = current["lat_p95"] >= (prev["lat_p95"] * latency_mult if prev["lat_p95"] > 0 else float("inf"))

    if latency_spike and (plateau_or_decline or sharp_decline):
        reason = (
            f"latency spike with throughput leveling/decline "
            f"(prev_out_tps={prev['output_tokens_per_sec']:.1f}, curr_out_tps={current['output_tokens_per_sec']:.1f}, "
            f"prev_p95={prev['lat_p95']:.2f}s, curr_p95={current['lat_p95']:.2f}s)"
        )
        return True, reason
    return False, ""


def run_benchmark(config: dict[str, Any], benchmark_profile_name: str) -> None:
    active_config = config.get("active") or {}
    benchmark_root = config.get("benchmark") or {}
    benchmark_profiles = config.get("benchmark_profiles") or benchmark_root.get("profiles") or {}
    profile = benchmark_profiles.get(benchmark_profile_name)
    if not profile:
        raise SystemExit(f"Benchmark profile '{benchmark_profile_name}' not found in config.")

    llm_profile_name = (
        profile.get("llm_profile")
        or profile.get("llm_profile_ref")
        or active_config.get("llm_profile")
        or config.get("active_profile")
    )
    if not llm_profile_name:
        raise SystemExit(
            f"Benchmark profile '{benchmark_profile_name}' is missing llm profile reference."
        )

    llm = resolve_llm_settings(config, llm_profile_name)
    dataset_path = str(profile.get("dataset_path", "")).strip()
    if not dataset_path:
        raise SystemExit(f"Benchmark profile '{benchmark_profile_name}' is missing dataset_path.")

    corpus = read_corpus(dataset_path)
    sample_with_replacement = bool(profile.get("sample_with_replacement", True))
    random_seed = int(profile.get("random_seed", 0))
    concurrency_levels = [int(x) for x in profile.get("concurrency_levels", [1, 2, 4])]
    progress_every = int(profile.get("progress_update_every", 2))
    timeout_s = float(profile.get("request_timeout_s", 120))
    max_output_tokens_mode, max_output_tokens_fixed = resolve_max_output_tokens_policy(profile)
    stop_on_saturation = bool(profile.get("stop_on_saturation", True))
    min_levels_before_stop = int(profile.get("min_levels_before_stop", 3))
    sat_cfg = profile.get("saturation") or {}
    diagnostic_mode = bool(profile.get("diagnostic_mode", config.get("diagnostic_mode", False)))
    token_diagnostics = bool(profile.get("token_diagnostics", diagnostic_mode))

    if not concurrency_levels:
        raise SystemExit("concurrency_levels is empty.")

    if diagnostic_mode:
        concurrency_levels = [1]
        progress_every = 1
        stop_on_saturation = False

    api_key = os.getenv(llm["api_key_env"])
    if not api_key:
        raise SystemExit(f"{llm['api_key_env']} is not set (env var or .env).")
    client = OpenAI(api_key=api_key, base_url=llm["base_url"])

    rng = random.Random(random_seed)
    history: list[dict[str, Any]] = []
    print("Starting benchmark...")
    print(f"Benchmark profile: {benchmark_profile_name}")
    print(f"LLM profile: {llm['profile_name']}")
    print(f"Provider: {llm['provider_name']}")
    print(f"Model: {llm['model']}")
    print(f"API style: {llm['api_style']}")
    print(f"Dataset: {dataset_path} ({len(corpus)} records)")
    print(f"Concurrency ramp: {concurrency_levels}")
    print(f"Diagnostic mode: {diagnostic_mode}")
    if max_output_tokens_mode == "fixed":
        print(f"Max output tokens: mode=fixed, value={max_output_tokens_fixed}")
    else:
        print(f"Max output tokens: mode={max_output_tokens_mode}")
    request_sizing = profile.get("request_sizing")
    if request_sizing:
        print(
            "Request sizing: "
            f"mode={request_sizing.get('mode', 'concurrency_scaled')}, "
            f"waves_per_level={request_sizing.get('waves_per_level', 6)}, "
            f"min_requests={request_sizing.get('min_requests', 'n/a')}, "
            f"max_requests={request_sizing.get('max_requests', 'unbounded')}"
        )
    else:
        print(f"Requests per level (static): {profile.get('requests_per_level', 10)}")
    print("")

    for level_idx, concurrency in enumerate(concurrency_levels, start=1):
        requests_for_level = 1 if diagnostic_mode else compute_requests_for_level(concurrency, profile)
        print(
            f"[Level {level_idx}/{len(concurrency_levels)}] "
            f"concurrency={concurrency}, requests={requests_for_level}"
        )
        progress_headers = [
            "done",
            "ok",
            "err",
            "elapsed_s",
            "req/s",
            "out_tok/s",
            "lat_p50_s",
            "lat_p95_s",
            "avg_out/target",
        ]
        print_table(progress_headers, [], indent="  ")
        if diagnostic_mode:
            samples = [corpus[0]]
        elif sample_with_replacement:
            samples = [rng.choice(corpus) for _ in range(requests_for_level)]
        else:
            if requests_for_level > len(corpus):
                raise SystemExit("requests_per_level cannot exceed dataset size when sampling without replacement.")
            samples = rng.sample(corpus, k=requests_for_level)

        start_level = time.perf_counter()
        results: list[dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {}
            for sample in samples:
                if max_output_tokens_mode == "disabled":
                    sample_target = None
                elif max_output_tokens_mode == "fixed":
                    sample_target = max_output_tokens_fixed
                else:
                    sample_target = resolve_output_token_target(sample)
                fut = executor.submit(
                    single_request,
                    client,
                    llm["model"],
                    llm["api_style"],
                    sample,
                    sample_target,
                    timeout_s,
                    llm.get("reasoning_effort"),
                )
                futures[fut] = sample_target

            for i, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results.append(result)
                result_target = futures[future]
                if result.get("ok") and result_target:
                    result["target_output_tokens"] = result_target

                if i % progress_every == 0 or i == requests_for_level:
                    elapsed = time.perf_counter() - start_level
                    done_output_tokens = sum(int(r.get("completion_tokens", 0)) for r in results if r.get("ok"))
                    done_ok = sum(1 for r in results if r.get("ok"))
                    done_err = i - done_ok
                    latencies = [float(r["latency_s"]) for r in results if r.get("ok")]
                    done_target_tokens = sum(
                        int(r.get("target_output_tokens", 0) or 0) for r in results if r.get("ok")
                    )
                    p50 = percentile(latencies, 0.5)
                    p95 = percentile(latencies, 0.95)
                    req_per_s = i / elapsed if elapsed > 0 else 0.0
                    output_tps = done_output_tokens / elapsed if elapsed > 0 else 0.0
                    avg_actual = done_output_tokens / done_ok if done_ok else 0.0
                    avg_target = done_target_tokens / done_ok if done_ok else 0.0
                    target_ratio = (avg_actual / avg_target) if avg_target else 0.0
                    print(
                        "  "
                        f"{i:>4}/{requests_for_level:<4} | "
                        f"{done_ok:>3} | "
                        f"{done_err:>3} | "
                        f"{elapsed:>8.2f} | "
                        f"{req_per_s:>5.2f} | "
                        f"{output_tps:>9.1f} | "
                        f"{p50:>8.2f} | "
                        f"{p95:>8.2f} | "
                        f"avg_out={avg_actual:>6.0f}/{avg_target:>6.0f} ({target_ratio:>5.2f}x)"
                    )

        elapsed_level = time.perf_counter() - start_level
        metrics = summarize_level(concurrency, elapsed_level, results)
        history.append(metrics)

        print("  level summary:")
        print_table(
            [
                "attempt/s",
                "req/s",
                "out_tok/s",
                "req_out_tok/s(avg/p50/p95)",
                "lat_p50_s",
                "lat_p95_s",
                "success",
            ],
            [[
                f"{metrics['attempts_per_sec']:.2f}",
                f"{metrics['requests_per_sec']:.2f}",
                f"{metrics['output_tokens_per_sec']:.1f}",
                (
                    f"{metrics['avg_req_output_tokens_per_sec']:.1f}/"
                    f"{metrics['req_output_tps_p50']:.1f}/"
                    f"{metrics['req_output_tps_p95']:.1f}"
                ),
                f"{metrics['lat_p50']:.2f}",
                f"{metrics['lat_p95']:.2f}",
                f"{metrics['successes']}/{metrics['completed']}",
            ]],
            indent="  ",
        )
        if metrics["error_examples"]:
            print(f"  sample_error: {metrics['error_examples'][0]}")
        if token_diagnostics and results:
            print("  token diagnostics:")
            diag_headers = [
                "idx",
                "ok",
                "finish",
                "reported_out_tok",
                "visible_tok_est",
                "usage_reasoning_tok",
                "reasoning_tok_est",
                "visible_chars",
                "reasoning_chars",
                "lat_s",
            ]
            diag_rows: list[list[str]] = []
            for idx, result in enumerate(results, start=1):
                diag_rows.append(
                    [
                        str(idx),
                        "Y" if result.get("ok") else "N",
                        str(result.get("finish_reason", "")),
                        str(int(result.get("completion_tokens", 0) or 0)),
                        str(int(result.get("visible_token_est", 0) or 0)),
                        str(int(result.get("reasoning_tokens", 0) or 0)),
                        str(int(result.get("reasoning_token_est", 0) or 0)),
                        str(int(result.get("visible_chars", 0) or 0)),
                        str(int(result.get("reasoning_chars", 0) or 0)),
                        f"{float(result.get('latency_s', 0.0)):.2f}",
                    ]
                )
            print_table(diag_headers, diag_rows, indent="  ")
            if metrics["successes"] > 0:
                print(
                    "  token diagnostics summary: "
                    f"avg_reported_out={metrics['avg_completion_tokens_per_request']:.1f}, "
                    f"avg_visible_est={metrics['avg_visible_token_est_per_request']:.1f}, "
                    f"avg_usage_reasoning={metrics['avg_reasoning_tokens_per_request']:.1f}, "
                    f"avg_reasoning_est={metrics['avg_reasoning_token_est_per_request']:.1f}, "
                    f"avg_visible_chars={metrics['avg_visible_chars_per_request']:.1f}, "
                    f"avg_reasoning_chars={metrics['avg_reasoning_chars_per_request']:.1f}"
                )
                preview = next((r for r in results if r.get("ok")), None)
                if preview:
                    print(f"  visible_preview: {preview.get('visible_preview', '')!r}")
                    print(f"  reasoning_preview: {preview.get('reasoning_preview', '')!r}")

        saturated, reason = detect_saturation(history[:-1], metrics, sat_cfg)
        if saturated and stop_on_saturation and level_idx >= min_levels_before_stop:
            print(f"  saturation_detected: {reason}")
            print("Stopping early due to saturation criteria.")
            break
        print("")

    print("\nFinal ramp summary:")
    summary_rows: list[list[str]] = []
    for item in history:
        summary_rows.append(
            [
                str(item["concurrency"]),
                f"{item['attempts_per_sec']:.2f}",
                f"{item['requests_per_sec']:.2f}",
                f"{item['output_tokens_per_sec']:.1f}",
                f"{item['req_output_tps_p50']:.1f}",
                f"{item['req_output_tps_p95']:.1f}",
                f"{item['lat_p50']:.2f}",
                f"{item['lat_p95']:.2f}",
                f"{item['avg_prompt_tokens_per_request']:.1f}",
                f"{item['avg_completion_tokens_per_request']:.1f}",
                f"{item['success_rate'] * 100:.1f}%",
            ]
        )
    print_table(
        [
            "concurrency",
            "attempt/s",
            "req/s",
            "out_tok/s",
            "req_out_tok/s_p50",
            "req_out_tok/s_p95",
            "lat_p50_s",
            "lat_p95_s",
            "avg_input_tok",
            "avg_output_tok",
            "success_rate",
        ],
        summary_rows,
    )

    if history:
        best = max(history, key=lambda x: x["output_tokens_per_sec"])
        print(
            "\nPeak throughput: "
            f"concurrency={best['concurrency']}, "
            f"out_tok/s={best['output_tokens_per_sec']:.1f}, "
            f"lat_p95={best['lat_p95']:.2f}s"
        )


def main() -> None:
    load_local_env()
    args = parse_args()
    config = load_yaml_config(args.config)

    active_config = config.get("active") or {}
    bench_root = config.get("benchmark") or {}
    benchmark_profile_name = (
        args.benchmark_profile
        or active_config.get("benchmark_profile")
        or bench_root.get("active_profile")
    )
    if not benchmark_profile_name:
        raise SystemExit(
            "Missing benchmark profile. Set active.benchmark_profile (or benchmark.active_profile) or use --benchmark-profile."
        )

    run_benchmark(config, benchmark_profile_name)


if __name__ == "__main__":
    main()
