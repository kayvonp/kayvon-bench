import argparse
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import yaml
from openai import OpenAI


DEFAULT_CONFIG_PATH = "benchmarker_config.yaml"
DEFAULT_SUMMARY_SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/1RTTeX6_jh_GVz0FN6yfhEfbx36Gy-OkgEigy6EMMXgQ/edit?usp=sharing"
)
DEFAULT_SUMMARY_WORKSHEET = "Summary"
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


def extract_stream_delta_text(chunk: Any) -> str:
    choices = getattr(chunk, "choices", None) or []
    parts: list[str] = []
    for choice in choices:
        delta = getattr(choice, "delta", None)
        if delta is None and isinstance(choice, dict):
            delta = choice.get("delta")
        if not delta:
            continue

        content = getattr(delta, "content", None)
        if content is None and isinstance(delta, dict):
            content = delta.get("content")

        if isinstance(content, str):
            if content:
                parts.append(content)
            continue

        if isinstance(content, list):
            for item in content:
                text = getattr(item, "text", None) if not isinstance(item, dict) else item.get("text")
                if text:
                    parts.append(str(text))
            continue

        text = getattr(delta, "text", None) if not isinstance(delta, dict) else delta.get("text")
        if text:
            parts.append(str(text))

    return "".join(parts)


def extract_stream_finish_reason(chunk: Any) -> str:
    choices = getattr(chunk, "choices", None) or []
    for choice in choices:
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason is None and isinstance(choice, dict):
            finish_reason = choice.get("finish_reason")
        if finish_reason:
            return str(finish_reason)
    return ""


def extract_stream_reasoning_text(chunk: Any) -> str:
    choices = getattr(chunk, "choices", None) or []
    parts: list[str] = []
    for choice in choices:
        delta = getattr(choice, "delta", None)
        if delta is None and isinstance(choice, dict):
            delta = choice.get("delta")
        if not delta:
            continue
        reasoning = getattr(delta, "reasoning", None)
        if reasoning is None and isinstance(delta, dict):
            reasoning = delta.get("reasoning")
        if isinstance(reasoning, str) and reasoning:
            parts.append(reasoning)
        elif isinstance(reasoning, list):
            parts.extend(str(item) for item in reasoning if item)
    return "".join(parts)


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


def estimate_message_input_token_split(sample: dict[str, Any]) -> tuple[int, int]:
    raw_messages = sample.get("messages") or []
    system_texts: list[str] = []
    user_texts: list[str] = []

    for msg in raw_messages:
        role = str(msg.get("role", ""))
        content = str(msg.get("content", ""))
        if not content:
            continue
        if role == "system":
            system_texts.append(content)
        elif role == "user":
            user_texts.append(content)

    system_tokens = estimate_visible_tokens("\n\n".join(system_texts)) if system_texts else 0
    user_tokens = estimate_visible_tokens("\n\n".join(user_texts)) if user_texts else 0
    return system_tokens, user_tokens


def resolve_sample_input_token_estimates(sample: dict[str, Any]) -> tuple[int, int, int]:
    total_input_tokens = (
        get_int_field(sample.get("prompt_tokens_estimate"))
        or get_int_field(sample.get("input_len_est"))
        or 0
    )
    system_tokens = get_int_field(sample.get("system_tokens_estimate"))
    user_prompt_tokens = get_int_field(sample.get("user_tokens_estimate"))

    inferred_system_tokens, inferred_user_tokens = estimate_message_input_token_split(sample)
    if system_tokens is None:
        system_tokens = inferred_system_tokens
    if user_prompt_tokens is None:
        user_prompt_tokens = inferred_user_tokens

    system_tokens = system_tokens or 0
    user_prompt_tokens = user_prompt_tokens or 0

    split_total = system_tokens + user_prompt_tokens
    if total_input_tokens > 0:
        if split_total > 0:
            scale = total_input_tokens / split_total
            system_tokens = int(round(system_tokens * scale))
            user_prompt_tokens = max(0, total_input_tokens - system_tokens)
        else:
            system_tokens = 0
            user_prompt_tokens = total_input_tokens
    else:
        total_input_tokens = split_total

    return system_tokens, user_prompt_tokens, total_input_tokens


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


def parse_spreadsheet_id(url_or_id: str) -> str:
    raw = (url_or_id or "").strip()
    if not raw:
        raise ValueError("Spreadsheet URL/ID is empty.")
    match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", raw)
    if match:
        return match.group(1)
    return raw


def load_google_service_account_info() -> dict[str, Any] | None:
    raw = (os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON") or "").strip()
    if not raw:
        return None
    if raw.startswith("{"):
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
        return None
    path = Path(raw).expanduser()
    if not path.exists():
        return None
    parsed = json.loads(path.read_text(encoding="utf-8"))
    return parsed if isinstance(parsed, dict) else None


def append_summary_to_google_sheet(
    history: list[dict[str, Any]],
    benchmark_profile_name: str,
    llm: dict[str, str],
) -> None:
    if not history:
        return

    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except Exception as exc:  # noqa: BLE001
        print(f"Google Sheets export skipped: missing dependency ({exc}).")
        return

    sa_info = load_google_service_account_info()
    if not sa_info:
        print("Google Sheets export skipped: GOOGLE_SERVICE_ACCOUNT_JSON is missing/invalid.")
        return

    sheet_target = os.getenv("GOOGLE_SHEET_URL", DEFAULT_SUMMARY_SHEET_URL)
    worksheet_name = os.getenv("GOOGLE_SHEET_WORKSHEET", DEFAULT_SUMMARY_WORKSHEET)
    spreadsheet_id = parse_spreadsheet_id(sheet_target)

    headers = [
        "timestamp_utc",
        "benchmark_profile",
        "llm_profile",
        "provider",
        "model",
        "concurrency",
        "samples",
        "attempt/s",
        "req/s",
        "out_tok/s",
        "req_out_tok/s_p50",
        "req_out_tok/s_p95",
        "ttft_p50_s",
        "ttft_p95_s",
        "lat_p50_s",
        "lat_p95_s",
        "avg_input_tok_total",
        "avg_system_tok_est",
        "avg_user_prompt_tok_est",
        "avg_visible_out_tok_est",
        "avg_thinking_tok_est",
        "avg_total_out_tok_usage",
        "success_rate",
    ]

    rows: list[list[str]] = []
    timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for item in history:
        rows.append(
            [
                timestamp_utc,
                benchmark_profile_name,
                llm["profile_name"],
                llm["provider_name"],
                llm["model"],
                str(item["concurrency"]),
                str(item["completed"]),
                f"{item['attempts_per_sec']:.2f}",
                f"{item['requests_per_sec']:.2f}",
                f"{item['output_tokens_per_sec']:.1f}",
                f"{item['req_output_tps_p50']:.1f}",
                f"{item['req_output_tps_p95']:.1f}",
                f"{item['ttft_p50']:.2f}",
                f"{item['ttft_p95']:.2f}",
                f"{item['lat_p50']:.2f}",
                f"{item['lat_p95']:.2f}",
                f"{item['avg_total_input_tokens_per_request']:.1f}",
                f"{item['avg_system_tokens_est_per_request']:.1f}",
                f"{item['avg_user_prompt_tokens_est_per_request']:.1f}",
                f"{item['avg_visible_output_tokens_est_per_request']:.1f}",
                f"{item['avg_thinking_tokens_est_per_request']:.1f}",
                f"{item['avg_usage_total_out_tokens_per_request']:.1f}",
                f"{item['success_rate'] * 100:.1f}%",
            ]
        )

    try:
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_info(sa_info, scopes=scopes)
        client = gspread.authorize(creds)
        spreadsheet = client.open_by_key(spreadsheet_id)
        try:
            worksheet = spreadsheet.worksheet(worksheet_name)
        except gspread.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(title=worksheet_name, rows=1000, cols=len(headers) + 4)

        existing_header = worksheet.row_values(1)
        if not existing_header:
            worksheet.update(values=[headers], range_name="A1")
        else:
            # Backward-compat: if user inserted a blank column between
            # "concurrency" and "attempt/s", label it as "samples".
            try:
                concurrency_idx = existing_header.index("concurrency")
                samples_idx = concurrency_idx + 1
                attempt_idx = concurrency_idx + 2
                if (
                    attempt_idx < len(existing_header)
                    and existing_header[samples_idx] == ""
                    and existing_header[attempt_idx] == "attempt/s"
                ):
                    worksheet.update_cell(1, samples_idx + 1, "samples")
            except ValueError:
                pass
            if existing_header != headers:
                worksheet.update(values=[headers], range_name="A1")
        worksheet.append_rows(rows, value_input_option="RAW")
        print(
            f"Google Sheets export: appended {len(rows)} row(s) "
            f"to '{worksheet_name}' in spreadsheet {spreadsheet_id}."
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Google Sheets export failed: {exc}")


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


def resolve_llm_settings(config: dict[str, Any], llm_profile_name: str) -> dict[str, Any]:
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
        "reasoning": profile.get("reasoning"),
        "temperature": profile.get("temperature"),
        "top_p": profile.get("top_p"),
        "empty_visible_stream_retries": int(profile.get("empty_visible_stream_retries", 0) or 0),
        "streaming_enabled": bool(profile.get("streaming_enabled", False)),
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


def attach_result_metadata(
    result: dict[str, Any],
    meta: dict[str, int | None],
) -> None:
    result_target = meta.get("target_output_tokens")
    if result.get("ok") and result_target:
        result["target_output_tokens"] = result_target
    if result.get("ok"):
        result["system_input_tokens_est"] = int(meta.get("system_input_tokens_est") or 0)
        result["user_prompt_tokens_est"] = int(meta.get("user_prompt_tokens_est") or 0)
        result["total_input_tokens_est"] = int(meta.get("total_input_tokens_est") or 0)


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


def extract_system_user_prompts(sample: dict[str, Any]) -> tuple[str, str]:
    raw_messages = sample.get("messages") or []
    system_parts: list[str] = []
    user_parts: list[str] = []
    for msg in raw_messages:
        role = str(msg.get("role", ""))
        content = str(msg.get("content", ""))
        if not content:
            continue
        if role == "system":
            system_parts.append(content)
        elif role == "user":
            user_parts.append(content)
    if not raw_messages:
        return "", str(sample)
    return "\n\n".join(system_parts), "\n\n".join(user_parts)


def single_request(
    client: OpenAI,
    model: str,
    api_style: str,
    sample: dict[str, Any],
    output_tokens: int | None,
    timeout_s: float,
    reasoning_effort: str | None = None,
    reasoning: dict[str, Any] | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    empty_visible_stream_retries: int = 0,
    include_full_text: bool = False,
    use_streaming: bool = False,
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
        if reasoning:
            request_kwargs["reasoning"] = reasoning
        if temperature is not None:
            request_kwargs["temperature"] = temperature
        if top_p is not None:
            request_kwargs["top_p"] = top_p

        ttft_s: float | None = None
        finish_reason = ""
        visible_text = ""
        reasoning_text = ""

        if use_streaming:
            if api_style != "chat_completions":
                raise ValueError("Streaming is only supported for chat_completions requests.")
            attempts = max(1, empty_visible_stream_retries + 1)
            last_empty_visible_error = ""
            for attempt_idx in range(attempts):
                request_kwargs["stream"] = True
                request_kwargs["stream_options"] = {"include_usage": True}
                if output_tokens is not None:
                    request_kwargs["max_completion_tokens"] = output_tokens
                    try:
                        stream = client.chat.completions.create(**request_kwargs)
                    except TypeError:
                        request_kwargs.pop("reasoning_effort", None)
                        request_kwargs.pop("reasoning", None)
                        request_kwargs.pop("temperature", None)
                        request_kwargs.pop("top_p", None)
                        request_kwargs.pop("stream_options", None)
                        request_kwargs.pop("max_completion_tokens", None)
                        request_kwargs["max_tokens"] = output_tokens
                        stream = client.chat.completions.create(**request_kwargs)
                else:
                    try:
                        stream = client.chat.completions.create(**request_kwargs)
                    except TypeError:
                        request_kwargs.pop("reasoning_effort", None)
                        request_kwargs.pop("reasoning", None)
                        request_kwargs.pop("temperature", None)
                        request_kwargs.pop("top_p", None)
                        request_kwargs.pop("stream_options", None)
                        stream = client.chat.completions.create(**request_kwargs)

                visible_parts: list[str] = []
                reasoning_parts: list[str] = []
                chunk_count = 0
                usage_chunk = None
                ttft_s = None
                finish_reason = ""
                for chunk in stream:
                    chunk_count += 1
                    delta_text = extract_stream_delta_text(chunk)
                    delta_reasoning = extract_stream_reasoning_text(chunk)
                    if delta_text:
                        visible_parts.append(delta_text)
                        if ttft_s is None:
                            ttft_s = time.perf_counter() - start
                    if delta_reasoning:
                        reasoning_parts.append(delta_reasoning)
                    if getattr(chunk, "usage", None) is not None:
                        usage_chunk = chunk
                    finish_reason = extract_stream_finish_reason(chunk) or finish_reason

                visible_text = "".join(visible_parts)
                reasoning_text = "".join(reasoning_parts)
                if visible_text:
                    break

                last_empty_visible_error = (
                    f"Streaming response completed without visible content "
                    f"(chunks={chunk_count}, reasoning_chars={len(reasoning_text)}, attempt={attempt_idx + 1}/{attempts})."
                )
            else:
                latency = time.perf_counter() - start
                return {
                    "ok": False,
                    "latency_s": latency,
                    "ttft_s": ttft_s,
                    "reasoning_text": reasoning_text,
                    "error": last_empty_visible_error,
                }
            prompt_tokens, completion_tokens, total_tokens = extract_usage_tokens(usage_chunk)
            usage = getattr(usage_chunk, "usage", None) if usage_chunk is not None else None
        else:
            if output_tokens is not None:
                if api_style == "responses":
                    request_kwargs["max_output_tokens"] = output_tokens
                    try:
                        response = client.responses.create(**request_kwargs)
                    except TypeError:
                        request_kwargs.pop("reasoning_effort", None)
                        request_kwargs.pop("reasoning", None)
                        request_kwargs.pop("temperature", None)
                        request_kwargs.pop("top_p", None)
                        request_kwargs.pop("max_output_tokens", None)
                        request_kwargs["max_tokens"] = output_tokens
                        response = client.responses.create(**request_kwargs)
                else:
                    request_kwargs["max_completion_tokens"] = output_tokens
                    try:
                        response = client.chat.completions.create(**request_kwargs)
                    except TypeError:
                        request_kwargs.pop("reasoning_effort", None)
                        request_kwargs.pop("reasoning", None)
                        request_kwargs.pop("temperature", None)
                        request_kwargs.pop("top_p", None)
                        request_kwargs.pop("max_completion_tokens", None)
                        request_kwargs["max_tokens"] = output_tokens
                        response = client.chat.completions.create(**request_kwargs)
            else:
                if api_style == "responses":
                    try:
                        response = client.responses.create(**request_kwargs)
                    except TypeError:
                        request_kwargs.pop("reasoning_effort", None)
                        request_kwargs.pop("reasoning", None)
                        request_kwargs.pop("temperature", None)
                        request_kwargs.pop("top_p", None)
                        response = client.responses.create(**request_kwargs)
                else:
                    try:
                        response = client.chat.completions.create(**request_kwargs)
                    except TypeError:
                        request_kwargs.pop("reasoning_effort", None)
                        request_kwargs.pop("reasoning", None)
                        request_kwargs.pop("temperature", None)
                        request_kwargs.pop("top_p", None)
                        response = client.chat.completions.create(**request_kwargs)

            prompt_tokens, completion_tokens, total_tokens = extract_usage_tokens(response)
            usage = getattr(response, "usage", None)
            visible_text = extract_response_text(response, api_style)
            reasoning_text = extract_reasoning_text(response, api_style)
            if api_style == "chat_completions":
                choices = getattr(response, "choices", None) or []
                if choices:
                    finish_reason = str(getattr(choices[0], "finish_reason", "") or "")

        reasoning_tokens = extract_usage_detail_int(usage, "completion_tokens_details", "reasoning_tokens")
        if reasoning_tokens == 0:
            reasoning_tokens = extract_usage_detail_int(usage, "output_tokens_details", "reasoning_tokens")
        accepted_prediction_tokens = extract_usage_detail_int(
            usage, "completion_tokens_details", "accepted_prediction_tokens"
        )
        rejected_prediction_tokens = extract_usage_detail_int(
            usage, "completion_tokens_details", "rejected_prediction_tokens"
        )
        visible_chars = len(visible_text)
        visible_words = len(visible_text.split()) if visible_text else 0
        visible_token_est = estimate_visible_tokens(visible_text)
        reasoning_chars = len(reasoning_text)
        reasoning_token_est = estimate_visible_tokens(reasoning_text)
        latency = time.perf_counter() - start
        req_output_tps = completion_tokens / latency if latency > 0 else 0.0
        result = {
            "ok": True,
            "latency_s": latency,
            "ttft_s": ttft_s,
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
        if include_full_text:
            result["visible_text"] = visible_text
            result["reasoning_text"] = reasoning_text
        return result
    except Exception as e:  # noqa: BLE001
        latency = time.perf_counter() - start
        return {
            "ok": False,
            "latency_s": latency,
            "ttft_s": None,
            "error": str(e),
        }


def summarize_level(concurrency: int, elapsed_s: float, results: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [r for r in results if r["ok"]]
    errors = [r for r in results if not r["ok"]]
    latencies = [r["latency_s"] for r in successes]
    ttfts = [float(r["ttft_s"]) for r in successes if r.get("ttft_s") is not None]
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
    system_input_tokens_est = sum(int(r.get("system_input_tokens_est", 0) or 0) for r in successes)
    user_prompt_tokens_est = sum(int(r.get("user_prompt_tokens_est", 0) or 0) for r in successes)

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
        "avg_total_input_tokens_per_request": (prompt_tokens / success_count) if success_count else 0.0,
        "avg_system_tokens_est_per_request": (system_input_tokens_est / success_count) if success_count else 0.0,
        "avg_user_prompt_tokens_est_per_request": (user_prompt_tokens_est / success_count) if success_count else 0.0,
        "avg_completion_tokens_per_request": (completion_tokens / success_count) if success_count else 0.0,
        "avg_reasoning_tokens_per_request": (reasoning_tokens / success_count) if success_count else 0.0,
        "avg_visible_token_est_per_request": (visible_token_est / success_count) if success_count else 0.0,
        "avg_visible_chars_per_request": (visible_chars / success_count) if success_count else 0.0,
        "avg_reasoning_token_est_per_request": (reasoning_token_est / success_count) if success_count else 0.0,
        "avg_reasoning_chars_per_request": (reasoning_chars / success_count) if success_count else 0.0,
        # Output-token views:
        # - usage_total_out: provider-reported completion tokens.
        # - visible_est/thinking_est: text-length-based estimates from visible/reasoning text.
        # - total_est_out: visible_est + thinking_est.
        "avg_usage_total_out_tokens_per_request": (completion_tokens / success_count) if success_count else 0.0,
        "avg_visible_output_tokens_est_per_request": (visible_token_est / success_count) if success_count else 0.0,
        "avg_thinking_tokens_est_per_request": (reasoning_token_est / success_count) if success_count else 0.0,
        "avg_total_output_tokens_est_per_request": (
            (visible_token_est + reasoning_token_est) / success_count
        ) if success_count else 0.0,
        "avg_req_output_tokens_per_sec": mean(req_output_tps_values) if req_output_tps_values else 0.0,
        "ttft_p50": percentile(ttfts, 0.50),
        "ttft_p95": percentile(ttfts, 0.95),
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
    measure_ttft = bool(profile.get("measure_ttft", llm["api_style"] == "chat_completions"))
    use_streaming = bool(
        profile.get("use_streaming", measure_ttft and llm["api_style"] == "chat_completions")
    )
    request_spacing_s = float(profile.get("request_spacing_s", 0.0))

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
    print(f"Streaming enabled: {use_streaming}")
    print(f"TTFT enabled: {measure_ttft}")
    print(f"Dataset: {dataset_path} ({len(corpus)} records)")
    print(f"Concurrency ramp: {concurrency_levels}")
    print(f"Diagnostic mode: {diagnostic_mode}")
    if request_spacing_s > 0:
        print(f"Request spacing: {request_spacing_s:.1f}s")
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

        def print_progress(done_count: int) -> None:
            elapsed = time.perf_counter() - start_level
            done_output_tokens = sum(int(r.get("completion_tokens", 0)) for r in results if r.get("ok"))
            done_ok = sum(1 for r in results if r.get("ok"))
            done_err = done_count - done_ok
            latencies = [float(r["latency_s"]) for r in results if r.get("ok")]
            done_target_tokens = sum(
                int(r.get("target_output_tokens", 0) or 0) for r in results if r.get("ok")
            )
            p50 = percentile(latencies, 0.5)
            p95 = percentile(latencies, 0.95)
            req_per_s = done_count / elapsed if elapsed > 0 else 0.0
            output_tps = done_output_tokens / elapsed if elapsed > 0 else 0.0
            avg_actual = done_output_tokens / done_ok if done_ok else 0.0
            avg_target = done_target_tokens / done_ok if done_ok else 0.0
            target_ratio = (avg_actual / avg_target) if avg_target else 0.0
            print(
                "  "
                f"{done_count:>4}/{requests_for_level:<4} | "
                f"{done_ok:>3} | "
                f"{done_err:>3} | "
                f"{elapsed:>8.2f} | "
                f"{req_per_s:>5.2f} | "
                f"{output_tps:>9.1f} | "
                f"{p50:>8.2f} | "
                f"{p95:>8.2f} | "
                f"avg_out={avg_actual:>6.0f}/{avg_target:>6.0f} ({target_ratio:>5.2f}x)"
            )

        sample_payloads: list[tuple[dict[str, Any], dict[str, int | None]]] = []
        for sample in samples:
            if max_output_tokens_mode == "disabled":
                sample_target = None
            elif max_output_tokens_mode == "fixed":
                sample_target = max_output_tokens_fixed
            else:
                sample_target = resolve_output_token_target(sample)
            sample_system_tokens_est, sample_user_prompt_tokens_est, sample_total_input_tokens_est = (
                resolve_sample_input_token_estimates(sample)
            )
            sample_payloads.append(
                (
                    sample,
                    {
                        "target_output_tokens": sample_target,
                        "system_input_tokens_est": sample_system_tokens_est,
                        "user_prompt_tokens_est": sample_user_prompt_tokens_est,
                        "total_input_tokens_est": sample_total_input_tokens_est,
                    },
                )
            )

        if concurrency == 1:
            for i, (sample, meta) in enumerate(sample_payloads, start=1):
                if i > 1 and request_spacing_s > 0:
                    time.sleep(request_spacing_s)
                result = single_request(
                    client,
                    llm["model"],
                    llm["api_style"],
                    sample,
                    meta["target_output_tokens"],
                    timeout_s,
                    llm.get("reasoning_effort"),
                    llm.get("reasoning"),
                    llm.get("temperature"),
                    llm.get("top_p"),
                    int(llm.get("empty_visible_stream_retries", 0) or 0),
                    diagnostic_mode,
                    use_streaming,
                )
                attach_result_metadata(result, meta)
                results.append(result)
                if i % progress_every == 0 or i == requests_for_level:
                    print_progress(i)
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures: dict[Any, dict[str, int | None]] = {}
                for sample, meta in sample_payloads:
                    fut = executor.submit(
                        single_request,
                        client,
                        llm["model"],
                        llm["api_style"],
                        sample,
                        meta["target_output_tokens"],
                        timeout_s,
                        llm.get("reasoning_effort"),
                        llm.get("reasoning"),
                        llm.get("temperature"),
                        llm.get("top_p"),
                        int(llm.get("empty_visible_stream_retries", 0) or 0),
                        diagnostic_mode,
                        use_streaming,
                    )
                    futures[fut] = meta

                for i, future in enumerate(as_completed(futures), start=1):
                    result = future.result()
                    results.append(result)
                    attach_result_metadata(result, futures[future])
                    if i % progress_every == 0 or i == requests_for_level:
                        print_progress(i)

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
                "avg_tok(vis/thk/total_est)",
                "ttft_p50_s",
                "ttft_p95_s",
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
                (
                    f"{metrics['avg_visible_output_tokens_est_per_request']:.1f}/"
                    f"{metrics['avg_thinking_tokens_est_per_request']:.1f}/"
                    f"{metrics['avg_total_output_tokens_est_per_request']:.1f}"
                ),
                f"{metrics['ttft_p50']:.2f}",
                f"{metrics['ttft_p95']:.2f}",
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
                "ttft_s",
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
                        "" if result.get("ttft_s") is None else f"{float(result.get('ttft_s', 0.0)):.2f}",
                        f"{float(result.get('latency_s', 0.0)):.2f}",
                    ]
                )
            print_table(diag_headers, diag_rows, indent="  ")
            if metrics["successes"] > 0:
                print(
                    "  token diagnostics summary: "
                    f"avg_usage_total_out={metrics['avg_usage_total_out_tokens_per_request']:.1f}, "
                    f"avg_visible_out_est={metrics['avg_visible_output_tokens_est_per_request']:.1f}, "
                    f"avg_usage_reasoning={metrics['avg_reasoning_tokens_per_request']:.1f}, "
                    f"avg_thinking_est={metrics['avg_thinking_tokens_est_per_request']:.1f}, "
                    f"avg_total_out_est={metrics['avg_total_output_tokens_est_per_request']:.1f}, "
                    f"avg_visible_chars={metrics['avg_visible_chars_per_request']:.1f}, "
                    f"avg_reasoning_chars={metrics['avg_reasoning_chars_per_request']:.1f}"
                )
                preview = next((r for r in results if r.get("ok")), None)
                if preview:
                    print(f"  visible_preview: {preview.get('visible_preview', '')!r}")
                    print(f"  reasoning_preview: {preview.get('reasoning_preview', '')!r}")
        if diagnostic_mode and results:
            first_sample = samples[0] if samples else {}
            system_prompt, user_prompt = extract_system_user_prompts(first_sample)
            first_result = results[0]
            visible_text = str(first_result.get("visible_text", ""))
            reasoning_text = str(first_result.get("reasoning_text", ""))
            error_text = str(first_result.get("error", "")) if not first_result.get("ok") else ""

            capture_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            output_dir = Path("output")
            output_dir.mkdir(parents=True, exist_ok=True)
            prompts_path = output_dir / f"diagnostic_{capture_ts}_prompts.txt"
            thinking_path = output_dir / f"diagnostic_{capture_ts}_thinking_trace.txt"
            response_path = output_dir / f"diagnostic_{capture_ts}_visible_response.txt"

            with prompts_path.open("w", encoding="utf-8") as handle:
                handle.write("=== SYSTEM PROMPT ===\n")
                handle.write(system_prompt)
                handle.write("\n\n=== USER PROMPT ===\n")
                handle.write(user_prompt)
                handle.write("\n")

            with thinking_path.open("w", encoding="utf-8") as handle:
                handle.write(reasoning_text)
                if error_text:
                    handle.write("\n\n=== ERROR ===\n")
                    handle.write(error_text)
                handle.write("\n")

            with response_path.open("w", encoding="utf-8") as handle:
                handle.write(visible_text)
                if error_text:
                    handle.write("\n\n=== ERROR ===\n")
                    handle.write(error_text)
                handle.write("\n")

            print(f"  diagnostic_prompts_file: {prompts_path}")
            print(f"  diagnostic_thinking_file: {thinking_path}")
            print(f"  diagnostic_response_file: {response_path}")

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
                f"{item['ttft_p50']:.2f}",
                f"{item['ttft_p95']:.2f}",
                f"{item['lat_p50']:.2f}",
                f"{item['lat_p95']:.2f}",
                f"{item['avg_total_input_tokens_per_request']:.1f}",
                f"{item['avg_system_tokens_est_per_request']:.1f}",
                f"{item['avg_user_prompt_tokens_est_per_request']:.1f}",
                f"{item['avg_visible_output_tokens_est_per_request']:.1f}",
                f"{item['avg_thinking_tokens_est_per_request']:.1f}",
                f"{item['avg_usage_total_out_tokens_per_request']:.1f}",
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
            "ttft_p50_s",
            "ttft_p95_s",
            "lat_p50_s",
            "lat_p95_s",
            "avg_input_tok_total",
            "avg_system_tok_est",
            "avg_user_prompt_tok_est",
            "avg_visible_out_tok_est",
            "avg_thinking_tok_est",
            "avg_total_out_tok_usage",
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
    append_summary_to_google_sheet(history, benchmark_profile_name, llm)


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
