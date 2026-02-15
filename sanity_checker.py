#!/usr/bin/env python3
"""Run a detailed sanity check using the existing benchmarker config."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI


DEFAULT_CONFIG_PATH = "benchmarker_config.yaml"
DEFAULT_DATASET = "corpus/v1_summarization_13000to600.jsonl"
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


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a transparent single-request LLM sanity check.")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to benchmarker YAML (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--benchmark-profile",
        default=None,
        help="Benchmark profile name from benchmarker config. Defaults to active.benchmark_profile.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="JSONL dataset to read first record from. Defaults to profile dataset_path or v1_summarization_13000to600.jsonl.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Path to write full log output. Default: logs/sanity_check_<timestamp>.log",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Client timeout seconds.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        help="Optional reasoning effort setting for reasoning-capable chat models.",
    )
    parser.add_argument(
        "--no-max-completion-tokens",
        action="store_true",
        help="Do not send max_completion_tokens/max_output_tokens in the request.",
    )
    return parser.parse_args()


def setup_logger(log_file: str | None = None) -> tuple[logging.Logger, Path]:
    if log_file is None:
        Path("logs").mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_path = Path("logs") / f"sanity_check_{timestamp}.log"
    else:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("sanity_checker")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d UTC %(levelname)s %(message)s",
        "%Y-%m-%dT%H:%M:%S",
    )

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger, log_path


def log_step(logger: logging.Logger, label: str, start: float, details: dict[str, Any] | None = None) -> float:
    elapsed_ms = (time.perf_counter() - start) * 1000
    payload = json.dumps(details, ensure_ascii=False) if details else ""
    if payload:
        logger.info("%s | ts=%s | elapsed_ms=%.2f | %s", label, now_iso(), elapsed_ms, payload)
    else:
        logger.info("%s | ts=%s | elapsed_ms=%.2f", label, now_iso(), elapsed_ms)
    return time.perf_counter()


def log_block(logger: logging.Logger, header: str, body: str) -> None:
    logger.info("%s START", header)
    if body == "":
        logger.info("<empty>")
    else:
        for line in body.splitlines() or [""]:
            logger.info("%s", line)
    logger.info("%s END", header)


def load_local_env(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_yaml_config(path: str) -> dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise SystemExit(f"Config file not found: {path}")
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"Config file '{path}' must be a mapping.")
    return data


def resolve_benchmark_and_provider(config: dict[str, Any], benchmark_profile_override: str | None) -> tuple[str, dict[str, Any]]:
    active = config.get("active") or {}
    benchmark_root = config.get("benchmark") or {}
    profiles = config.get("benchmark_profiles") or benchmark_root.get("profiles") or {}

    profile_name = (
        benchmark_profile_override
        or active.get("benchmark_profile")
        or active.get("active_profile")
        or benchmark_root.get("active_profile")
    )
    if not profile_name:
        raise SystemExit("No benchmark profile configured. Set --benchmark-profile or active.benchmark_profile.")

    profile = profiles.get(profile_name)
    if not profile:
        raise SystemExit(f"Benchmark profile '{profile_name}' not found in config.")

    dataset_path = str(profile.get("dataset_path", DEFAULT_DATASET)).strip()
    if not dataset_path:
        dataset_path = DEFAULT_DATASET

    llm_profile_name = profile.get("llm_profile") or profile.get("llm_profile_ref") or active.get("llm_profile")
    if not llm_profile_name:
        raise SystemExit(f"Benchmark profile '{profile_name}' missing llm_profile.")

    providers = {**DEFAULT_PROVIDERS, **(config.get("providers") or {})}
    llm_profiles = config.get("llm_profiles") or config.get("profiles") or {}
    llm_profile = llm_profiles.get(llm_profile_name)
    if not llm_profile:
        raise SystemExit(f"LLM profile '{llm_profile_name}' not found in config.")

    provider_name = llm_profile.get("provider")
    if not provider_name:
        raise SystemExit(f"LLM profile '{llm_profile_name}' missing provider.")
    provider_cfg = providers.get(provider_name, {})

    model = llm_profile.get("model")
    base_url = llm_profile.get("base_url") or provider_cfg.get("base_url")
    api_key_env = llm_profile.get("api_key_env") or provider_cfg.get("api_key_env")
    api_style = llm_profile.get("api_style") or provider_cfg.get("api_style", "chat_completions")

    if not model or not base_url or not api_key_env:
        raise SystemExit(
            f"LLM profile '{llm_profile_name}' missing required fields "
            "(model, base_url, api_key_env)."
        )

    return dataset_path, {
        "provider_name": provider_name,
        "llm_profile_name": llm_profile_name,
        "model": model,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "api_style": api_style,
    }


def read_first_record(path: str) -> dict[str, Any]:
    ds_path = Path(path)
    if not ds_path.exists():
        raise SystemExit(f"Dataset not found: {path}")
    with ds_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            return json.loads(line)
    raise SystemExit(f"No valid records in {path}")


def build_chat_messages(sample: dict[str, Any]) -> list[dict[str, str]]:
    raw_messages = sample.get("messages") or []
    messages: list[dict[str, str]] = []
    for msg in raw_messages:
        role = str(msg.get("role", "user")).strip()
        content = str(msg.get("content", "")).strip()
        if role in {"system", "user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    if not messages:
        return [{"role": "user", "content": str(sample)}]
    return messages


def build_responses_input(sample: dict[str, Any]) -> str:
    raw_messages = sample.get("messages") or []
    parts = []
    for msg in raw_messages:
        role = str(msg.get("role", "user")).strip()
        content = str(msg.get("content", "")).strip()
        if content:
            parts.append(f"{role}: {content}")
    return "\n\n".join(parts) if parts else str(sample)


def get_output_target(sample: dict[str, Any]) -> int | None:
    for key in (
        "max_tokens",
        "output_tokens",
        "max_output_tokens",
        "target_output_tokens",
        "output_tokens_requested",
    ):
        value = sample.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


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


def extract_output_text(response: Any, api_style: str) -> str:
    if api_style == "responses":
        direct = getattr(response, "output_text", None)
        if isinstance(direct, str) and direct.strip():
            return direct
        output = getattr(response, "output", None)
        if isinstance(output, list):
            chunks: list[str] = []
            for block in output:
                if isinstance(block, dict) and isinstance(block.get("content"), list):
                    for c in block.get("content", []):
                        if isinstance(c, dict) and isinstance(c.get("text"), str):
                            chunks.append(c["text"])
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    chunks.append(str(block["text"]))
            if chunks:
                return "\n".join(chunks)

    if api_style == "chat_completions":
        choices = getattr(response, "choices", None)
        if isinstance(choices, list) and choices:
            msg = getattr(choices[0], "message", None)
            if msg is not None:
                content = getattr(msg, "content", "")
                if isinstance(content, str):
                    return content

    raise RuntimeError("Unable to extract response text.")


def estimate_tokens(text: str, model: str | None = None) -> int:
    try:
        import tiktoken

        try:
            enc = tiktoken.encoding_for_model(model or "gpt-4o-mini")
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text or ""))
    except Exception:
        return len(re.findall(r"\S+", text or ""))


def run_sanity_check(logger: logging.Logger, args: argparse.Namespace) -> None:
    module_started = time.perf_counter()
    logger.info("MODULE INVOKED | ts=%s", now_iso())

    # Load environment
    t_env_start = time.perf_counter()
    log_step(logger, "LOAD_ENV_START", t_env_start)
    load_local_env()
    t_env_end = log_step(logger, "LOAD_ENV_END", t_env_start)

    # Load config + resolve benchmark profile + llm settings
    t_config_start = t_env_end
    config = load_yaml_config(args.config)
    t_config_end = log_step(logger, "LOAD_CONFIG", t_config_start, {"path": args.config})
    dataset_from_profile, llm_cfg = resolve_benchmark_and_provider(config, args.benchmark_profile)
    t_profile_end = log_step(
        logger,
        "RESOLVE_PROFILE",
        t_config_end,
        {
            "provider": llm_cfg["provider_name"],
            "llm_profile": llm_cfg["llm_profile_name"],
            "model": llm_cfg["model"],
            "api_style": llm_cfg["api_style"],
            "base_url": llm_cfg["base_url"],
            "dataset_from_profile": dataset_from_profile,
        },
    )

    dataset_path = args.dataset if args.dataset is not None else dataset_from_profile
    if not dataset_path:
        dataset_path = DEFAULT_DATASET
    logger.info("Using dataset: %s", dataset_path)

    # Read first dataset item
    t_dataset_start = t_profile_end
    log_step(logger, "READ_DATASET_START", t_dataset_start)
    sample = read_first_record(dataset_path)
    t_dataset_end = log_step(logger, "READ_DATASET_END", t_dataset_start)

    # Build prompt payload
    t_prompt_start = t_dataset_end
    log_step(logger, "BUILD_PROMPT_START", t_prompt_start)
    messages = build_chat_messages(sample)
    responses_input = build_responses_input(sample)
    output_target = get_output_target(sample)
    t_prompt_end = log_step(logger, "BUILD_PROMPT_END", t_prompt_start, {"sample_keys": sorted(sample.keys())})

    log_block(logger, "PROMPT_CHAT_MESSAGES", json.dumps(messages, ensure_ascii=False, indent=2))
    log_block(logger, "PROMPT_RESPONSES_INPUT", responses_input)

    # API client
    api_key = os.getenv(llm_cfg["api_key_env"])
    if not api_key:
        raise SystemExit(f"{llm_cfg['api_key_env']} is not set.")
    client = OpenAI(api_key=api_key, base_url=llm_cfg["base_url"])

    payload: dict[str, Any] = {"model": llm_cfg["model"], "timeout": args.timeout}
    if llm_cfg["api_style"] == "responses":
        payload["input"] = responses_input
        if output_target is not None and not args.no_max_completion_tokens:
            payload["max_output_tokens"] = output_target
    else:
        payload["messages"] = messages
        if output_target is not None and not args.no_max_completion_tokens:
            payload["max_completion_tokens"] = output_target

    if args.reasoning_effort:
        payload["reasoning_effort"] = args.reasoning_effort

    logger.info("LLM_REQUEST_PAYLOAD_KEYS=%s", sorted(payload.keys()))

    logger.info("TIMESTAMP_BEFORE_LLM_CALL | %s", now_iso())
    api_call_started = time.perf_counter()
    try:
        if llm_cfg["api_style"] == "responses":
            response = client.responses.create(**payload)
        else:
            response = client.chat.completions.create(**payload)
    except TypeError:
        fallback_payload = dict(payload)
        if "max_output_tokens" in fallback_payload:
            fallback_payload["max_tokens"] = fallback_payload.pop("max_output_tokens")
        if "max_completion_tokens" in fallback_payload:
            fallback_payload["max_tokens"] = fallback_payload.pop("max_completion_tokens")
        if llm_cfg["api_style"] == "responses":
            response = client.responses.create(**fallback_payload)
        else:
            response = client.chat.completions.create(**fallback_payload)

    api_call_after_request = time.perf_counter()
    logger.info("TIMESTAMP_AFTER_LLM_CALL_RETURN | %s", now_iso())

    # Parse result
    output = extract_output_text(response, llm_cfg["api_style"])
    result_parsed = time.perf_counter()
    logger.info("TIMESTAMP_AFTER_RESULT_RETURNED | %s", now_iso())

    prompt_tokens, completion_tokens_reported, total_tokens_reported = extract_usage_tokens(response)
    output_char_count = len(output)
    output_token_count_estimated = estimate_tokens(output, llm_cfg["model"])

    log_block(logger, "MODEL_OUTPUT", output)
    logger.info("OUTPUT_METRICS | chars=%d | reported_tokens=%d | estimated_tokens=%d", output_char_count, completion_tokens_reported, output_token_count_estimated)
    logger.info("USAGE_REPORT | input_tokens=%d | completion_tokens=%d | total_tokens=%d", prompt_tokens, completion_tokens_reported, total_tokens_reported)

    logger.info(
        "TIMING_MS | env=%.2f | load_config=%.2f | resolve_profile=%.2f | dataset=%.2f | prompt=%.2f | api=%.2f | parse=%.2f | total=%.2f",
        (t_env_end - t_env_start) * 1000,
        (t_config_end - t_config_start) * 1000,
        (t_profile_end - t_config_end) * 1000,
        (t_dataset_end - t_dataset_start) * 1000,
        (t_prompt_end - t_prompt_start) * 1000,
        (api_call_after_request - api_call_started) * 1000,
        (result_parsed - api_call_after_request) * 1000,
        (result_parsed - module_started) * 1000,
    )

    # Recompute precise phase durations from logged checkpoints.
    logger.info(
        "API_TIMING_DETAILS_MS | before_call=%.2f | call_return=%.2f | parse=%.2f",
        (api_call_started - module_started) * 1000,
        (api_call_after_request - api_call_started) * 1000,
        (result_parsed - api_call_after_request) * 1000,
    )
    logger.info("FULL_TIMELINE_COMPLETE | ts=%s", now_iso())


def main() -> None:
    args = parse_args()
    logger, log_path = setup_logger(args.log_file)
    logger.info("Logging to: %s", log_path)
    logger.info("Starting sanity checker with config=%s", args.config)
    run_sanity_check(logger, args)


if __name__ == "__main__":
    main()
