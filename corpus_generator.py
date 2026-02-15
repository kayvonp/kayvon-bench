#!/usr/bin/env python3
"""Generate synthetic summarization corpus records for benchmarking."""

from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None
from openai import OpenAI


DEFAULT_CONFIG_PATH = "corpus_generator_config.yaml"
DEFAULT_TAG = "meeting_notes"


@dataclass(frozen=True)
class CorpusGeneratorSettings:
    provider_name: str
    model: str
    base_url: str
    api_key_env: str
    input_tokens: int
    output_tokens: int
    records: int
    output_file: str
    default_tag: str
    concurrency: int
    max_attempts: int
    tolerance: float
    temperature: float
    seed: int | None
    timeout: float
    status_interval: float
    chunk_concurrency: int
    chunk_target: int
    max_chunks: int
    max_retries_per_record_multiplier: int
    rate_limit_trigger_count: int
    rate_limit_cooldown_seconds: float
    adaptive_ramp_waves: int
    growth_multiplier: float


def load_local_env(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'").strip('"'))


def load_yaml_config(path: str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise RuntimeError(f"Corpus generator config file '{path}' not found.")
    if yaml is None:
        raise RuntimeError("PyYAML is required. Install with: pip install pyyaml")

    raw = cfg_path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw) or {}

    if not isinstance(data, dict):
        raise RuntimeError(f"Config file '{path}' must contain a top-level mapping.")
    return data


def _require_dict(payload: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RuntimeError(f"Config section '{field_name}' must be a mapping.")
    return payload


def _require_str(payload: Any, field_name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(payload, str):
        raise RuntimeError(f"Config value '{field_name}' must be a string.")
    value = payload.strip()
    if not allow_empty and not value:
        raise RuntimeError(f"Config value '{field_name}' cannot be empty.")
    return value


def _require_int(payload: Any, field_name: str, *, minimum: int | None = None) -> int:
    if isinstance(payload, bool) or not isinstance(payload, int):
        raise RuntimeError(f"Config value '{field_name}' must be an integer.")
    if minimum is not None and payload < minimum:
        raise RuntimeError(f"Config value '{field_name}' must be >= {minimum}.")
    return payload


def _require_float(payload: Any, field_name: str, *, minimum: float | None = None) -> float:
    if not isinstance(payload, (int, float)):
        raise RuntimeError(f"Config value '{field_name}' must be a number.")
    value = float(payload)
    if minimum is not None and value < minimum:
        raise RuntimeError(f"Config value '{field_name}' must be >= {minimum}.")
    return value


def load_corpus_config(path: str = DEFAULT_CONFIG_PATH) -> CorpusGeneratorSettings:
    config = load_yaml_config(path)
    provider_cfg = _require_dict(config.get("provider"), "provider")
    generation_cfg = _require_dict(config.get("generation"), "generation")

    provider_name = _require_str(provider_cfg.get("name"), "provider.name")
    model = _require_str(provider_cfg.get("model"), "provider.model")
    base_url = _require_str(provider_cfg.get("base_url"), "provider.base_url")
    api_key_env = _require_str(provider_cfg.get("api_key_env"), "provider.api_key_env")

    input_tokens = _require_int(generation_cfg.get("input_tokens"), "generation.input_tokens", minimum=1)
    output_tokens = _require_int(generation_cfg.get("output_tokens"), "generation.output_tokens", minimum=1)
    records = _require_int(generation_cfg.get("records"), "generation.records", minimum=1)
    output_file_template = _require_str(
        generation_cfg.get("output_file"),
        "generation.output_file",
    )

    default_tag = _require_str(
        generation_cfg.get("default_tag", DEFAULT_TAG),
        "generation.default_tag",
    )
    concurrency = _require_int(generation_cfg.get("concurrency"), "generation.concurrency", minimum=1)
    max_attempts = _require_int(generation_cfg.get("max_attempts"), "generation.max_attempts", minimum=1)
    tolerance = _require_float(generation_cfg.get("tolerance"), "generation.tolerance", minimum=0.0)
    temperature = _require_float(generation_cfg.get("temperature"), "generation.temperature", minimum=0.0)

    seed_raw = generation_cfg.get("seed")
    if seed_raw is None:
        seed = None
    elif isinstance(seed_raw, bool) or not isinstance(seed_raw, int):
        raise RuntimeError("Config value 'generation.seed' must be an integer or null.")
    else:
        seed = seed_raw

    timeout = _require_float(generation_cfg.get("timeout"), "generation.timeout", minimum=0.0)
    status_interval = _require_float(
        generation_cfg.get("status_interval", 10.0),
        "generation.status_interval",
        minimum=0.1,
    )

    chunk_concurrency = _require_int(
        generation_cfg.get("chunk_concurrency", 8),
        "generation.chunk_concurrency",
        minimum=1,
    )
    chunk_target = _require_int(generation_cfg.get("section_chunk_target", 2200), "generation.section_chunk_target", minimum=250)
    max_chunks = _require_int(generation_cfg.get("max_sections", 40), "generation.max_sections", minimum=1)

    max_retries_per_record_multiplier = _require_int(
        generation_cfg.get("max_retries_per_record_multiplier", 4),
        "generation.max_retries_per_record_multiplier",
        minimum=1,
    )
    rate_limit_trigger_count = _require_int(
        generation_cfg.get("rate_limit_trigger_count", 4),
        "generation.rate_limit_trigger_count",
        minimum=1,
    )
    rate_limit_cooldown_seconds = _require_float(
        generation_cfg.get("rate_limit_cooldown_seconds", 0.75),
        "generation.rate_limit_cooldown_seconds",
        minimum=0.0,
    )
    adaptive_ramp_waves = _require_int(
        generation_cfg.get("adaptive_ramp_waves", 1),
        "generation.adaptive_ramp_waves",
        minimum=0,
    )
    growth_multiplier = _require_float(
        generation_cfg.get("growth_multiplier", 2.0),
        "generation.growth_multiplier",
        minimum=1.0,
    )

    try:
        output_file = output_file_template.format(input_tokens=input_tokens, output_tokens=output_tokens)
    except KeyError as exc:  # pragma: no cover
        raise RuntimeError(
            "generation.output_file must support placeholders {input_tokens} and {output_tokens}. "
            f"Failed on missing key: {exc}"
        )

    return CorpusGeneratorSettings(
        provider_name=provider_name,
        model=model,
        base_url=base_url,
        api_key_env=api_key_env,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        records=records,
        output_file=output_file,
        default_tag=default_tag,
        concurrency=concurrency,
        max_attempts=max_attempts,
        tolerance=tolerance,
        temperature=temperature,
        seed=seed,
        timeout=timeout,
        status_interval=status_interval,
        chunk_concurrency=chunk_concurrency,
        chunk_target=chunk_target,
        max_chunks=max_chunks,
        max_retries_per_record_multiplier=max_retries_per_record_multiplier,
        rate_limit_trigger_count=rate_limit_trigger_count,
        rate_limit_cooldown_seconds=rate_limit_cooldown_seconds,
        adaptive_ramp_waves=adaptive_ramp_waves,
        growth_multiplier=growth_multiplier,
    )


def estimate_tokens(text: str) -> int:
    words = re.findall(r"\b\w+\b", text)
    if not words:
        return 0
    char_based = len(text) / 4
    word_based = len(words) * 1.25
    return int((0.6 * char_based + 0.4 * word_based))


def build_prompt(
    target_tokens: int,
    attempt: int,
    observed: int | None = None,
    section_hint: int = 1,
    total_sections: int = 1,
    previous_context: str = "",
) -> str:
    base = (
        "Generate one synthetic but realistic meeting notes document for a workload benchmark. "
        "Return only the meeting notes text. Include date, time, attendees, agenda, notes, action items, and outcomes. "
        "Make the document internally coherent, domain-diverse, and suitable for summarization. "
    )
    if attempt == 0:
        return (
            f"{base}\nTarget length: about {target_tokens} tokens.\n"
            "Aim for around 95% to 105% of that target.\n"
            "Do not write a summary.\n"
        )
    direction = "shorter" if observed is not None and observed > target_tokens else "longer"
    delta = abs(target_tokens - (observed or target_tokens))
    section_prefix = f"This is section {section_hint} of {total_sections} in one meeting-document.\n"
    context_line = ""
    if previous_context:
        context_line = f"Previous section ending: {previous_context}\n\n"
    return (
        f"{section_prefix}"
        f"{context_line}"
        f"Attempt {attempt + 1}: adjust your length.\n"
        f"Previous attempt was about {observed} tokens; this version should be roughly {delta} tokens {direction}. "
        f"New target is still {target_tokens} tokens total.\n"
        f"{base}\n"
        "Do not write a summary and return only the meeting notes text.\n"
    )


def _request_segment(
    client: OpenAI,
    model: str,
    prompt: str,
    settings: CorpusGeneratorSettings,
    generation_seed: int | None,
    segment_target_tokens: int,
) -> tuple[str, int, int]:
    response_kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": "You are a synthetic data generator."},
            {"role": "user", "content": prompt},
        ],
        temperature=settings.temperature,
        timeout=settings.timeout,
    )
    # Prefer the newer API argument name first.
    segment_max = max(1024, int(segment_target_tokens * 1.25))
    response: Any
    try:
        response_kwargs["max_completion_tokens"] = segment_max
        response = client.chat.completions.create(**response_kwargs)
    except TypeError:
        response_kwargs.pop("max_completion_tokens", None)
        response_kwargs["max_tokens"] = segment_max
        response = client.chat.completions.create(**response_kwargs)

    content = response.choices[0].message.content or ""
    token_estimate = estimate_tokens(content)
    return content, token_estimate, int(getattr(response.usage, "completion_tokens", 0))


def _synthesize_section(
    client: OpenAI,
    model: str,
    section_no: int,
    section_target: int,
    total_sections: int,
    settings: CorpusGeneratorSettings,
    generation_seed: int | None,
    record_id: str,
) -> tuple[int, str, int, int]:
    prompt = build_prompt(
        target_tokens=section_target,
        attempt=0,
        section_hint=section_no,
        total_sections=total_sections,
    )
    content = ""
    segment_tokens = 0
    completion_tokens_total = 0
    accepted = False

    for attempt in range(settings.max_attempts):
        content, segment_tokens, completion_tokens = _request_segment(
            client=client,
            model=model,
            prompt=prompt,
            settings=settings,
            generation_seed=generation_seed,
            segment_target_tokens=section_target,
        )
        completion_tokens_total += completion_tokens
        completion_ratio_low = section_target * (1 - settings.tolerance)
        completion_ratio_high = section_target * (1 + settings.tolerance)
        if completion_ratio_low <= segment_tokens <= completion_ratio_high:
            accepted = True
            break

        if attempt + 1 < settings.max_attempts:
            prompt = build_prompt(
                target_tokens=section_target,
                attempt=attempt + 1,
                observed=segment_tokens,
                section_hint=section_no,
                total_sections=total_sections,
            )
            time.sleep(0.15)

    measured = estimate_tokens(content)
    print(
        f"{record_id}: section {section_no}/{total_sections} generated "
        f"(section_tokens={segment_tokens}, measured_tokens={measured}, accepted={accepted})",
        flush=True,
    )
    return section_no, content, measured, completion_tokens_total


def _strict_summary_prompt(target_tokens: int) -> str:
    return (
        "Summarize the following text into EXACTLY "
        f"{target_tokens} output tokens. Do not use more or fewer tokens.\n"
        "Return only the summary text. No preamble, no headings, no labels, no signoff.\n"
        "If you cannot hit the exact count, make the output as close as possible."
    )


def synthesize_meeting_notes(
    client: OpenAI,
    model: str,
    target_tokens: int,
    settings: CorpusGeneratorSettings,
    generation_seed: int | None,
    record_id: str,
) -> tuple[str, int, int]:
    lower = target_tokens * (1 - settings.tolerance)
    upper = target_tokens * (1 + settings.tolerance)
    chunk_target = min(settings.chunk_target, max(250, int(target_tokens / 8)))
    planned_sections = max(2, ceil(target_tokens / chunk_target))
    sections: list[str] = []
    section_count = 0
    measured_tokens = 0
    completion_tokens_total = 0

    while measured_tokens < lower and section_count < settings.max_chunks:
        section_workers = max(1, min(settings.chunk_concurrency, settings.max_chunks - section_count))
        target_sections = min(section_workers, max(1, planned_sections - section_count))
        if target_sections <= 0:
            target_sections = min(section_workers, settings.max_chunks - section_count)

        requested_total_sections = max(planned_sections, section_count + target_sections)

        if target_sections <= 0:
            break

        section_futures: list[Any] = []
        with ThreadPoolExecutor(max_workers=section_workers) as section_executor:
            for i in range(target_sections):
                section_no = section_count + i + 1
                section_target = chunk_target
                section_futures.append(
                    section_executor.submit(
                        _synthesize_section,
                        client=client,
                        model=model,
                        section_no=section_no,
                        section_target=section_target,
                        total_sections=requested_total_sections,
                        settings=settings,
                        generation_seed=(generation_seed + section_no) if generation_seed is not None else None,
                        record_id=record_id,
                    )
                )

            batch_outputs: dict[int, tuple[str, int, int]] = {}
            for fut in as_completed(section_futures):
                section_no, content, section_measured, section_completion = fut.result()
                batch_outputs[section_no] = (content, section_measured, section_completion)

        for section_no in sorted(batch_outputs):
            content, section_measured, section_completion = batch_outputs[section_no]
            sections.append(content)
            measured_tokens += section_measured
            completion_tokens_total += section_completion

        section_count += target_sections

        if section_count >= planned_sections and measured_tokens >= lower:
            break
        if section_count >= 1 and measured_tokens >= upper:
            break

    while measured_tokens > upper and sections:
        # Trim from the last section if we're over target by a wide margin.
        joined = "\n\n".join(sections)
        words = re.findall(r"\S+", joined)
        if len(words) < 2:
            break
        joined = " ".join(words[:-int(len(words) * 0.05)]) if len(words) > 50 else " ".join(words)
        sections = [joined]
        measured_tokens = estimate_tokens(joined)
        break

    note_text = "\n\n".join(sections)
    return note_text, measured_tokens, completion_tokens_total


def build_record(
    record_id: str,
    client: OpenAI,
    model: str,
    provider_name: str,
    settings: CorpusGeneratorSettings,
    generation_seed: int | None,
) -> dict[str, Any]:
    generated_text, measured_input_tokens, prompt_tokens = synthesize_meeting_notes(
        client=client,
        model=model,
        target_tokens=settings.input_tokens,
        settings=settings,
        generation_seed=generation_seed,
        record_id=record_id,
    )
    return {
        "id": record_id,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict summarization model. Obey the output-token budget exactly. "
                    "Never add explanations or extras outside the requested summary."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{_strict_summary_prompt(settings.output_tokens)}\n\n"
                    f"{generated_text}"
                ),
            },
        ],
        "task_type": "summarization",
        "input_len_est": measured_input_tokens,
        "target_output_tokens": settings.output_tokens,
        "max_tokens": settings.output_tokens,
        "tags": [settings.default_tag],
        "seed": generation_seed,
        "generator": {
            "type": "openai",
            "model": model,
            "provider": provider_name,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_tokens_requested": settings.input_tokens,
        "output_tokens_requested": settings.output_tokens,
        "prompt_tokens_estimate": prompt_tokens,
    }


@dataclass
class WorkerResult:
    index: int
    record: dict[str, Any] | None
    error: str | None
    rate_limited: bool = False


def run_generation(settings: CorpusGeneratorSettings) -> None:
    output_file = Path(settings.output_file)
    if output_file.suffix.lower() != ".jsonl":
        raise RuntimeError("Output file must have .jsonl extension.")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=os.environ[settings.api_key_env], base_url=settings.base_url)
    existing_count = 0
    if output_file.exists():
        try:
            with output_file.open("r", encoding="utf-8") as existing_f:
                for line in existing_f:
                    if line.strip():
                        existing_count += 1
        except OSError:
            existing_count = 0
    start_index = existing_count + 1

    def is_rate_limit_error(error_text: str | None) -> bool:
        if not error_text:
            return False
        lowered = error_text.lower()
        return (
            "429" in lowered
            or "rate limit" in lowered
            or "rate_limit" in lowered
            or "too many requests" in lowered
        )

    def worker(idx: int) -> WorkerResult:
        record_id = f"doc-{idx:04d}"
        local_seed = settings.seed + idx if settings.seed is not None else None
        try:
            record = build_record(
                record_id=record_id,
                client=client,
                model=settings.model,
                provider_name=settings.provider_name,
                settings=settings,
                generation_seed=local_seed,
            )
            return WorkerResult(index=idx, record=record, error=None, rate_limited=False)
        except Exception as exc:  # noqa: BLE001
            error_text = str(exc)
            return WorkerResult(
                index=idx,
                record=None,
                error=error_text,
                rate_limited=is_rate_limit_error(error_text),
            )

    results: list[tuple[int, dict[str, Any]]] = []
    completed = 0
    started_at = time.time()
    max_concurrency = max(1, settings.concurrency)
    current_concurrency = max_concurrency
    stable_waves = 0
    next_progress = started_at
    retry_queue: deque[int] = deque(range(start_index, start_index + settings.records))
    attempts: dict[int, int] = {}
    max_retries_per_record = max(1, settings.records * settings.max_retries_per_record_multiplier)
    cooldown_until = 0.0

    while retry_queue:
        now = time.time()
        if now < cooldown_until:
            time.sleep(max(0.05, cooldown_until - now))
            continue

        wave_size = min(current_concurrency, len(retry_queue))
        wave_indices: list[int] = []
        for _ in range(wave_size):
            idx = retry_queue.popleft()
            attempts[idx] = attempts.get(idx, 0) + 1
            if attempts[idx] > max_retries_per_record:
                raise RuntimeError(f"Record {idx} failed too many times; giving up.")
            wave_indices.append(idx)

        with ThreadPoolExecutor(max_workers=max(1, wave_size)) as executor:
            wave = {executor.submit(worker, idx): idx for idx in wave_indices}
            wave_rate_limit_hits = 0
            for fut in as_completed(wave):
                now = time.time()
                result = fut.result()
                if result.error is not None:
                    if result.rate_limited:
                        wave_rate_limit_hits += 1
                        retry_queue.append(result.index)
                        print(
                            f"Record {result.index} rate-limited; requeueing for retry (attempt {attempts[result.index]}).",
                            flush=True,
                        )
                    else:
                        raise RuntimeError(f"Record {result.index} failed: {result.error}")
                elif result.record is not None:
                    results.append((result.index, result.record))
                    completed += 1
                else:
                    raise RuntimeError(
                        f"Record {result.index} returned no content and no error; aborting."
                    )

                elapsed = max(0.0001, now - started_at)
                rate = completed / elapsed
                if elapsed >= next_progress:
                    queue = max(0, len(retry_queue))
                    print(
                        f"Generated {completed}/{settings.records} | elapsed={elapsed:.1f}s | rate={rate:.2f} rec/s | "
                        f"queue={queue} outstanding | target_concurrency={current_concurrency} | "
                        f"window_429={wave_rate_limit_hits}",
                        flush=True,
                    )
                    next_progress = elapsed + settings.status_interval

        if wave_rate_limit_hits >= settings.rate_limit_trigger_count:
            reduced = max(1, current_concurrency // 2)
            if reduced < current_concurrency:
                print(
                    f"Rate-limit pressure detected ({wave_rate_limit_hits} rate-limit errors). "
                    f"Reducing concurrency {current_concurrency} -> {reduced}",
                    flush=True,
                )
                current_concurrency = reduced
            cooldown_until = time.time() + settings.rate_limit_cooldown_seconds
            stable_waves = 0
        else:
            stable_waves += 1
            if current_concurrency < max_concurrency and stable_waves >= max(1, settings.adaptive_ramp_waves):
                next_concurrency = int(current_concurrency * settings.growth_multiplier)
                current_concurrency = min(
                    max_concurrency,
                    max(current_concurrency + 1, next_concurrency),
                )
                print(
                    f"No rate-limit signals detected for {settings.adaptive_ramp_waves} waves. "
                    f"Increasing concurrency to {current_concurrency}.",
                    flush=True,
                )
                stable_waves = 0

    print(f"Generation done: {len(results)}/{settings.records}", flush=True)

    with output_file.open("a", encoding="utf-8") as f:
        for _, record in sorted(results, key=lambda item: item[0]):
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")

    print(f"Wrote {len(results)} records to {output_file}")


def main() -> None:
    if len(sys.argv) > 1:
        raise SystemExit(
            "Corpus Generator is configured via corpus_generator_config.yaml and does not accept CLI arguments."
        )

    load_local_env()
    settings = load_corpus_config()
    api_key = os.getenv(settings.api_key_env)
    if not api_key:
        raise RuntimeError(f"Environment variable '{settings.api_key_env}' is not set.")
    run_generation(settings)


if __name__ == "__main__":
    main()
