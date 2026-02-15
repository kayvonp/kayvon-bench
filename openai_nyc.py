import argparse
import os
from pathlib import Path
from typing import Any

from openai import OpenAI
import yaml


DEFAULT_PROMPT = "what are some fun things to do in new york city?"
DEFAULT_CONFIG_PATH = "llm_config.yaml"

DEFAULT_PROVIDERS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "model": "gpt-5-nano",
        "api_style": "responses",
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "api_key_env": "TOGETHER_API_KEY",
        "model": "kpirestani_0236/openai/gpt-oss-20b-398585b7",
        "api_style": "chat_completions",
    },
}


def load_local_env(path: str = ".env") -> None:
    """Load simple KEY=VALUE pairs from a local .env file if present."""
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
        return {"providers": {}, "profiles": {}}

    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"Config file '{path}' must contain a top-level mapping.")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send a prompt to an OpenAI-compatible inference endpoint."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to YAML config file (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--profile",
        help="Profile name from config to use. Defaults to active_profile in config.",
    )
    parser.add_argument(
        "--provider",
        help="Named provider preset to use (or 'custom').",
    )
    parser.add_argument(
        "--model",
        help="Model ID. Defaults to the selected provider's model.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt text to send to the model.",
    )
    parser.add_argument(
        "--base-url",
        help="Override API base URL (OpenAI-compatible).",
    )
    parser.add_argument(
        "--api-key-env",
        help="Override API key environment variable name.",
    )
    parser.add_argument(
        "--api-style",
        choices=["responses", "chat_completions"],
        help="Override API style for the selected endpoint.",
    )
    return parser.parse_args()


def resolve_runtime_settings(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, str]:
    providers = {**DEFAULT_PROVIDERS, **(config.get("providers") or {})}
    profiles = config.get("profiles") or {}

    profile_name = args.profile or config.get("active_profile")
    profile = {}
    if profile_name:
        profile = profiles.get(profile_name) or {}
        if not profile:
            raise SystemExit(f"Profile '{profile_name}' was not found in {args.config}.")

    provider_name = args.provider or profile.get("provider")
    if not provider_name:
        provider_name = "openai"

    provider_defaults = providers.get(provider_name) or {}

    base_url = args.base_url or profile.get("base_url") or provider_defaults.get("base_url")
    api_key_env = args.api_key_env or profile.get("api_key_env") or provider_defaults.get("api_key_env")
    model = args.model or profile.get("model") or provider_defaults.get("model")
    api_style = args.api_style or profile.get("api_style") or provider_defaults.get("api_style")

    if not api_style:
        api_style = "chat_completions"

    if provider_name == "custom":
        if not base_url:
            raise SystemExit("Custom provider requires base_url (profile or --base-url).")
        if not api_key_env:
            raise SystemExit("Custom provider requires api_key_env (profile or --api-key-env).")
        if not model:
            raise SystemExit("Custom provider requires model (profile or --model).")
    else:
        if not provider_defaults and not profile:
            raise SystemExit(
                f"Unknown provider '{provider_name}'. Add it in {args.config} or use one of: "
                f"{', '.join(sorted(DEFAULT_PROVIDERS))}, custom."
            )
        missing = []
        if not base_url:
            missing.append("base_url")
        if not api_key_env:
            missing.append("api_key_env")
        if not model:
            missing.append("model")
        if missing:
            raise SystemExit(f"Missing required setting(s) for '{provider_name}': {', '.join(missing)}.")

    return {
        "profile_name": profile_name or "",
        "provider_name": provider_name,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "model": model,
        "api_style": api_style,
    }


def main() -> None:
    print("Starting LLM client...")
    load_local_env()
    args = parse_args()
    config = load_yaml_config(args.config)
    settings = resolve_runtime_settings(args, config)

    if settings["profile_name"]:
        print(f"Profile: {settings['profile_name']}")
    else:
        print("Profile: (none)")
    print(f"Provider: {settings['provider_name']}")
    print(f"Model: {settings['model']}")

    api_key = os.getenv(settings["api_key_env"])
    if not api_key:
        raise SystemExit(f"{settings['api_key_env']} is not set (env var or .env).")

    client = OpenAI(api_key=api_key, base_url=settings["base_url"])
    print("Making LLM request...")

    if settings["api_style"] == "responses":
        response = client.responses.create(
            model=settings["model"],
            input=args.prompt,
        )
        output_text = response.output_text
    elif settings["api_style"] == "chat_completions":
        response = client.chat.completions.create(
            model=settings["model"],
            messages=[{"role": "user", "content": args.prompt}],
        )
        output_text = response.choices[0].message.content or ""
    else:
        raise SystemExit(f"Unsupported api_style: {settings['api_style']}")

    print(output_text)


if __name__ == "__main__":
    main()
