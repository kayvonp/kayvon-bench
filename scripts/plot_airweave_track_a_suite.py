import csv
import json
import os
from pathlib import Path
from typing import Any

import gspread
import matplotlib.pyplot as plt
from google.oauth2.service_account import Credentials


DEFAULT_SPREADSHEET_ID = "1RTTeX6_jh_GVz0FN6yfhEfbx36Gy-OkgEigy6EMMXgQ"
DEFAULT_WORKSHEET = "Summary"
DEFAULT_OUTPUT_DIR = Path("figures/airweave_track_a_suite")

# Verified on 2026-03-13 from Together's public model pages / serverless pricing docs:
# - GLM-5: https://www.together.ai/models/glm-5
# - Kimi K2.5: https://www.together.ai/models/kimi-k2-5
# - MiniMax M2.5: https://www.together.ai/models/minimax-m2-5
# - Qwen 3.5 397B A17B pricing: https://docs.together.ai/docs/serverless-models
MODEL_METADATA = {
    "airweave_march_2026_glm_track_a_retrieval_smoke": {
        "label": "GLM-5",
        "params_b": 744.0,
        "input_price_per_1m": 1.00,
        "output_price_per_1m": 3.20,
        "color": "#0b6e4f",
    },
    "airweave_march_2026_qwen_track_a_retrieval_smoke": {
        "label": "Qwen 3.5 397B",
        "params_b": 397.0,
        "input_price_per_1m": 0.70,
        "output_price_per_1m": 0.70,
        "color": "#7c3aed",
    },
    "airweave_march_2026_minimax_track_a_retrieval_smoke": {
        "label": "MiniMax M2.5",
        "params_b": 228.7,
        "input_price_per_1m": 0.30,
        "output_price_per_1m": 1.20,
        "color": "#d97706",
    },
    "airweave_march_2026_kimi_track_a_retrieval_smoke": {
        "label": "Kimi K2.5",
        "params_b": 1000.0,
        "input_price_per_1m": 0.50,
        "output_price_per_1m": 2.80,
        "color": "#2563eb",
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
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_google_service_account_info() -> dict[str, Any]:
    raw = (os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON") or "").strip()
    if not raw:
        raise SystemExit("GOOGLE_SERVICE_ACCOUNT_JSON is missing.")
    if raw.startswith("{"):
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise SystemExit("GOOGLE_SERVICE_ACCOUNT_JSON JSON blob is invalid.")
        return parsed
    path = Path(raw).expanduser()
    if not path.exists():
        raise SystemExit(f"Google service account file not found: {path}")
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise SystemExit("Google service account file does not contain a JSON object.")
    return parsed


def get_summary_rows() -> list[dict[str, Any]]:
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    creds = Credentials.from_service_account_info(load_google_service_account_info(), scopes=scopes)
    client = gspread.authorize(creds)
    sheet = client.open_by_key(DEFAULT_SPREADSHEET_ID)
    worksheet = sheet.worksheet(DEFAULT_WORKSHEET)
    return worksheet.get_all_records()


def to_float(row: dict[str, Any], key: str) -> float:
    value = row.get(key, "")
    text = str(value).strip().rstrip("%")
    return float(text) if text else 0.0


def latest_rows_for_profiles(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        profile = str(row.get("benchmark_profile", "")).strip()
        if profile not in MODEL_METADATA:
            continue
        samples = str(row.get("samples", "")).strip()
        if samples != "5":
            continue
        current = latest.get(profile)
        if current is None or str(row.get("timestamp_utc", "")) > str(current.get("timestamp_utc", "")):
            latest[profile] = row

    missing = [profile for profile in MODEL_METADATA if profile not in latest]
    if missing:
        raise SystemExit(f"Missing 5-sample summary rows for profiles: {', '.join(missing)}")

    ordered = []
    for profile, meta in MODEL_METADATA.items():
        row = dict(latest[profile])
        row["profile"] = profile
        row["label"] = meta["label"]
        row["params_b"] = meta["params_b"]
        row["color"] = meta["color"]
        input_price = meta["input_price_per_1m"]
        output_price = meta["output_price_per_1m"]
        avg_input = to_float(row, "avg_input_tok_total")
        avg_output = to_float(row, "avg_total_out_tok_usage")
        row["estimated_cost_per_request_usd"] = (
            (avg_input / 1_000_000.0) * input_price
            + (avg_output / 1_000_000.0) * output_price
        )
        ordered.append(row)
    return ordered


def ensure_output_dir() -> Path:
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_OUTPUT_DIR


def annotate_points(ax: Any, rows: list[dict[str, Any]], x_key: str, y_key: str) -> None:
    for row in rows:
        ax.annotate(
            row["label"],
            (float(row[x_key]), float(row[y_key])),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
        )


def save_summary_csv(rows: list[dict[str, Any]], out_dir: Path) -> Path:
    csv_path = out_dir / "airweave_track_a_latest_summary.csv"
    headers = [
        "label",
        "timestamp_utc",
        "samples",
        "params_b",
        "estimated_cost_per_request_usd",
        "success_rate",
        "ttft_p50_s",
        "ttft_p95_s",
        "lat_p50_s",
        "lat_p95_s",
        "avg_input_tok_total",
        "avg_total_out_tok_usage",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in headers})
    return csv_path


def plot_params_vs_ttft(rows: list[dict[str, Any]], out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    for ax, metric, title in [
        (axes[0], "ttft_p50_s", "TTFT p50"),
        (axes[1], "ttft_p95_s", "TTFT p95"),
    ]:
        for row in rows:
            ax.scatter(
                float(row["params_b"]),
                to_float(row, metric),
                s=130,
                color=row["color"],
                edgecolor="black",
                linewidth=0.8,
            )
        annotate_points(ax, rows, "params_b", metric)
        ax.set_xscale("log")
        ax.set_xlabel("Model size (billions of parameters, log scale)")
        ax.set_ylabel("Seconds")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    fig.suptitle("Airweave Track A: Model Size vs Time to First Token")
    path = out_dir / "params_vs_ttft.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_params_vs_e2e(rows: list[dict[str, Any]], out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    for ax, metric, title in [
        (axes[0], "lat_p50_s", "End-to-end latency p50"),
        (axes[1], "lat_p95_s", "End-to-end latency p95"),
    ]:
        for row in rows:
            ax.scatter(
                float(row["params_b"]),
                to_float(row, metric),
                s=130,
                color=row["color"],
                edgecolor="black",
                linewidth=0.8,
            )
        annotate_points(ax, rows, "params_b", metric)
        ax.set_xscale("log")
        ax.set_xlabel("Model size (billions of parameters, log scale)")
        ax.set_ylabel("Seconds")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    fig.suptitle("Airweave Track A: Model Size vs End-to-End Latency")
    path = out_dir / "params_vs_e2e_latency.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_cost_vs_latency(rows: list[dict[str, Any]], out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    for ax, metric, title in [
        (axes[0], "lat_p50_s", "Cost vs latency p50"),
        (axes[1], "lat_p95_s", "Cost vs latency p95"),
    ]:
        for row in rows:
            ax.scatter(
                float(row["estimated_cost_per_request_usd"]),
                to_float(row, metric),
                s=130,
                color=row["color"],
                edgecolor="black",
                linewidth=0.8,
            )
        annotate_points(ax, rows, "estimated_cost_per_request_usd", metric)
        ax.set_xlabel("Estimated cost per request (USD)")
        ax.set_ylabel("Seconds")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    fig.suptitle("Airweave Track A: Cost vs End-to-End Latency")
    path = out_dir / "cost_vs_latency.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_ttft_vs_e2e(rows: list[dict[str, Any]], out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    for ax, x_metric, y_metric, title in [
        (axes[0], "ttft_p50_s", "lat_p50_s", "p50 TTFT vs p50 E2E"),
        (axes[1], "ttft_p95_s", "lat_p95_s", "p95 TTFT vs p95 E2E"),
    ]:
        max_value = 0.0
        for row in rows:
            x = to_float(row, x_metric)
            y = to_float(row, y_metric)
            max_value = max(max_value, x, y)
            ax.scatter(
                x,
                y,
                s=130,
                color=row["color"],
                edgecolor="black",
                linewidth=0.8,
            )
        annotate_points(ax, rows, x_metric, y_metric)
        ax.plot([0, max_value * 1.05], [0, max_value * 1.05], linestyle="--", color="#666666", linewidth=1)
        ax.set_xlabel("TTFT (seconds)")
        ax.set_ylabel("End-to-end latency (seconds)")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    fig.suptitle("Airweave Track A: TTFT vs End-to-End Latency")
    path = out_dir / "ttft_vs_e2e_latency.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    load_local_env()
    out_dir = ensure_output_dir()
    rows = latest_rows_for_profiles(get_summary_rows())
    csv_path = save_summary_csv(rows, out_dir)
    plot_paths = [
        plot_params_vs_ttft(rows, out_dir),
        plot_params_vs_e2e(rows, out_dir),
        plot_cost_vs_latency(rows, out_dir),
        plot_ttft_vs_e2e(rows, out_dir),
    ]
    print(f"Wrote summary CSV: {csv_path}")
    for path in plot_paths:
        print(f"Wrote plot: {path}")


if __name__ == "__main__":
    main()
