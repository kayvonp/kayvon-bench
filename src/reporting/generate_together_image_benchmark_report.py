from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


@dataclass(frozen=True)
class ResultRow:
    rank: int
    display_name: str
    model: str
    avg_s: float
    p50_s: float
    p95_s: float
    success: str


REPORT_DATE = "March 18, 2026"
REPORT_TITLE = "Together Serverless Image Generation Latency Benchmark"
PROMPT = (
    "A premium product photo of a ceramic coffee mug on a wooden table, "
    "soft natural window light, clean background, high detail."
)

RESULTS = [
    ResultRow(
        rank=1,
        display_name="Qwen Image 2.0",
        model="Qwen/Qwen-Image-2.0",
        avg_s=7.53,
        p50_s=7.19,
        p95_s=9.27,
        success="3/3",
    ),
    ResultRow(
        rank=2,
        display_name="Seedream 4.0",
        model="ByteDance-Seed/Seedream-4.0",
        avg_s=8.40,
        p50_s=8.08,
        p95_s=9.01,
        success="3/3",
    ),
    ResultRow(
        rank=3,
        display_name="Ideogram 3.0",
        model="ideogram/ideogram-3.0",
        avg_s=8.67,
        p50_s=8.69,
        p95_s=8.71,
        success="3/3",
    ),
    ResultRow(
        rank=4,
        display_name="FLUX.2 Pro",
        model="black-forest-labs/FLUX.2-pro",
        avg_s=11.88,
        p50_s=11.39,
        p95_s=12.87,
        success="3/3",
    ),
    ResultRow(
        rank=5,
        display_name="GPT Image 1.5",
        model="openai/gpt-image-1.5",
        avg_s=21.55,
        p50_s=22.20,
        p95_s=24.47,
        success="3/3",
    ),
]


def build_markdown(contact_sheet_relpath: str) -> str:
    lines = [
        f"# {REPORT_TITLE}",
        "",
        f"Date: {REPORT_DATE}",
        "",
        "## Scope",
        "",
        "This report summarizes a rough latency benchmark for five image-generation models "
        "served via Together serverless endpoints. The goal was to estimate how long it takes "
        "to generate one image per model under matched input conditions.",
        "",
        "## Methodology",
        "",
        "- Benchmark runner: project `BenchmarkSuite`-style image suite added to `benchmarker.py`.",
        "- Endpoint/provider: Together serverless image generation API.",
        "- Workflow tested: text-to-image only.",
        "- Shared prompt:",
        f"  - `{PROMPT}`",
        "- Shared settings:",
        "  - Resolution: `1024x1024`",
        "  - Number of images: `n=1`",
        "  - Response format: `url`",
        "  - Model-specific quality controls left at defaults",
        "  - No explicit `steps` parameter was set",
        "- Validation process:",
        "  - Smoke test: 1 run per model to validate connectivity, auth, and model IDs",
        "  - Rough benchmark: 3 sequential runs per model",
        "- Measurement:",
        "  - Wall-clock latency per request",
        "  - Results summarized using average, p50, and p95 latency",
        "",
        "## Results",
        "",
        "| Rank | Model | Together model ID | Avg (s) | p50 (s) | p95 (s) | Success |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in RESULTS:
        lines.append(
            f"| {row.rank} | {row.display_name} | `{row.model}` | "
            f"{row.avg_s:.2f} | {row.p50_s:.2f} | {row.p95_s:.2f} | {row.success} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- These results are appropriate for a rough client-facing latency comparison, not a strict "
            "quality-normalized evaluation.",
            "- Qualitatively, all five models produced usable outputs for the shared prompt, but they differed "
            "somewhat in composition, realism, and stylistic choices.",
            "- In this run, `Qwen Image 2.0` was fastest and `GPT Image 1.5` was the slowest by a wide margin.",
            "",
            "## Contact Sheet",
            "",
            f"![Benchmark contact sheet]({contact_sheet_relpath})",
            "",
        ]
    )
    return "\n".join(lines)


def build_pdf(pdf_path: Path, contact_sheet_path: Path) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
    )

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="BodySmall",
            parent=styles["BodyText"],
            fontSize=10,
            leading=14,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionTitle",
            parent=styles["Heading2"],
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#1d1d1d"),
            spaceBefore=8,
            spaceAfter=8,
        )
    )

    story = [
        Paragraph(REPORT_TITLE, styles["Title"]),
        Spacer(1, 0.1 * inch),
        Paragraph(f"Date: {REPORT_DATE}", styles["BodySmall"]),
        Spacer(1, 0.12 * inch),
        Paragraph("Scope", styles["SectionTitle"]),
        Paragraph(
            "This report summarizes a rough latency benchmark for five image-generation models "
            "served via Together serverless endpoints. The purpose was to estimate the time "
            "required to generate one image per model under matched input conditions.",
            styles["BodySmall"],
        ),
        Paragraph("Methodology", styles["SectionTitle"]),
        Paragraph(
            "Benchmark runner: project image suite added to <font name='Courier'>benchmarker.py</font>.",
            styles["BodySmall"],
        ),
        Paragraph("Provider/endpoint: Together serverless image generation API.", styles["BodySmall"]),
        Paragraph("Workflow tested: text-to-image only.", styles["BodySmall"]),
        Paragraph(f"Shared prompt: <font name='Courier'>{PROMPT}</font>", styles["BodySmall"]),
        Paragraph(
            "Shared settings: 1024x1024 resolution, n=1, response_format=url, "
            "default model settings, and no explicit steps parameter.",
            styles["BodySmall"],
        ),
        Paragraph(
            "Validation process: 1-run smoke test per model followed by 3 sequential runs per model "
            "for the rough benchmark.",
            styles["BodySmall"],
        ),
        Paragraph(
            "Metric reported: wall-clock latency summarized as average, p50, and p95.",
            styles["BodySmall"],
        ),
        Spacer(1, 0.08 * inch),
        Paragraph("Latency Results", styles["SectionTitle"]),
    ]

    table_data = [
        ["Rank", "Model", "Together model ID", "Avg (s)", "p50 (s)", "p95 (s)", "Success"],
    ]
    for row in RESULTS:
        table_data.append(
            [
                str(row.rank),
                row.display_name,
                row.model,
                f"{row.avg_s:.2f}",
                f"{row.p50_s:.2f}",
                f"{row.p95_s:.2f}",
                row.success,
            ]
        )

    table = Table(
        table_data,
        colWidths=[0.45 * inch, 1.25 * inch, 2.55 * inch, 0.7 * inch, 0.7 * inch, 0.7 * inch, 0.7 * inch],
        repeatRows=1,
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2f3640")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("LEADING", (0, 0), (-1, -1), 11),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#f4efe8")]),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b8b0a6")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (0, 0), (0, -1), "CENTER"),
                ("ALIGN", (3, 1), (-1, -1), "RIGHT"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(table)
    story.extend(
        [
            Spacer(1, 0.18 * inch),
            Paragraph("Interpretation", styles["SectionTitle"]),
            Paragraph(
                "For this prompt and default-serving configuration, Qwen Image 2.0 was the fastest model, "
                "followed by Seedream 4.0 and Ideogram 3.0. FLUX.2 Pro was meaningfully slower, and GPT Image 1.5 "
                "was the slowest by a wide margin.",
                styles["BodySmall"],
            ),
            Paragraph(
                "These outputs were broadly comparable in quality, but not tightly quality-normalized. "
                "The benchmark is therefore best used as a rough latency comparison rather than a strict "
                "speed-for-equivalent-quality ranking.",
                styles["BodySmall"],
            ),
            PageBreak(),
            Paragraph("Benchmark Contact Sheet", styles["SectionTitle"]),
        ]
    )

    contact_sheet = Image(str(contact_sheet_path))
    max_width = doc.width
    max_height = doc.height - 0.3 * inch
    scale = min(max_width / contact_sheet.drawWidth, max_height / contact_sheet.drawHeight)
    contact_sheet.drawWidth *= scale
    contact_sheet.drawHeight *= scale
    story.append(contact_sheet)

    doc.build(story)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the Together image benchmark report.")
    parser.add_argument(
        "--repo-root",
        default=Path(__file__).resolve().parents[2],
        type=Path,
        help="Repository root path.",
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()

    contact_sheet_path = (
        repo_root / "output" / "image_benchmarks" / "20260318_rough_run" / "contact_sheet_20260318.png"
    )
    if not contact_sheet_path.exists():
        raise SystemExit(f"Contact sheet not found: {contact_sheet_path}")

    docs_dir = repo_root / "docs" / "reports"
    assets_dir = docs_dir / "assets"
    docs_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    markdown_path = docs_dir / "together_image_benchmark_report_20260318.md"
    pdf_path = docs_dir / "together_image_benchmark_report_20260318.pdf"
    json_path = docs_dir / "together_image_benchmark_report_20260318_results.json"
    tracked_contact_sheet_path = assets_dir / "contact_sheet_20260318.png"

    shutil.copy2(contact_sheet_path, tracked_contact_sheet_path)

    markdown_relpath = "assets/contact_sheet_20260318.png"
    markdown_path.write_text(build_markdown(markdown_relpath), encoding="utf-8")
    build_pdf(pdf_path, tracked_contact_sheet_path)
    json_path.write_text(
        json.dumps(
            {
                "report_date": REPORT_DATE,
                "title": REPORT_TITLE,
                "prompt": PROMPT,
                "settings": {
                    "width": 1024,
                    "height": 1024,
                    "n": 1,
                    "response_format": "url",
                    "steps": None,
                },
                "results": [row.__dict__ for row in RESULTS],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(markdown_path)
    print(pdf_path)
    print(json_path)
    print(tracked_contact_sheet_path)


if __name__ == "__main__":
    main()
