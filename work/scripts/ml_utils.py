from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ── Paths ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = ROOT / "data" / "raw" / "content_refresh.parquet"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"
CHART_DIR = OUTPUT_DIR / "charts"


# ── Helpers ──────────────────────────────────────────────────────────────────

def ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def display_path(path: Path | str) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def normalize(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    minimum = values.min()
    maximum = values.max()
    if not np.isfinite(minimum) or not np.isfinite(maximum) or maximum == minimum:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - minimum) / (maximum - minimum)


def percentile_rank(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0)
    return values.rank(method="average", pct=True).fillna(0)


def precision_at_k(y_true, scores, k: int) -> float:
    frame = pd.DataFrame({"y": list(y_true), "score": list(scores)})
    if frame.empty:
        return 0.0
    top = frame.sort_values("score", ascending=False).head(min(k, len(frame)))
    return float(top["y"].mean()) if len(top) else 0.0


def safe_float(value, default: float = 0.0) -> float:
    try:
        number = float(value)
        if np.isfinite(number):
            return number
    except Exception:
        pass
    return default


def escape_xml(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def simple_svg_bar_chart(
    title: str,
    labels: list[str],
    values: list[float],
    path: Path,
    *,
    width: int = 960,
    height: int = 520,
    color: str = "#6F4E7C",
) -> None:
    """Create a simple SVG horizontal bar chart."""
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(label) for label in labels]
    values = [safe_float(value) for value in values]
    max_value = max(values) if values else 1
    max_value = max(max_value, 1)
    margin_left = 190
    margin_right = 40
    margin_top = 70
    margin_bottom = 50
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    bar_gap = 10
    bar_height = max(14, (plot_height - bar_gap * max(len(values) - 1, 0)) / max(len(values), 1))

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="34" text-anchor="middle" font-family="Arial" font-size="24" fill="#16232a">{escape_xml(title)}</text>',
    ]

    for index, (label, value) in enumerate(zip(labels, values)):
        y = margin_top + index * (bar_height + bar_gap)
        bar_width = (value / max_value) * plot_width
        lines.append(f'<text x="{margin_left - 12}" y="{y + bar_height * 0.65:.1f}" text-anchor="end" font-family="Arial" font-size="13" fill="#27343b">{escape_xml(label[:32])}</text>')
        lines.append(f'<rect x="{margin_left}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{color}" rx="4"/>')
        lines.append(f'<text x="{margin_left + bar_width + 8:.1f}" y="{y + bar_height * 0.65:.1f}" font-family="Arial" font-size="13" fill="#27343b">{value:,.3g}</text>')

    lines.append("</svg>")
    path.write_text("\n".join(lines))