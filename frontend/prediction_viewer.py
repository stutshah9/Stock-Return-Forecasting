"""Company-first Gradio viewer for calibrated earnings-return predictions."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

try:
    import gradio as gr
except ImportError as exc:  # pragma: no cover - optional UI dependency
    raise SystemExit(
        "Gradio is required for the prediction viewer. "
        "Install it with `python3 -m pip install gradio`."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OURS_METHOD = "ours"
METHOD_ORDER = [
    "ours",
    "ours_explanation_augmented",
    "naive_conformal",
    "full_multimodal",
    "text_only",
    "financial_only",
    "sentiment_only",
    "same_ticker_baseline",
]
COVERAGE_OPTIONS = ["80", "90", "95"]
APP_CSS = """
.gradio-container {
    background:
        radial-gradient(circle at top left, rgba(236, 253, 245, 0.95), transparent 32%),
        radial-gradient(circle at top right, rgba(254, 249, 195, 0.9), transparent 28%),
        linear-gradient(180deg, #f8fafc 0%, #eef6f2 100%);
    font-family: "Avenir Next", "Segoe UI", sans-serif;
}
#summary-card {
    background: rgba(255, 255, 255, 0.82);
    border: 1px solid rgba(15, 23, 42, 0.08);
    border-radius: 18px;
    box-shadow: 0 18px 44px rgba(15, 23, 42, 0.08);
    padding: 10px 18px;
}
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 10px;
    margin: 14px 0 6px 0;
}
.metric-card {
    background: linear-gradient(180deg, rgba(248, 250, 252, 0.95), rgba(241, 245, 249, 0.92));
    border: 1px solid rgba(148, 163, 184, 0.20);
    border-radius: 14px;
    padding: 12px 14px;
}
.metric-label {
    color: #475569;
    font-size: 0.82rem;
    margin-bottom: 6px;
}
.metric-value {
    color: #0f172a;
    font-size: 1.1rem;
    font-weight: 700;
}
.status-pill {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
    margin-right: 8px;
    margin-bottom: 8px;
}
.status-yes {
    background: rgba(22, 163, 74, 0.12);
    color: #166534;
}
.status-no {
    background: rgba(220, 38, 38, 0.10);
    color: #991b1b;
}
.status-neutral {
    background: rgba(148, 163, 184, 0.16);
    color: #334155;
}
.table-card table {
    table-layout: fixed;
    width: 100%;
}
.table-card th,
.table-card td {
    white-space: normal !important;
    overflow-wrap: anywhere;
    word-break: break-word;
    font-size: 0.83rem;
    line-height: 1.25;
}
@media (max-width: 900px) {
    .metric-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}
"""


def _load_predictions() -> pd.DataFrame:
    for candidate in (
        PROJECT_ROOT / "experiments" / "predictions.csv",
        PROJECT_ROOT / "predictions.csv",
    ):
        if candidate.is_file():
            predictions = pd.read_csv(candidate)
            predictions["ticker"] = predictions["ticker"].astype(str).str.upper()
            predictions["date"] = predictions["date"].astype(str)
            predictions["method"] = predictions["method"].astype(str)
            if "expected_return" not in predictions.columns and "predicted_return" in predictions.columns:
                predictions["expected_return"] = predictions["predicted_return"]
            for coverage in COVERAGE_OPTIONS:
                lower_col = f"coverage_{coverage}_lower"
                upper_col = f"coverage_{coverage}_upper"
                interval_col = f"interval_{coverage}"
                width_col = f"width_{coverage}"
                if (
                    interval_col not in predictions.columns
                    and lower_col in predictions.columns
                    and upper_col in predictions.columns
                ):
                    predictions[interval_col] = predictions.apply(
                        lambda row: f"[{float(row[lower_col]):.4f}, {float(row[upper_col]):.4f}]",
                        axis=1,
                    )
                if (
                    width_col not in predictions.columns
                    and lower_col in predictions.columns
                    and upper_col in predictions.columns
                ):
                    predictions[width_col] = (
                        predictions[upper_col].astype(float) - predictions[lower_col].astype(float)
                    )
            return predictions
    raise FileNotFoundError(
        "predictions.csv not found. Run `python3 experiments/evaluate.py` first."
    )


PREDICTIONS = _load_predictions()


def _is_missing(value: object) -> bool:
    return pd.isna(value) or (isinstance(value, float) and math.isnan(value))


def _format_decimal(value: object, digits: int = 4) -> str:
    if _is_missing(value):
        return "N/A"
    return f"{float(value):.{digits}f}"


def _format_percent(value: object, digits: int = 2) -> str:
    if _is_missing(value):
        return "N/A"
    return f"{100.0 * float(value):.{digits}f}%"


def _format_method_name(value: object) -> str:
    method = str(value)
    mapping = {
        "ours": "Ours",
        "ours_explanation_augmented": "Ours + Explanation",
        "naive_conformal": "Naive Conformal",
        "full_multimodal": "Full Multimodal",
        "text_only": "Text Only",
        "financial_only": "Financial Only",
        "sentiment_only": "Sentiment Only",
        "same_ticker_baseline": "Same-Ticker Baseline",
    }
    return mapping.get(method, method.replace("_", " ").title())


def _format_regime(value: object) -> str:
    return str(value).replace("_", " ").title()


def _company_choices() -> list[str]:
    ours_rows = PREDICTIONS.loc[PREDICTIONS["method"] == OURS_METHOD]
    source = ours_rows if not ours_rows.empty else PREDICTIONS
    return sorted(source["ticker"].dropna().astype(str).unique().tolist())


def _company_frame(company: str) -> pd.DataFrame:
    ours_rows = PREDICTIONS.loc[PREDICTIONS["method"] == OURS_METHOD].copy()
    if ours_rows.empty:
        ours_rows = PREDICTIONS.copy()
    if company:
        ours_rows = ours_rows.loc[ours_rows["ticker"] == company]
    return ours_rows.sort_values("date", ascending=False).reset_index(drop=True)


def _date_choices(company: str) -> list[str]:
    frame = _company_frame(company)
    return frame["date"].dropna().astype(str).tolist()


def _update_dates(company: str) -> gr.Dropdown:
    dates = _date_choices(company)
    return gr.Dropdown(choices=dates, value=dates[0] if dates else None, label="Event Date")


def _interval_hit(row: pd.Series, coverage: str) -> str:
    lower = row.get(f"coverage_{coverage}_lower")
    upper = row.get(f"coverage_{coverage}_upper")
    actual = row.get("actual_return")
    if _is_missing(lower) or _is_missing(upper) or _is_missing(actual):
        return "N/A"
    return "Yes" if float(lower) <= float(actual) <= float(upper) else "No"


def _company_history(company: str, coverage: str) -> pd.DataFrame:
    frame = _company_frame(company)
    if frame.empty:
        return pd.DataFrame()

    history = pd.DataFrame(
        {
            "Date": frame["date"].astype(str),
            "Regime": frame["regime"].map(_format_regime),
            "Expected": frame["expected_return"].map(_format_decimal),
            "Actual": frame["actual_return"].map(_format_decimal),
            f"{coverage}% Range": frame[f"interval_{coverage}"].astype(str),
            "Inside": frame.apply(lambda row: _interval_hit(row, coverage), axis=1),
        }
    )

    if "estimated_earnings" in frame.columns and "actual_earnings" in frame.columns:
        history["Est. EPS"] = frame["estimated_earnings"].map(_format_decimal)
        history["Rpt. EPS"] = frame["actual_earnings"].map(_format_decimal)
    return history


def _method_comparison(company: str, date: str, coverage: str) -> pd.DataFrame:
    frame = PREDICTIONS.copy()
    if company:
        frame = frame.loc[frame["ticker"] == company]
    if date:
        frame = frame.loc[frame["date"] == date]
    if frame.empty:
        return pd.DataFrame()

    method_rank = {method_name: index for index, method_name in enumerate(METHOD_ORDER)}
    frame["_method_rank"] = frame["method"].map(method_rank).fillna(len(method_rank))
    frame = frame.sort_values(["_method_rank", "method"]).reset_index(drop=True)

    comparison = pd.DataFrame(
        {
            "Method": frame["method"].map(_format_method_name),
            "Expected": frame["expected_return"].map(_format_decimal),
            "Actual": frame["actual_return"].map(_format_decimal),
            f"{coverage}% Range": frame[f"interval_{coverage}"].astype(str),
            "Width": frame[f"width_{coverage}"].map(_format_decimal),
            "Direction": frame["direction_match"].map(
                lambda value: "Yes" if int(float(value)) == 1 else "No"
            ),
        }
    )
    return comparison


def _event_row(company: str, date: str) -> pd.Series | None:
    if not company or not date:
        return None
    frame = _company_frame(company)
    frame = frame.loc[frame["date"] == date]
    if frame.empty:
        return None
    return frame.iloc[0]


def _detail_markdown(row: pd.Series, coverage: str) -> str:
    lower = row[f"coverage_{coverage}_lower"]
    upper = row[f"coverage_{coverage}_upper"]
    expected_return = row.get("expected_return", row.get("predicted_return"))
    actual_return = row.get("actual_return")
    inside_range = _interval_hit(row, coverage)
    inside_class = "status-neutral"
    if inside_range == "Yes":
        inside_class = "status-yes"
    elif inside_range == "No":
        inside_class = "status-no"

    estimated_earnings = row.get("estimated_earnings")
    actual_earnings = row.get("actual_earnings")
    earnings_surprise = row.get("earnings_surprise")
    earnings_markup = ""
    if not (
        _is_missing(estimated_earnings)
        and _is_missing(actual_earnings)
        and _is_missing(earnings_surprise)
    ):
        earnings_markup = (
            "<div class='metric-grid'>"
            f"<div class='metric-card'><div class='metric-label'>Estimated EPS</div><div class='metric-value'>{_format_decimal(estimated_earnings)}</div></div>"
            f"<div class='metric-card'><div class='metric-label'>Reported EPS</div><div class='metric-value'>{_format_decimal(actual_earnings)}</div></div>"
            f"<div class='metric-card'><div class='metric-label'>Earnings Surprise</div><div class='metric-value'>{_format_decimal(earnings_surprise)}</div></div>"
            f"<div class='metric-card'><div class='metric-label'>Regime</div><div class='metric-value'>{_format_regime(row['regime'])}</div></div>"
            "</div>"
        )

    return (
        f"## {row['ticker']} | {row['date']}\n"
        f"<span class='status-pill {inside_class}'>Inside {coverage}% range: {inside_range}</span>"
        f"<span class='status-pill status-neutral'>Coverage: {coverage}%</span>"
        f"<span class='status-pill status-neutral'>Confidence: {_format_decimal(row.get('introspective_score'))}</span>"
        "<div class='metric-grid'>"
        f"<div class='metric-card'><div class='metric-label'>Expected Return</div><div class='metric-value'>{_format_decimal(expected_return)}</div></div>"
        f"<div class='metric-card'><div class='metric-label'>Actual Return</div><div class='metric-value'>{_format_decimal(actual_return)}</div></div>"
        f"<div class='metric-card'><div class='metric-label'>Lower Bound</div><div class='metric-value'>{_format_decimal(lower)}</div></div>"
        f"<div class='metric-card'><div class='metric-label'>Upper Bound</div><div class='metric-value'>{_format_decimal(upper)}</div></div>"
        "</div>"
        + earnings_markup
    )


def _details_table(row: pd.Series, coverage: str) -> pd.DataFrame:
    expected_return = row.get("expected_return", row.get("predicted_return"))
    return pd.DataFrame(
        [
            {
                "Company": str(row["ticker"]),
                "Date": str(row["date"]),
                "Coverage": f"{coverage}%",
                "Expected": _format_decimal(expected_return),
                "Actual": _format_decimal(row.get("actual_return")),
                "Lower": _format_decimal(row.get(f"coverage_{coverage}_lower")),
                "Upper": _format_decimal(row.get(f"coverage_{coverage}_upper")),
                "Width": _format_decimal(row.get(f"width_{coverage}")),
                "Inside": _interval_hit(row, coverage),
                "Est. EPS": _format_decimal(row.get("estimated_earnings")),
                "Rpt. EPS": _format_decimal(row.get("actual_earnings")),
                "Surprise": _format_decimal(row.get("earnings_surprise")),
            }
        ]
    )


def _render_company_view(
    company: str,
    date: str,
    coverage: str,
) -> tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    row = _event_row(company, date)
    if row is None:
        empty_frame = pd.DataFrame()
        return (
            "No matching company event found.",
            empty_frame,
            empty_frame,
            empty_frame,
        )

    return (
        _detail_markdown(row, coverage),
        _details_table(row, coverage),
        _company_history(company, coverage),
        _method_comparison(company, date, coverage),
    )


def build_app() -> gr.Blocks:
    default_company_choices = _company_choices()
    default_company = default_company_choices[0] if default_company_choices else None
    default_date_choices = _date_choices(default_company) if default_company else []
    default_date = default_date_choices[0] if default_date_choices else None

    with gr.Blocks(title="Company Calibration Viewer", css=APP_CSS) as app:
        gr.Markdown("# Company Calibration Viewer")
        gr.Markdown(
            "Pick a company to inspect the proposal-style calibrated return range, "
            "the model's expected return, the realized return, and the supporting "
            "earnings context for each event."
        )

        with gr.Row():
            company = gr.Dropdown(
                choices=default_company_choices,
                value=default_company,
                label="Company",
            )
            date = gr.Dropdown(
                choices=default_date_choices,
                value=default_date,
                label="Event Date",
            )
            coverage = gr.Dropdown(
                choices=COVERAGE_OPTIONS,
                value="90",
                label="Coverage",
            )

        summary = gr.Markdown(elem_id="summary-card")
        selected_event = gr.Dataframe(
            label="Selected Event Summary",
            interactive=False,
            elem_classes=["table-card"],
        )

        history = gr.Dataframe(
            label="Company History (`ours`)",
            interactive=False,
            elem_classes=["table-card"],
        )
        comparison = gr.Dataframe(
            label="Method Comparison",
            interactive=False,
            elem_classes=["table-card"],
        )

        company.change(_update_dates, inputs=company, outputs=date)
        for control in (company, date, coverage):
            control.change(
                _render_company_view,
                inputs=[company, date, coverage],
                outputs=[summary, selected_event, history, comparison],
            )

        app.load(
            _render_company_view,
            inputs=[company, date, coverage],
            outputs=[summary, selected_event, history, comparison],
        )

    return app


if __name__ == "__main__":
    build_app().launch()
