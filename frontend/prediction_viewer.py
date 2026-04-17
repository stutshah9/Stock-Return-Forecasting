"""Simple Gradio viewer for per-event test predictions."""

from __future__ import annotations

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
METHOD_ORDER = [
    "ours",
    "naive_conformal",
    "full_multimodal",
    "text_only",
    "financial_only",
    "sentiment_only",
    "same_ticker_baseline",
]


def _load_predictions() -> pd.DataFrame:
    for candidate in (
        PROJECT_ROOT / "experiments" / "predictions.csv",
        PROJECT_ROOT / "predictions.csv",
    ):
        if candidate.is_file():
            predictions = pd.read_csv(candidate)
            predictions["ticker"] = predictions["ticker"].astype(str).str.upper()
            predictions["date"] = predictions["date"].astype(str)
            for coverage in ("80", "90", "95"):
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


def _year_choices() -> list[str]:
    years = sorted(int(year) for year in PREDICTIONS["year"].dropna().unique().tolist())
    return [str(year) for year in years]


def _ticker_choices(year: str) -> list[str]:
    filtered = PREDICTIONS.copy()
    if year:
        filtered = filtered.loc[filtered["year"] == int(year)]
    return sorted(filtered["ticker"].dropna().astype(str).unique().tolist())


def _date_choices(year: str, ticker: str) -> list[str]:
    filtered = PREDICTIONS.copy()
    if year:
        filtered = filtered.loc[filtered["year"] == int(year)]
    if ticker:
        filtered = filtered.loc[filtered["ticker"] == ticker]
    return sorted(filtered["date"].dropna().astype(str).unique().tolist())


def _update_tickers(year: str) -> tuple[gr.Dropdown, gr.Dropdown]:
    tickers = _ticker_choices(year)
    first_ticker = tickers[0] if tickers else None
    dates = _date_choices(year, first_ticker) if first_ticker else []
    return (
        gr.Dropdown(choices=tickers, value=first_ticker, label="Ticker"),
        gr.Dropdown(choices=dates, value=dates[0] if dates else None, label="Date"),
    )


def _update_dates(year: str, ticker: str) -> gr.Dropdown:
    dates = _date_choices(year, ticker)
    return gr.Dropdown(choices=dates, value=dates[0] if dates else None, label="Date")


def _render_prediction(year: str, ticker: str, date: str) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    filtered = PREDICTIONS.copy()
    if year:
        filtered = filtered.loc[filtered["year"] == int(year)]
    if ticker:
        filtered = filtered.loc[filtered["ticker"] == ticker]
    if date:
        filtered = filtered.loc[filtered["date"] == date]

    if filtered.empty:
        empty_frame = pd.DataFrame()
        return "No matching prediction row found.", empty_frame, empty_frame

    method_rank = {method_name: index for index, method_name in enumerate(METHOD_ORDER)}
    filtered["_method_rank"] = filtered["method"].map(method_rank).fillna(len(method_rank))
    filtered = filtered.sort_values(["_method_rank", "method"]).reset_index(drop=True)
    row = filtered.iloc[0]
    actual_return = float(row["actual_return"])
    ours_row = filtered.loc[filtered["method"] == "ours"]
    ours_row = ours_row.iloc[0] if not ours_row.empty else row

    markdown = "\n".join(
        [
            f"### {row['ticker']} on {row['date']}",
            f"Actual return: `{actual_return:.4f}`",
            f"Regime: `{row['regime']}`",
            f"Ours 80% conformal range: `{ours_row['interval_80']}`",
            f"Ours 90% conformal range: `{ours_row['interval_90']}`",
            f"Ours 95% conformal range: `{ours_row['interval_95']}`",
        ]
    )

    compact_table = filtered[
        [
            "method",
            "predicted_return",
            "prediction_error",
            "interval_80",
            "interval_90",
            "interval_95",
        ]
    ].copy()
    compact_table["predicted_return"] = compact_table["predicted_return"].map(
        lambda value: f"{float(value):.4f}"
    )
    compact_table["prediction_error"] = compact_table["prediction_error"].map(
        lambda value: f"{float(value):.4f}"
    )

    detail_table = filtered[
        [
            "method",
            "coverage_80_lower",
            "coverage_80_upper",
            "coverage_90_lower",
            "coverage_90_upper",
            "coverage_95_lower",
            "coverage_95_upper",
            "width_90",
            "introspective_score",
        ]
    ].copy()
    for column_name in detail_table.columns:
        if column_name != "method":
            detail_table[column_name] = detail_table[column_name].map(
                lambda value: f"{float(value):.4f}"
            )

    return markdown, compact_table, detail_table


def build_app() -> gr.Blocks:
    default_year = _year_choices()[0]
    default_ticker_choices = _ticker_choices(default_year)
    default_ticker = default_ticker_choices[0] if default_ticker_choices else None
    default_date_choices = _date_choices(default_year, default_ticker) if default_ticker else []
    default_date = default_date_choices[0] if default_date_choices else None

    with gr.Blocks(title="Earnings Prediction Viewer") as app:
        gr.Markdown("## Earnings Prediction Viewer")
        gr.Markdown(
            "Pick a year, ticker, and event date to compare the real return with "
            "every prediction method side-by-side."
        )

        with gr.Row():
            year = gr.Dropdown(choices=_year_choices(), value=default_year, label="Year")
            ticker = gr.Dropdown(
                choices=default_ticker_choices,
                value=default_ticker,
                label="Ticker",
            )
            date = gr.Dropdown(choices=default_date_choices, value=default_date, label="Date")

        summary = gr.Markdown()
        ranges = gr.Dataframe(label="Predicted Return And Conformal Ranges", interactive=False)
        details = gr.Dataframe(label="Calibration Details", interactive=False)

        year.change(_update_tickers, inputs=year, outputs=[ticker, date])
        ticker.change(_update_dates, inputs=[year, ticker], outputs=date)

        for control in (year, ticker, date):
            control.change(
                _render_prediction,
                inputs=[year, ticker, date],
                outputs=[summary, ranges, details],
            )

        app.load(
            _render_prediction,
            inputs=[year, ticker, date],
            outputs=[summary, ranges, details],
        )

    return app


if __name__ == "__main__":
    build_app().launch()
