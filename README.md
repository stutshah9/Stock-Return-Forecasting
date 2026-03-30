# S&P 500 Earnings Dataset Builder

This repository now does one job only: create reusable datasets for S&P 500 earnings-event research.

It extracts, normalizes, caches, and joins data from:

1. Hugging Face dataset `Bose345/sp500_earnings_transcripts`
2. `yfinance` for daily market data and basic company metadata
3. SEC `companyfacts` API or optional SEC bulk `companyfacts.zip`
4. Hugging Face dataset `emilpartow/reddit_finance_posts_sp500`
5. Hugging Face dataset `emilpartow/reddit_comments_sp500`

The main output is an event-level dataset with one row per transcript-backed earnings event. A company-level summary dataset is also produced.

## Repository Layout

```text
.
├── data/
│   ├── external/
│   ├── processed/
│   └── raw/
├── scripts/
│   └── build_dataset.sh
├── src/
│   └── data/
│       ├── build_dataset.py
│       ├── join_data.py
│       ├── load_reddit_comments.py
│       ├── load_reddit_posts.py
│       ├── load_sec_companyfacts.py
│       ├── load_transcripts.py
│       ├── load_yfinance.py
│       └── normalize.py
└── tests/
```

## What Gets Built

Running the builder writes these parquet files under `data/processed/`:

- `transcripts.parquet`
- `market_features.parquet`
- `sec_fundamentals.parquet`
- `reddit_posts.parquet`
- `reddit_comments.parquet`
- `company_level_dataset.parquet`
- `event_level_dataset.parquet`

Raw and cached source artifacts are written under `data/raw/`.

## Installation

Setup with `uv`:

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Setup with `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment Variables

Copy the example file and fill in the required values:

```bash
cp .env.example .env
```

The builder uses only these environment variables:

- `HF_TOKEN`
  Optional. Helps with Hugging Face rate limits and authenticated downloads.
- `SEC_USER_AGENT`
  Required for SEC access. Use a real contact string such as:
  `SEC_USER_AGENT="Stuti your_email@example.com"`
- `YFINANCE_PROXY`
  Optional. Only needed if your network requires a proxy for Yahoo Finance requests.

Load the variables into your shell before running:

```bash
set -a
source .env
set +a
```

## Running the Dataset Builder

The simplest entrypoint is:

```bash
python -m src.data.build_dataset
```

You can also use the shell wrapper:

```bash
bash scripts/build_dataset.sh
```

Useful optional arguments:

```bash
python -m src.data.build_dataset --start-date 2023-01-01 --end-date 2025-12-31
python -m src.data.build_dataset --max-tickers 25
python -m src.data.build_dataset --refresh
```

### Date-window behavior

If you do not pass `--start-date`, the builder defaults to the last 3 calendar years ending today. That keeps the Yahoo Finance pull tractable and makes the event-level join more coherent.

If you want a longer historical dataset, pass an explicit `--start-date`.

## Optional SEC Bulk Setup

The SEC API path works by default, but large builds are faster if you provide the bulk archive yourself.

If you download the SEC bulk companyfacts archive manually, place it at:

```text
data/external/sec/companyfacts.zip
```

The pipeline will use the bulk archive first and only fall back to live SEC API requests when needed.

## Output Schemas

### `transcripts.parquet`

Key columns:

- `event_id`
- `ticker`
- `company_name`
- `event_date`
- `event_datetime`
- `fiscal_year`
- `fiscal_quarter`
- `transcript_text`
- `transcript_word_count`
- `transcript_char_count`
- `transcript_source_dataset`

### `market_features.parquet`

Key columns:

- `event_id`
- `ticker`
- `event_date`
- `event_trading_date`
- `previous_close`
- `event_close`
- `next_trading_close`
- `log_return_target`
- `pre_return_5d`
- `pre_return_20d`
- `rolling_volatility_20d`
- `average_volume_20d`

### `sec_fundamentals.parquet`

Key columns:

- `event_id`
- `ticker`
- `cik`
- `event_date`
- `fiscal_year`
- `fiscal_quarter`
- `sec_fiscal_year`
- `sec_fiscal_quarter`
- `sec_filed_date`
- `sec_revenue`
- `sec_net_income`
- `sec_assets`
- `sec_liabilities`
- `sec_cash_and_cash_equivalents`
- `sec_shares_outstanding`
- `sec_eps`
- `fundamentals_available`

### `reddit_posts.parquet`

Key columns:

- `ticker`
- `post_id`
- `subreddit`
- `title`
- `body`
- `text`
- `created_timestamp`
- `score`
- `engagement`

### `reddit_comments.parquet`

Key columns:

- `ticker`
- `comment_id`
- `parent_post_id`
- `body`
- `created_timestamp`
- `score`

### `event_level_dataset.parquet`

One row per transcript-backed earnings event. This is the main modeling-ready dataset.

It joins:

- normalized transcript fields
- event-level market features from Yahoo Finance
- SEC companyfacts aligned to the event
- Reddit aggregates over 1-day, 3-day, and 7-day pre-event windows

Important Reddit aggregate columns include:

- `reddit_post_count_1d`, `reddit_post_count_3d`, `reddit_post_count_7d`
- `reddit_comment_count_1d`, `reddit_comment_count_3d`, `reddit_comment_count_7d`
- `reddit_post_score_mean_1d`, `reddit_post_score_mean_3d`, `reddit_post_score_mean_7d`
- `reddit_comment_score_mean_1d`, `reddit_comment_score_mean_3d`, `reddit_comment_score_mean_7d`
- `reddit_text_1d`, `reddit_text_3d`, `reddit_text_7d`

### `company_level_dataset.parquet`

One row per company with broadly useful summary information such as:

- transcript coverage
- Yahoo Finance metadata
- price history coverage
- latest aligned SEC values
- Reddit activity coverage

## Known Limitations

- The transcript dataset is the anchor table. If a company has market or Reddit data but no matching transcript event, it will not appear in the event-level dataset.
- `yfinance` event alignment uses the first trading session on or after `event_date` as the event close. The next trading session close becomes the target horizon.
- Pre-event market features are computed using data available strictly before the event trading session to reduce leakage when event timestamps are missing or ambiguous.
- SEC `companyfacts` coverage is uneven across concepts and companies. Missing SEC columns are expected.
- SEC fiscal-quarter matching prefers exact fiscal year and quarter when available, then falls back to the latest filed value available before the event date.
- Reddit coverage is incomplete and should be treated as a sparse auxiliary signal, not a complete record of investor discussion.
- Hugging Face dataset schemas may change slightly over time. The loaders use candidate column matching to stay resilient, but new schema changes may still require updates.

## Verification

Run the test suite with:

```bash
python -m pytest
```
