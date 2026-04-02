# Earnings Forecast

Multimodal next-day post-earnings return forecasting with:

- FinBERT transcript embeddings
- structured financial features
- social sentiment features
- cross-attention fusion
- Gaussian output head
- event-conditioned conformal calibration

## Setup

From the project root:

```bash
cd /Users/Stuti/Stock-Return-Forecasting
python3 -m pip install -r requirements.txt
```

If you use Reddit scraping through the public JSON endpoint, no API key is required.

## Project Workflow

The pipeline is split into:

1. `experiments/train.py`
   Trains the model on the chronological training split and saves:
   `experiments/model.pt`
   `data/cal_outputs.pt`

2. `experiments/evaluate.py`
   Loads the saved model and calibration outputs, calibrates conformal thresholds,
   evaluates on the held-out test split, prints a comparison table, and saves:
   `experiments/results.csv`

By default, the config is now set to:

- universe: `sp500`
- calibration years: `2024`
- test years: `2025`
- training years: everything earlier than the calibration/test years

## Quick Sanity Check

Run a dry-run that uses a tiny synthetic dataset and only 2 batches of 2 samples:

```bash
cd /Users/Stuti/Stock-Return-Forecasting
python3 experiments/train.py --dry-run
python3 experiments/evaluate.py
```

This is the fastest way to verify the pipeline end to end.

## Collect Real Financials

To build [financials.csv](data/financials.csv) from real earnings dates and price history:

```bash
cd /Users/Stuti/Stock-Return-Forecasting
python3 data/collect_real_data.py
```

This collector:

- defaults to the cached S&P 500 ticker list in `data/sp500_tickers.csv`
- defaults to years `2021 2022 2023 2024 2025`
- writes and updates `data/financials.csv`
- uses `yfinance` for earnings dates plus trailing price-based metrics
- resumes from existing CSVs and skips tickers that already cover the requested years

For a quick smoke test on a few names:

```bash
python3 data/collect_real_data.py --tickers AAPL MSFT GOOGL --years 2024 2025
```

The resulting `financials.csv` includes:

- `earnings_surprise`
- `estimated_earnings`
- `actual_earnings`
- `sue`
- `implied_vol`
- `momentum`

## Download Real Transcripts

To populate [transcripts.csv](data/transcripts.csv) from Hugging Face:

```bash
cd /Users/Stuti/Stock-Return-Forecasting
python3 data/download_transcripts_hf.py
```

This importer:

- reads `HF_TOKEN` from `.env` in either:
  - the parent directory (`/Users/Stuti/.env`)
  - this project root (`/Users/Stuti/Stock-Return-Forecasting/.env`)
- pulls from `Bose345/sp500_earnings_transcripts`
- maps `symbol -> ticker`, `date -> date`, and `content -> text`
- filters to S&P 500 names and years `2021 2022 2023 2024 2025`
- appends into `data/transcripts.csv` and de-duplicates on `(ticker, date)`

Those two files are what the training and evaluation scripts use to construct real events.

## Train On Training Data

To train on the real chronological training split:

```bash
cd /Users/Stuti/Stock-Return-Forecasting
python3 experiments/train.py
```

What happens:

- local events are loaded from the transcript and financial CSV indexes
- events are filtered to the configured universe, currently `sp500`
- events are split by year from `config.yaml`
- only the training split is used for gradient updates
- the calibration split is never used during training
- model weights are saved to `experiments/model.pt`
- calibration outputs are saved to `data/cal_outputs.pt`

Important:

- the sample repo data only contains a tiny 2023 example
- real training for a 2025 test set requires many more rows in:
  `data/transcripts.csv`
  `data/financials.csv`
- those files must contain enough S&P 500 events across earlier years, 2024 for calibration, and 2025 for testing

## Test / Evaluate On Held-Out Data

After training finishes, evaluate on the held-out test split:

```bash
cd /Users/Stuti/Stock-Return-Forecasting
python3 experiments/evaluate.py
```

This evaluates:

- `text_only`
- `financial_only`
- `sentiment_only`
- `full_multimodal`
- `naive_conformal`
- `ours`
- `same_ticker_baseline`

The script prints a results table with:

- `coverage_80`
- `coverage_90`
- `coverage_95`
- `avg_width`
- `MAE`
- `RMSE`
- `dir_acc`

And saves the CSV to:

```text
experiments/results.csv
```

If you intentionally want the old synthetic smoke-test behavior when real 2025 data is unavailable, run:

```bash
python3 experiments/evaluate.py --allow-synthetic-fallback
```

## Mass Train And Test

If you want the full train-then-test workflow, run:

```bash
cd /Users/Stuti/Stock-Return-Forecasting
python3 experiments/train.py
python3 experiments/evaluate.py
```

Device selection is controlled in [config.yaml](config.yaml) with `training.device`.

- `auto`: prefers `cuda`, then `mps`, then `cpu`
- `mps`: use Apple GPU acceleration on Apple Silicon Macs
- `cuda`: use an NVIDIA GPU
- `cpu`: force CPU execution

For VT ARC or any NVIDIA-backed environment, set:

```yaml
training:
  device: cuda
```

That is the standard "train on training data, calibrate on calibration data, test on held-out data" workflow for this repo.

## Run All Checks

To run every phase test script:

```bash
cd /Users/Stuti/Stock-Return-Forecasting
for test_file in tests/test_phase*.py; do
  python3 "$test_file"
done
```

## Important Files

- `config.yaml`: hyperparameters and split settings
- `data/transcripts.csv`: transcript source data
- `data/financials.csv`: structured feature source data
- `data/loader.py`: event construction
- `data/dataset.py`: dataset wrapper
- `models/fusion_model.py`: multimodal model
- `calibration/conformal.py`: conformal interval logic
- `experiments/train.py`: training entry point
- `experiments/evaluate.py`: evaluation entry point
