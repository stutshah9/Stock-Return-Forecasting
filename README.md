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
  Trains the model on cached events and saves:
  `experiments/model.pt`
  `data/cal_outputs.pt`

  For compatibility with older tests/scripts, it also writes:
  `model.pt`
  `cal_outputs.pt`

2. `experiments/evaluate.py`
  Loads cached events plus saved model/calibration outputs, calibrates conformal
  thresholds, evaluates on the held-out test split, prints a comparison table,
  and saves:
  `experiments/results.csv`

  For compatibility with older tests/scripts, it also writes:
  `results.csv`

Current split behavior in code:

- train years: `<= 2023`
- calibration year: `2024`
- test year: `2025`

Note: while `config.yaml` contains a `data.split` block, current train/evaluate
scripts use the fixed year split above.

## Quick Sanity Check

Run a dry-run training pass that uses a tiny synthetic dataset and only 2 batches of 2 samples:

```bash
cd /Users/Stuti/Stock-Return-Forecasting
python3 experiments/train.py --dry-run
```

This is the fastest way to verify model construction, forward pass, loss, and artifact writing.

To run full evaluation, you must have `data/events_cache.pt` available (see "Run End To End").

## Run End To End

Use this sequence from a clean clone:

```bash
cd /Users/Stuti/Stock-Return-Forecasting
python3 -m pip install -r requirements.txt

# Optional: refresh source CSVs
python3 data/collect_real_data.py
python3 data/download_transcripts_hf.py

# Build event cache used by train/evaluate
python3 data/build_cache.py

# Train + evaluate
python3 experiments/train.py
python3 experiments/evaluate.py
```

Expected outputs after successful run:

- `experiments/model.pt`
- `data/cal_outputs.pt`
- `experiments/results.csv`

Compatibility outputs also written:

- `model.pt`
- `cal_outputs.pt`
- `results.csv`

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

- cached events are loaded from `data/events_cache.pt`
- events are filtered to the configured universe, currently `sp500`
- events are split with current fixed code logic: train `<= 2023`, calibration `2024`, test `2025`
- only the training split is used for gradient updates
- the calibration split is never used during training
- model weights are saved to `experiments/model.pt`
- calibration outputs are saved to `data/cal_outputs.pt`

Compatibility artifacts also saved to project root:

- `model.pt`
- `cal_outputs.pt`

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

Compatibility artifact also saved to project root:

```text
results.csv
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
- `data/build_cache.py`: builds `data/events_cache.pt`
- `data/loader.py`: event construction
- `data/dataset.py`: dataset wrapper
- `models/fusion_model.py`: multimodal model
- `calibration/conformal.py`: conformal interval logic
- `experiments/train.py`: training entry point
- `experiments/evaluate.py`: evaluation entry point

# upload data to ARC
rsync -avh --progress stutishah9@falcon2.arc.vt.edu:~/earnings_forecast/

# run on ARC
- ssh stutishah9@falcon2.arc.vt.edu
- srun --account=cp-spring2026-iac      --partition=l40s_normal_q      --cpus-per-task=1      --mem=32G      --time=01:00:00      --gres=gpu:l40s:1      --pty bash -l
- source .venv/bin/activate
- cd earnings_forecast/
- python3 experiments/train.py
- python3 experiments/evaluate.py

# pull data from ARC
rsync -avh --progress stutishah9@falcon2.arc.vt.edu:~/earnings_forecast/ /Users/Stuti/Stock-Return-Forecasting/
