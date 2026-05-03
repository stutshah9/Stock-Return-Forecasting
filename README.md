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

- train years: `<= 2022`
- validation year: `2023`
- calibration year: `2024`
- test year: `2025`

The train/evaluate scripts read the year lists from `config.yaml`'s
`data.split` block. The default values in this repo are still:

- train years: `<= 2022`
- validation year: `2023`
- calibration year: `2024`
- test year: `2025`

## Quick Sanity Check

Run a dry-run training pass that uses a tiny synthetic dataset and only 2 batches of 2 samples:

```bash
cd /Users/Stuti/Stock-Return-Forecasting
python3 experiments/train.py --dry-run
```

This is the fastest way to verify model construction, forward pass, loss, and artifact writing.

To run full evaluation, you must have `data/events_cache.pt` available (see "Run End To End").

When the transcript encoder is frozen, transcript chunk embeddings are cached under
`data/transcript_cache/` by default so repeated epochs and evaluation passes do
not re-encode identical transcripts from scratch.

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
- events are split with current fixed code logic: train `<= 2022`, validation `2023`, calibration `2024`, test `2025`
- only the training split is used for gradient updates
- the validation split is used for early stopping
- the calibration split is reserved for conformal calibration during evaluation
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
- `event_conditioned_conformal`
- `normalized_conformal_modality`
- `same_ticker_baseline`

The main conformal comparison table is interval-focused and prints:

- `coverage_80`
- `coverage_90`
- `coverage_95`
- `avg_width_80`
- `avg_width_90`
- `avg_width_95`
- `avg_width`
- `calibration_error`

`MAE`, `RMSE`, and direction accuracy are still saved in the CSV for diagnostic
use, but they are not the headline comparison for conformal methods because
conformal calibration changes intervals rather than the center prediction.

It also saves subgroup-stratified metrics by surprise band, volatility regime,
and attention-volume band to:

```text
experiments/results_by_subgroup.csv
```

Selective prediction / abstention is kept as an optional add-on rather than a
default headline result. If you explicitly enable it in `config.yaml`, the
evaluation script will also write:

```text
experiments/results_selective.csv
```

And saves the CSV to:

```text
experiments/results.csv
```

It also exports per-event test predictions to:

```text
experiments/predictions.csv
```

Compatibility artifact also saved to project root:

```text
results.csv
```

The split tables described above are also written separately:

```text
experiments/results_point.csv       # MAE / RMSE / dir_acc per modality
experiments/results_intervals.csv   # coverage, width, calibration error per conformal method
results_point.csv                   # project-root mirrors of the same files
results_intervals.csv
```

## Conformal Calibration Method

The interval methods evaluated above all share the same multimodal quantile
center prediction `mu` and base interval `[q_low(x), q_high(x)]`; they differ
only in how the conformal correction is computed. Notation: a calibration
sample has nonconformity score
`s_i = max(q_low(x_i) - y_i, y_i - q_high(x_i))`, and the conformal threshold
`q_hat(alpha)` at coverage `alpha` is the finite-sample-corrected upper
quantile of the calibration scores at level `ceil((n+1) * alpha) / n`,
computed in [`calibration/conformal.py`](calibration/conformal.py).

**Global (naive) conformal.** Pool all calibration scores; produce
`[q_low(x) - q_hat, q_high(x) + q_hat]`. No conditioning.

**Event-conditioned conformal.** The conditioning variable is a
`(surprise band, volatility band)` regime computed from the cached event
features used by the model: `|SUE|` (standardized unexpected earnings) split
into `low / medium / high` by quantiles of the calibration distribution, and
`implied_vol` split into `low_vol / high_vol`. Bands and thresholds are
learned by `fit_regime_thresholds` on a pre-calibration split. For each
`(regime, alpha)` cell we collect the cell's calibration scores and take the
finite-sample-corrected quantile to obtain `q_hat(regime, alpha)`. **Fallback
when groups are small.** If a regime has fewer than `minimum_bucket_size`
calibration samples (default 24), we substitute the global `q_hat(alpha)`
computed from the pooled calibration set; the fallback decision is recorded
per-cell in `threshold_sources`. This avoids unstable per-regime quantile
estimates on thin buckets while still benefiting from conditioning where the
data supports it.

**Normalized conformal.** Rescales the conformal correction by a
per-event difficulty score `h(x)`. The calibration score is
`s_i / h(x_i)`, the threshold is a bias-corrected empirical target quantile of
the normalized scores, and the predicted interval is
`[q_low(x) - q_hat * h(x), q_high(x) + q_hat * h(x)]`. To keep `h(x)`
near `1` on average and to bound it away from zero or extreme values, we use
`h(x) = clip(raw(x) / median_cal(raw), 0.5, 2.0)`. The main reported
normalized method is:

- `normalized_conformal_modality` — `raw(x) = std(mu_text(x), mu_fin(x), mu_sent(x))`,
  where each `mu_m` is the midpoint of that modality-only interval,
  `(q_low,m + q_high,m) / 2`. Events whose text, financial, and sentiment
  branches conflict receive a larger correction.

All conformal variants use the same calibration / test split fixed by the
training pipeline, and all share the same point prediction `mu`, so MAE and
RMSE are identical across them. They are reported in the point-forecast
table only; the interval-comparison table reports coverage, width, and
calibration error. The pooled `naive_conformal` row keeps the conservative
finite-sample conformal quantile; the normalized row is tuned for the project
objective of getting empirical coverage closer to 80/90/95 while keeping
intervals narrow. The normalized tuning target subtracts `1.5` binomial
standard errors from the nominal level on the calibration set, which offsets
the overcoverage that otherwise made the adaptive rows collapse toward the
naive conservative intervals. After thresholds are fit on the calibration
split, a held-out validation split selects a separate correction multiplier for
each coverage level and interval mode by minimizing coverage error, with width
used as the tie-breaker.

The ablation table reports the raw multimodal quantile model, global conformal,
event-conditioned conformal, one normalized conformal method, and the
same-ticker baseline. The normalized method uses modality disagreement because
it is the strongest project-aligned difficulty score: intervals expand when the
text, financial, and sentiment branches disagree and shrink when the branches
align.

To print a few real-vs-predicted examples from the held-out test set:

```bash
python3 experiments/show_prediction_examples.py --year 2025 --ticker AAPL --date 2025-05-01 --limit 20
python3 experiments/show_prediction_examples.py --method all --year 2025 --ticker MSFT --limit 20
```

## Hyperparameter Sweep

To search for the strongest overall checkpoint under the proposal-style
validation objective instead of manually comparing runs:

```bash
cd /Users/Stuti/Stock-Return-Forecasting
python3 experiments/sweep.py
```

This sweep:

- perturbs learning rate plus the uncertainty/confidence alignment weights
- trains and evaluates each candidate config
- saves per-run configs, logs, and CSV outputs under `experiments/sweeps/...`
- ranks runs by validation `proposal_score`
- reruns the selected best config and saves its final artifacts under
  `experiments/sweeps/.../best/`

On Falcon, submit the sweep with:

```bash
sbatch scripts/falcon_sweep.sbatch
```

For a simple local frontend after evaluation:

```bash
python3 -m pip install gradio
python3 frontend/prediction_viewer.py
```

Then open the local URL Gradio prints and pick a company ticker plus event date.
The viewer centers the normalized modality-disagreement conformal calibration and shows:

- the calibrated 80/90/95 percent return range for the selected company event
- the model's expected return and the actual realized return
- whether the realized return landed inside the calibrated interval
- the reported-vs-estimated earnings values when they exist in `data/financials.csv`
- a side-by-side method comparison table for the same event

In the current evaluation code, `normalized_conformal_modality` is the proposed
modality-disagreement normalized conformal row. All conformal variants use the
same full multimodal point prediction; they differ only in how the interval
correction is calibrated.

Event-conditioned conformal calibration conditions on the observable earnings
event regime:

- surprise band from absolute standardized unexpected earnings (`low`,
  `medium`, or `high`)
- volatility band from implied volatility (`low_vol` or `high_vol`)

This creates six regimes such as `low_surprise_low_vol` and
`high_surprise_high_vol`. For each target coverage level, the calibration split
computes the CQR nonconformity score:

```text
max(q_low(x_i) - y_i, y_i - q_high(x_i))
```

The regime-specific threshold is the conservative conformal quantile of those
scores. If a regime bucket has too few calibration examples, the code falls back
to the global conformal threshold and records the threshold source in the
predictor diagnostics.

The normalized conformal rows implement the difficulty-scaled correction:

```text
s_i = max(q_low(x_i) - y_i, y_i - q_high(x_i)) / h(x_i)
interval = [q_low(x) - q_hat * h(x), q_high(x) + q_hat * h(x)]
```

The reported difficulty function is:

- `normalized_conformal_modality`: `h(x)` is disagreement among text-only,
  financial-only, and sentiment-only point predictions

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

## VT ARC

For VT ARC, the simplest path is:

1. Log in to ARC and load a Python module that includes `python3` and `pip`.
2. Create a virtual environment in your project or home directory.
3. Install the repo requirements.
4. Build the event cache on a CPU node if needed.
5. Train on a GPU node with `training.device: cuda`.
6. Evaluate from the saved checkpoint on a GPU or CPU node.

Example setup on an ARC login node:

```bash
cd $HOME
git clone <your-repo-url> Stock-Return-Forecasting
cd Stock-Return-Forecasting

module avail python
module load python

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

If `pip install -r requirements.txt` fails on the `AutoModel` line, remove that line from the requirements file or install the listed packages individually. `AutoModel` is imported from `transformers`; it is not a separate package.

Build the cache before training:

```bash
python3 data/build_cache.py
```

Then train with a Slurm batch script such as:

```bash
#!/bin/bash
#SBATCH --job-name=earnings-train
#SBATCH --account=<your_account>
#SBATCH --partition=<gpu_partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

cd $HOME/Stock-Return-Forecasting
source .venv/bin/activate

python3 experiments/train.py
```

And evaluate with:

```bash
#!/bin/bash
#SBATCH --job-name=earnings-eval
#SBATCH --account=<your_account>
#SBATCH --partition=<gpu_partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

cd $HOME/Stock-Return-Forecasting
source .venv/bin/activate

python3 experiments/evaluate.py
```

Submit and monitor jobs with:

```bash
mkdir -p logs
sbatch train.slurm
sbatch eval.slurm
squeue -u $USER
```

If you want a quick local sanity check on ARC before the full run:

```bash
python3 experiments/train.py --dry-run
```

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
rsync -avh --progress \
  --exclude='.venv' \
  --exclude='__pycache__' \
  --exclude='.pytest_cache' \
  --exclude='*.pyc' \
  --exclude='.git' \
  --exclude='logs' \
  --exclude='*.pt' \
  --exclude='predictions.csv' \
  --exclude='results.csv' \
  --exclude='results_by_subgroup.csv' \
  --exclude='.env' \
  --exclude='*.egg-info' \
  --exclude='experiments/sweeps' \
  --exclude='data/transcript_cache*' \
  /Users/Stuti/Stock-Return-Forecasting/ stutishah9@falcon2.arc.vt.edu:~/earnings_forecast/

# run on ARC
ssh stutishah9@falcon2.arc.vt.edu
srun --account=cp-spring2026-iac      --partition=v100_normal_q   --qos=fal_v100_normal_short    --cpus-per-task=1      --mem=32G      --time=01:00:00      --gres=gpu:v100:1      --pty bash -l
source .venv/bin/activate
cd earnings_forecast/
python3 experiments/train.py
python3 experiments/evaluate.py

# pull data from ARC
rsync -avh --progress stutishah9@falcon2.arc.vt.edu:~/earnings_forecast/ /Users/Stuti/Stock-Return-Forecasting/

# run frontend locally
python3 -m pip install gradio
python3 frontend/prediction_viewer.py
