# Drug Review Sentiment Analysis

This project is a mock healthcare case study using NLP to classify patient drug reviews as positive or negative by fine-tuning transformer models. It includes a training pipeline, reproducible outputs, and a small Gradio demo for quick experimentation.

> [!CAUTION] This repository has been created for educational purposes. Obviously, better solutions exist to perform drug review sentiment analysis, but the approaches here used were the ones that made more sense for the purpose.

---

## Dataset
- Source: Drug Review Dataset (Kaggle): https://www.kaggle.com/datasets/andrewmvd/drug-review-dataset
- Reviews include text, condition, and ratings (1–10). Ratings are binarized to sentiment: 0 = negative (≤5), 1 = positive (>5).
- This repo includes example TSVs under `data/`: `drugsComTrain_raw.tsv` and `drugsComTest_raw.tsv`.

---

## Models
- Works with any Hugging Face sequence classification backbone; tested with:
  - `dmis-lab/biobert-base-cased-v1.1`
  - `distilbert-base-uncased`
- Fine-tuning uses LoRA adapters to keep training light and fast.

---

## Quickstart
1) Install dependencies
   - `pip install -r requirements.txt`

2) Train a model
   - Minimal run: `python src/train_model.py --model distilbert-base-uncased --epochs 1 --batch-size 16`
   - Faster trial on a subset: add `--subset-frac 0.1`
   - Enable TensorBoard logging: add `--tensorboard`

3) Inspect outputs
   - TensorBoard: `tensorboard --logdir results/tensorboard`
   - MLflow UI (local files): artifacts tracked under `results/mlruns`
   - Trained LoRA adapters: saved under `results/outputs/<run_name>-adapters`
   - Confusion matrix: `results/outputs/<run_name>/cm.png`

4) Run the demo app (after training)
   - `python src/app.py`
   - Optional env vars:
     - `BASE_MODEL` — HF base model to load (default: `dmis-lab/biobert-base-cased-v1.1`)
     - `ADAPTER_DIR` — path to a specific adapters folder; if not set, the app picks the most recent `*-adapters` under `results/outputs/`

---

## CLI Options (`src/train_model.py`)
- `--model` (repeatable): HF model ID(s), e.g. `--model distilbert-base-uncased --model dmis-lab/biobert-base-cased-v1.1`
- `--epochs`: training epochs (default 3)
- `--batch-size`: per-device train batch size (default 16)
- `--imbalance`: `none` or `weighted_loss` (default `weighted_loss`)
- `--subset-frac`: float in (0,1) to subsample data for quick trials
- `--tensorboard`: enable TensorBoard logging under `results/tensorboard`
- `--fp16` / `--no-fp16`: force mixed precision on/off (defaults to CUDA availability or `TRAIN_FP16` env var)

---

## Repository Structure
```
drug-review-sentiment/
├── data/
│   ├── drugsComTrain_raw.tsv
│   └── drugsComTest_raw.tsv
├── notebooks/
│   ├── eda.ipynb
│   ├── modelling.ipynb
│   ├── run_pipeline.ipynb
│   └── experiments.ipynb
├── src/
│   ├── app.py                # Gradio demo for inference
│   ├── cleaning.py           # Reusable text-cleaning utilities
│   ├── config.py             # Centralized results/paths config
│   ├── load_datasets.py      # Load TSVs → HF DatasetDict
│   ├── logging_config.py     # Console+file logging helpers
│   └── train_model.py        # LoRA fine-tuning + metrics
├── requirements.txt
└── README.md
```

Outputs (created on first run):
- `results/outputs/` — training outputs and artifacts (incl. confusion matrix)
- `results/mlruns/` — MLflow tracking data (local file store)
- `results/tensorboard/` — TensorBoard event files when enabled
- `results/*.log` — training/app logs

You can change the base results directory by setting `RESULTS_DIR=/custom/path`.

---

## Notebooks
- `notebooks/eda.ipynb`: exploratory data analysis
- `notebooks/modelling.ipynb`, `notebooks/run_pipeline.ipynb`: example training runs with options (incl. TensorBoard)

---

## Notes & Tips
- Class imbalance: training uses weighted cross-entropy when `--imbalance weighted_loss`.
- Multi-run: pass multiple `--model` flags to train several backbones sequentially.
- Mixed precision: `--fp16` usually speeds up training on GPUs with minimal loss.
- Flagging in app: from the Single Review tab, click "Flag this prediction" to report issues; saved to `results/flags/flags.csv`.

---

## Disclaimer
This repository is for educational demonstration only and must not be used for clinical or real-world decision-making.

---

## Troubleshooting
- No adapters found when launching app: run training first, or set `ADAPTER_DIR` to a valid `*-adapters` folder under `results/outputs/`.
- Data files missing: ensure `data/drugsComTrain_raw.tsv` and `data/drugsComTest_raw.tsv` exist (see Kaggle link above).
- CUDA out of memory: reduce `--batch-size`, add `--subset-frac 0.1`, or use `--no-fp16` if mixed precision causes instability.
- Slow/large downloads: models/tokenizers are pulled from Hugging Face on first run; pre-download or pick a smaller model like `distilbert-base-uncased`.
- MLflow/TensorBoard paths: override the base outputs by setting `RESULTS_DIR=/custom/path`.
