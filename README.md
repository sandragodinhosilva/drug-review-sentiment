# Drug Review Sentiment Analysis

> [!CAUTION] This repository has been created for educational purposes. Obviously, better solutions exist to perform drug review sentiment analysis, but the approaches here used were the ones that made more sense for the purpose.

This project is a **mock case study in healthcare**, exploring the use of **Natural Language Processing (NLP)** to analyze patient reviews of medications.
The goal is to classify a review as **positive** or **negative**, using a **fine-tuned transformer model**.

---

## Dataset
- **Source**: [Drug Review Dataset (Kaggle)](https://www.kaggle.com/datasets/andrewmvd/drug-review-dataset)
- Contains patient-written reviews of various drugs, along with ratings and conditions.
- For this project, ratings were binarized into **positive vs. negative sentiment**.

---

## Model
- **Base model**: [DistilBERT (Hugging Face)](https://huggingface.co/distilbert-base-uncased)
- Fine-tuned for binary sentiment classification.
- Chosen for being lightweight and efficient while still maintaining strong performance.

---

## Repository Structure
```
drug-review-sentiment/
├── notebooks/ # Exploratory Data Analysis (EDA)
│ └── eda.ipynb
├── src/ # Core pipeline scripts
│ ├── data_processing.py # Data loading & tokenization
│ ├── train.py # Fine-tuning DistilBERT
│ ├── evaluate.py # Model evaluation & metrics
├── results/ # Outputs (metrics, confusion matrix)
│ ├── metrics.json
│ └── confusion_matrix.png
├── requirements.txt # Dependencies
└── README.md # Project documentation
```
---
## How to Run

1. Clone repository:
```bash
git clone https://github.com/<your-username>/drug-review-sentiment.git
cd drug-review-sentiment
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run training:
```bash
python src/train.py
```

---
## Results

**Exploratory Data Analysis (EDA):** (see notebooks/eda.ipynb)

**Sentiment distribution: **(to be added)

**Performance metrics: **stored in results/metrics.json

**Confusion matrix:** available at results/confusion_matrix.png

---
## TensorBoard

Training can optionally log to TensorBoard under `results/tensorboard/<run_name>`.

- CLI
  - Enable: `python src/train_model.py --model distilbert-base-uncased --epochs 1 --batch-size 16 --tensorboard`
  - Launch UI: `tensorboard --logdir results/tensorboard`

- Notebook
  - Use `notebooks/run_pipeline.ipynb`, which passes `use_tensorboard=True` and includes a cell to launch TensorBoard.
  - Or call `train_model(..., use_tensorboard=True)` and then run `%tensorboard --logdir results/tensorboard`.

Notes
- All outputs are centralized under `results/` (artifacts: `results/outputs/`, MLflow: `results/mlruns`, logs: `results/*.log`).
- Override base directory by setting env var `RESULTS_DIR=/custom/path`.

---
### ⚠️ **Disclaimer**

This repository is for educational demonstration only and should not be used for any clinical or decision-making purposes.
