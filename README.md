# Drug Review Sentiment Analysis

This repository contains a **mock case study in healthcare**, focusing on **sentiment analysis of patient drug reviews**.  
The goal is to predict whether a patient's opinion about a medication is **positive** or **negative**, using fine-tuned LLMs.

Dataset used: [Drug Review Dataset](https://www.kaggle.com/datasets/andrewmvd/drug-review-dataset) from Kaggle.

Model used: [DistilBERT](https://huggingface.co/distilbert-base-uncased) from Hugging Face.

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


## Results

### Exploratory Data Analysis of drug_review data

TODO

### Sentiment Distribution

TODO

## Future Work

TODO