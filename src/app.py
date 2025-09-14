"""
Drug Review Sentiment Classifier App
"""
import gradio as gr
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch

from logging_config import setup_logger
setup_logger("app.log")
from logging_config import info, error

# --------------------------
# Load tokenizer & LoRA model
# --------------------------
BASE_MODEL = "dmis-lab/biobert-base-cased-v1.1"
ADAPTER = "results/biobert-lora-adapters"   # <- change if loading from Drive or HF Hub

tokenizer = AutoTokenizer.from_pretrained(ADAPTER)
model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()
info(f"Loaded model {ADAPTER} with base {BASE_MODEL}")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# --------------------------
# Prediction functions
# --------------------------
def predict(review):
    """
    Predict sentiment for a single review.
    Returns a dictionary of probabilities and a final prediction string.
    """
    inputs = tokenizer(review, return_tensors="pt", truncation=True,
                       padding=True, max_length=128).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

    labels = ["Negative", "Positive"]
    prob_dict = {labels[i]: float(probs[i]) for i in range(len(labels))}
    final = f"Prediction: {labels[probs.argmax()]} (confidence {probs.max():.2f})"

    return prob_dict, final


def predict_batch(file):
    """
    Predict sentiment for a batch of reviews from a CSV file.
    """
    df = pd.read_csv(file.name)
    reviews = df["review"].tolist()
    preds = []
    for r in reviews:
        probs, final = predict(r)
        preds.append(final)
    df["prediction"] = preds
    return df

# --------------------------
# Gradio UI
# --------------------------
examples = [
    "This medicine helped me a lot, I feel much better now.",
    "The side effects were terrible and I had to stop using it.",
    "It worked okay but gave me headaches sometimes."
]

with gr.Blocks(theme="soft") as demo:
    gr.Markdown("# 💊 Drug Review Sentiment Classifier")
    gr.Markdown("""Fine-tuned BioBERT (LoRA) prototype. 
                ⚠️ This demo is for educational purposes only and
                does not provide medical advice.""")

    with gr.Tab("🔹 Single Review"):
        review_in = gr.Textbox(lines=4, 
                               placeholder="Write a drug review here...",
                               label="Review")
        btn1 = gr.Button("Classify")
        probs_out = gr.Label(num_top_classes=2, label="Probabilities")
        final_out = gr.Textbox(label="Prediction")
        btn1.click(fn=predict, inputs=review_in, outputs=[probs_out, final_out])

    with gr.Tab("📂 Batch Upload (CSV)"):
        gr.Markdown("Upload a CSV with a column named **review**.")
        file_in = gr.File(file_types=[".csv"], label="Upload CSV")
        df_out = gr.Dataframe(label="Predictions", wrap=True)
        file_in.upload(fn=predict_batch, inputs=file_in, outputs=df_out)

if __name__ == "__main__":
    demo.launch(share=True)
