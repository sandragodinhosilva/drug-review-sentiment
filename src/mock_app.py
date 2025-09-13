import gradio as gr
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt

# 🔹 Mock model (replace with your BioBERT + LoRA later)
sentiment = pipeline("sentiment-analysis")

# ---------- Functions ----------
def predict_single(review):
    result = sentiment(review)[0]
    label = result["label"]
    score = round(result["score"], 4)
    probs = {"Positive": 0.0, "Negative": 0.0}
    probs[label] = score
    return probs, f"Prediction: {label} (confidence {score:.2f})"

def predict_batch(file):
    df = pd.read_csv(file.name)
    reviews = df["review"].tolist()
    preds = [sentiment(r)[0]["label"] for r in reviews]
    df["prediction"] = preds
    return df


# ---------- Gradio UI ----------
with gr.Blocks(theme="soft") as demo:
    gr.Markdown("# 💊 Drug Review Sentiment Classifier")
    gr.Markdown("Fine-tuned BioBERT (LoRA) prototype. "
                "⚠️ This demo is for educational purposes only and does not provide medical advice.")

    with gr.Tab("🔹 Single Review"):
        review_in = gr.Textbox(lines=4, placeholder="Type a drug review here...", label="Review")
        btn1 = gr.Button("Classify")
        probs_out = gr.Label(num_top_classes=2, label="Probabilities")
        final_out = gr.Textbox(label="Prediction")
        btn1.click(fn=predict_single, inputs=review_in, outputs=[probs_out, final_out])

    with gr.Tab("📂 Batch Upload (CSV)"):
        gr.Markdown("Upload a CSV with a column named **review**.")
        file_in = gr.File(file_types=[".csv"], label="Upload CSV")
        df_out = gr.Dataframe(label="Predictions", wrap=True)
        file_in.upload(fn=predict_batch, inputs=file_in, outputs=df_out)
        
		
#demo.launch(share=True, allow_flagging="never")

if __name__ == "__main__":
    demo.launch() # share=True