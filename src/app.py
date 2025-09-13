import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch

# 🔹 Caminhos
base_ckpt = "dmis-lab/biobert-base-cased-v1.1"
lora_dir = "results/biobert-lora-adapters"  # pasta onde salvaste os adapters

# 🔹 Carregar modelo + adapters
tokenizer = AutoTokenizer.from_pretrained(base_ckpt)
base_model = AutoModelForSequenceClassification.from_pretrained(base_ckpt, num_labels=2)
model = PeftModel.from_pretrained(base_model, lora_dir)
model.eval()

# 🔹 Função de predição
def predict(review):
    inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    sentiment = "Positive" if probs[1] > probs[0] else "Negative"
    return { "Positive": float(probs[1]), "Negative": float(probs[0]), "Prediction": sentiment }

# 🔹 Interface Gradio
demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=4, placeholder="Enter a drug review here..."),
    outputs=[
        gr.Label(num_top_classes=2),
        gr.Textbox(label="Predicted Sentiment")
    ],
    title="Drug Review Sentiment Classifier",
    description="Fine-tuned BioBERT with LoRA (PEFT)."
)

if __name__ == "__main__":
    demo.launch()