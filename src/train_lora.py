import os, sys, mlflow, numpy as np
from dataclasses import dataclass
from typing import Dict, Any
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback, set_seed
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from src.load_datasets import main as load_local_csv

from src.logging_config import setup_logger, info, error

setup_logger("lora.log")



MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"
NUM_LABELS = 2
MAX_LEN = 128
SEED = 42
set_seed(SEED)


# --- Metrics ---
def compute_metrics(eval_pred):
    logits, y_true = eval_pred
    y_pred = np.argmax(logits, axis=-1)
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}


def main(
    model_name: str = MODEL_NAME,
    num_labels: int = NUM_LABELS,
    max_len: int = MAX_LEN,
    output_dir: str = "lora_model",
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    batch_size: int = 16,
    learning_rate: float = 5e-5,
    num_train_epochs: int = 3,
    weight_decay: float = 0.01,
    logging_dir: str = "logs",
    logging_steps: int = 10,
    evaluation_strategy: str = "steps",
    eval_steps: int = 50,
    save_steps: int = 50,
    save_total_limit: int = 2,
    load_best_model_at_end: bool = True,
    metric_for_best_model: str = "f1",
    greater_is_better: bool = True,
    early_stopping_patience: int = 3
):

    ds = load_local_csv()
    info("Datasets loaded successfully.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(
            batch["review"],
            truncation=True,
            padding="max_length",
            max_length=max_len
        )

    tokenized = ds.map(tokenize, batched=True)
    tokenized = tokenized.remove_columns([c for c in tokenized["train"].column_names if c not in ["input_ids","attention_mask","label"]])
    tokenized = tokenized.with_format("torch")
    info("Datasets tokenized.")

    # --- Base model + LoRA ---
    base = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    # Optional: freeze embeddings for extra speed
    base.bert.embeddings.requires_grad_(False)

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        task_type=TaskType.SEQ_CLS,
        target_modules=["query","value"]   # good default for BERT
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()

    run_name = "biobert-lora-128"
    out_dir = f"outputs/{run_name}"


    args = TrainingArguments(
        output_dir=out_dir,
        learning_rate=2e-4,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        gradient_accumulation_steps=2,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=True,
        logging_steps=100,
        weight_decay=0.01,
        fp16=True,
        report_to=["none"],  # Trainer logs; MLflow handled below
        seed=SEED
    )

    # --- MLflow ---
    info("Starting training, logging to MLflow...")
    mlflow.set_experiment("drug-review-sentiment")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model": MODEL_NAME, "lora_r": lora_cfg.r, "lora_alpha": lora_cfg.lora_alpha,
            "lora_dropout": lora_cfg.lora_dropout, "targets": ",".join(lora_cfg.target_modules),
            "lr": args.learning_rate, "max_len": MAX_LEN, "batch_train": args.per_device_train_batch_size,
            "grad_accum": args.gradient_accumulation_steps, "epochs": args.num_train_epochs, "seed": SEED
        })

        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=callbacks
        )

        trainer.train()
        metrics = trainer.evaluate(tokenized["test"])
        for k, v in metrics.items():
            mlflow.log_metric(f"test_{k}", float(v) if isinstance(v, (int, float)) else v)

        # Save adapters only
        adapters_dir = f"{out_dir}-adapters"
        os.makedirs(adapters_dir, exist_ok=True)
        model.save_pretrained(adapters_dir)
        tokenizer.save_pretrained(adapters_dir)  # store tokenizer alongside

        # Confusion matrix image (optional)
        try:
            import matplotlib.pyplot as plt
            y_logits = trainer.predict(tokenized["test"]).predictions
            y_pred = y_logits.argmax(axis=-1)
            y_true = tokenized["test"]["label"]
            from sklearn.metrics import ConfusionMatrixDisplay
            disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
            plt.title("Confusion Matrix (Test)")
            fig_path = os.path.join(out_dir, "cm.png")
            plt.savefig(fig_path, bbox_inches="tight")
            mlflow.log_artifact(fig_path)
        except Exception as e:
            info("CM plot failed:", e)

        info("Done. Adapters at:", adapters_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        error(f"Training failed: {e}")
        sys.exit(1)
