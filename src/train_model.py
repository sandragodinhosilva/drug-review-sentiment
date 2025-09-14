import os, sys, argparse, mlflow, numpy as np
from typing import Any
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback, set_seed
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from src.load_datasets import main as load_local_csv
from src.logging_config import setup_logger, info, error

setup_logger("train_model.log")

# --- Global ---
SEED = 42
set_seed(SEED)
NUM_LABELS = 2
MAX_LEN = 128


# --- Metrics ---
def compute_metrics(eval_pred):
    logits, y_true = eval_pred
    y_pred = np.argmax(logits, axis=-1)
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}


# --- Imbalance utilities ---
def _compute_class_weights(labels: np.ndarray, num_labels: int = NUM_LABELS) -> torch.Tensor:
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=num_labels).astype(float)
    counts[counts == 0] = 1.0
    total = counts.sum()
    weights = total / (num_labels * counts)
    return torch.tensor(weights, dtype=torch.float32)


class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        # Avoid boolean ops on tensors; explicitly pick key
        labels = inputs.get("labels", inputs.get("label"))
        outputs = model(**inputs)
        logits = getattr(outputs, "logits", None)
        if labels is not None and logits is not None:
            if self.class_weights is not None:
                cw = self.class_weights.to(logits.device)
                loss_fct = torch.nn.CrossEntropyLoss(weight=cw)
            else:
                loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss = getattr(outputs, "loss", None)
        return (loss, outputs) if return_outputs else loss


# --- LoRA target selection ---
def guess_lora_targets(model) -> list[str]:
    """Guess sensible LoRA target modules automatically."""
    candidate_keywords = [
        "q_proj", "k_proj", "v_proj", "out_proj",  # RoBERTa/DeBERTa style
        "q_lin", "k_lin", "v_lin", "o_lin",          # DistilBERT style
        "query", "key", "value"                       # BERT style
    ]
    found = set()
    for name, _ in model.named_modules():
        for kw in candidate_keywords:
            if kw in name:
                found.add(kw)
    found_sorted = sorted(found)
    if not found_sorted:
        # Sensible default for BERT-like models
        return ["query", "value"]
    return found_sorted


# --- Main training ---
def train_model(
    model_name: str,
    batch_size: int = 16,
    num_train_epochs: int = 3,
    imbalance_strategy: str = "weighted_loss",  # 'none' | 'weighted_loss'
    fp16: bool | None = None,
):
    ds = load_local_csv()
    info("Datasets loaded successfully.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(
            batch["review"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
        )

    tokenized = ds.map(tokenize, batched=True)
    keep_cols = ["input_ids", "attention_mask", "label"]
    tokenized = tokenized.remove_columns([c for c in tokenized["train"].column_names if c not in keep_cols])
    tokenized = tokenized.rename_column("label", "labels")  # standardize
    tokenized = tokenized.with_format("torch")
    info("Datasets tokenized.")

    # --- Base model ---
    base = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=NUM_LABELS)

    if hasattr(base, "bert"):
        base.bert.embeddings.requires_grad_(False)

    # --- LoRA config ---
    targets = guess_lora_targets(base)
    info(f"Selected LoRA target modules: {targets}")

    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        task_type=TaskType.SEQ_CLS,
        target_modules=targets
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()

    run_name = f"{model_name.split('/')[-1]}-lora-{MAX_LEN}-{imbalance_strategy}"
    out_dir = f"outputs/{run_name}"

    # Resolve mixed precision: CLI/env override, else auto on CUDA
    def _resolve_fp16(user_fp16: bool | None) -> bool:
        if user_fp16 is not None:
            return bool(user_fp16)
        env_val = os.getenv("TRAIN_FP16")
        if env_val is not None:
            return env_val.strip().lower() in {"1", "true", "yes", "y"}
        return torch.cuda.is_available()

    use_fp16 = _resolve_fp16(fp16)

    args = TrainingArguments(
        output_dir=out_dir,
        learning_rate=2e-4,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=32,
        num_train_epochs=num_train_epochs,
        gradient_accumulation_steps=2,
        evaluation_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=100,
        weight_decay=0.01,
        fp16=use_fp16,
        report_to=["none"],
        seed=SEED,
    )

    # --- MLflow ---
    mlflow.set_experiment("drug-review-sentiment")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model": model_name,
            "imbalance_strategy": imbalance_strategy,
            "lora_r": lora_cfg.r,
            "lora_alpha": lora_cfg.lora_alpha,
            "lora_dropout": lora_cfg.lora_dropout,
            "targets": ",".join(lora_cfg.target_modules),
            "lr": args.learning_rate,
            "max_len": MAX_LEN,
            "batch_train": args.per_device_train_batch_size,
            "epochs": args.num_train_epochs,
            "seed": SEED,
        })

        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]

        # Prepare imbalance handling
        trainer_cls = Trainer
        trainer_kwargs: dict[str, Any] = dict(
            model=model,
            args=args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )

        if imbalance_strategy == "weighted_loss":
            # Accessing column directly returns a Python list under HF Datasets
            y_train = np.asarray(tokenized["train"]["labels"])  # no .cpu() here
            class_weights = _compute_class_weights(y_train, num_labels=NUM_LABELS)
            info(f"Using weighted loss with class_weights: {class_weights.tolist()}")
            trainer_cls = WeightedLossTrainer
            trainer_kwargs["class_weights"] = class_weights
        else:
            info("No imbalance strategy applied.")

        trainer = trainer_cls(**trainer_kwargs)

        trainer.train()
        metrics = trainer.evaluate(tokenized["test"])
        for k, v in metrics.items():
            # Ensure MLflow receives plain Python floats
            try:
                mlflow.log_metric(f"test_{k}", float(v))
            except Exception:
                # Skip non-numeric values
                continue

        adapters_dir = f"{out_dir}-adapters"
        os.makedirs(adapters_dir, exist_ok=True)
        model.save_pretrained(adapters_dir)
        tokenizer.save_pretrained(adapters_dir)

        try:
            import matplotlib.pyplot as plt
            from sklearn.metrics import ConfusionMatrixDisplay
            y_logits = trainer.predict(tokenized["test"]).predictions
            y_pred = y_logits.argmax(axis=-1)
            y_true = tokenized["test"]["labels"].cpu().numpy()
            disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
            plt.title(f"Confusion Matrix ({model_name})")
            fig_path = os.path.join(out_dir, "cm.png")
            plt.savefig(fig_path, bbox_inches="tight")
            mlflow.log_artifact(fig_path)
        except Exception as e:
            info(f"CM plot failed: {e}")

        info(f"Done. Adapters at: {adapters_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sentiment model with optional LoRA and fp16.")
    parser.add_argument("--model", dest="models", action="append", help="HF model name (can be used multiple times)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--imbalance", choices=["none", "weighted_loss"], default="weighted_loss")
    # fp16 flags: --fp16 to enable, --no-fp16 to disable, default comes from env or CUDA availability
    fp16_group = parser.add_mutually_exclusive_group()
    fp16_group.add_argument("--fp16", dest="fp16", action="store_true")
    fp16_group.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.set_defaults(fp16=None)

    args_cli = parser.parse_args()

    models = args_cli.models or [
        "dmis-lab/biobert-base-cased-v1.1",
        "distilbert-base-uncased",
    ]

    try:
        for model in models:
            train_model(
                model_name=model,
                batch_size=args_cli.batch_size,
                num_train_epochs=args_cli.epochs,
                imbalance_strategy=args_cli.imbalance,
                fp16=args_cli.fp16,
            )
    except Exception as e:
        error(f"Training failed: {e}")
        sys.exit(1)
