"""
Utility to load local drug review datasets and return a DatasetDict
with train/validation/test splits and binary labels.
"""
import os
import logging
import pandas as pd
from datasets import Dataset, DatasetDict
from src.cleaning import clean_series, filter_reviews

# Bind to project logger if configured; otherwise fall back to standard logger
try:
    import src.logging_config as _logcfg  # type: ignore
    _LOGGER = logging.getLogger("load_datasets")
    _INFO = getattr(_logcfg, "info", _LOGGER.info)
except Exception:
    _LOGGER = logging.getLogger("load_datasets")
    _INFO = _LOGGER.info


def main(
    train_dir: str = "data",
    test_dir: str = "data",
    train_file: str = "drugsComTrain_raw.tsv",
    test_file: str = "drugsComTest_raw.tsv",
    sep: str = "\t",
    sample_frac: float | None = None,
    random_state: int = 42,
) -> DatasetDict:
    """
    Load CSV/TSV files, clean and filter reviews, optionally subsample for a quick trial,
    and return a Hugging Face DatasetDict with train/validation/test splits.
    """
    # Read data from the provided directories relative to the working dir
    train_path = os.path.join(train_dir, train_file)
    test_path = os.path.join(test_dir, test_file)
    train = pd.read_csv(train_path, sep=sep)
    test = pd.read_csv(test_path, sep=sep)
    try:
        _INFO(f"Loaded train={train.shape} from {train_path}; test={test.shape} from {test_path}")
    except Exception:
        pass

    def create_label(rating):
        """Convert numeric rating (1-10) into binary label: 0=negative, 1=positive."""
        return 1 if rating > 5 else 0

    train["sentiment"] = train["rating"].apply(create_label)
    test["sentiment"] = test["rating"].apply(create_label)
    try:
        tr_counts = train["sentiment"].value_counts().to_dict()
        te_counts = test["sentiment"].value_counts().to_dict()
        _INFO(f"Label counts — train: {tr_counts}; test: {te_counts}")
    except Exception:
        pass

    # Clean review text and filter short/empty/duplicate rows
    train["review"] = clean_series(train["review"])
    test["review"] = clean_series(test["review"])

    train = filter_reviews(train, text_col="review", min_len=5)
    test = filter_reviews(test, text_col="review", min_len=5)

    # Optional: downsample for quick trials (stratified by label)
    if sample_frac is not None:
        try:
            frac = float(sample_frac)
            if 0 < frac < 1:
                tb, te = len(train), len(test)
                train = train.groupby("sentiment", group_keys=False).sample(
                    frac=frac, random_state=random_state
                )
                test = test.groupby("sentiment", group_keys=False).sample(
                    frac=frac, random_state=random_state
                )
                _INFO(
                    f"Subset sampling applied (frac={frac}) — "
                    f"train: {tb}->{len(train)}, test: {te}->{len(test)}"
                )
            else:
                _INFO(f"sample_frac {sample_frac} ignored (must be in (0,1)).")
        except Exception as e:
            _INFO(f"Subset sampling skipped due to error: {e}")

    # Keep only required columns
    train = train[["review", "sentiment"]]
    test = test[["review", "sentiment"]]

    # Convert to Hugging Face datasets
    train_ds = Dataset.from_pandas(train)
    test_ds = Dataset.from_pandas(test)
    try:
        _INFO(f"HF datasets created — train={len(train_ds)}, test={len(test_ds)}")
    except Exception:
        pass

    # Rename label column to the conventional 'label'
    if "sentiment" in train_ds.column_names:
        train_ds = train_ds.rename_column("sentiment", "label")
    if "sentiment" in test_ds.column_names:
        test_ds = test_ds.rename_column("sentiment", "label")

    # Create validation split from train (prefer stratified by label; fallback if unsupported)
    try:
        split = train_ds.train_test_split(test_size=0.1, seed=42, stratify_by_column="label")
        stratified = True
    except TypeError:
        split = train_ds.train_test_split(test_size=0.1, seed=42)
        stratified = False
    except Exception:
        split = train_ds.train_test_split(test_size=0.1, seed=42)
        stratified = False
    try:
        _INFO(
            f"Validation split created (stratified={stratified}) — "
            f"train={len(split['train'])}, val={len(split['test'])}"
        )
    except Exception:
        pass

    # Assemble DatasetDict with validation
    dataset = DatasetDict({
        "train": split["train"],
        "validation": split["test"],
        "test": test_ds,
    })

    try:
        _INFO(
            f"DatasetDict ready — train={len(dataset['train'])}, "
            f"val={len(dataset['validation'])}, test={len(dataset['test'])}"
        )
    except Exception:
        pass

    return dataset
