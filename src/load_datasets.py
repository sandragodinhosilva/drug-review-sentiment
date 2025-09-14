"""
Utility to load local drug review datasets and return a DatasetDict
with train/validation/test splits and binary labels.
"""
import os
import re
import pandas as pd
from datasets import Dataset, DatasetDict

def main(
    train_dir: str = "data",
    test_dir: str = "data",
    train_file: str = "drugsComTrain_raw.tsv",
    test_file: str = "drugsComTest_raw.tsv",
    sep: str = "\t"
) -> DatasetDict:
    """
    Takes in a training directory and testing directory path and turns
      them into PyTorch Datasets and then into PyTorch DataLoaders.
      
    Args:
        train_dir (str): Directory path for training data.
        test_dir (str): Directory path for testing data.	
        train_file (str): File name for training data.
        test_file (str): File name for testing data.
        sep (str): Separator used in the CSV/TSV files.

    Returns:
        dataset (DatasetDict): A DatasetDict containing 'train', 'validation', and 'test' datasets
    """
    # Read data from the provided directories relative to the working dir
    train = pd.read_csv(os.path.join(train_dir, train_file), sep=sep)
    test = pd.read_csv(os.path.join(test_dir, test_file), sep=sep)


    def create_label(rating):
        """
        Convert numeric rating (1 - 10) into binary label: 0 = negative, 1 = positive.
        """
        return 1 if rating > 5 else 0

    def clean_text(text: object) -> str:
        """Basic review cleaner: remove HTML tags and normalize spaces."""
        if isinstance(text, str) is False:
            # Convert NaN/other types to string safely
            text = "" if pd.isna(text) else str(text)
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    train["sentiment"] = train["rating"].apply(create_label)
    test["sentiment"]  = test["rating"].apply(create_label)

    # Clean review text and keep only required columns
    train["review"] = train["review"].apply(clean_text)
    test["review"] = test["review"].apply(clean_text)
    train = train[["review", "sentiment"]]
    test  = test[["review", "sentiment"]]

    # Convert to Hugging Face datasets
    train_ds = Dataset.from_pandas(train)
    test_ds  = Dataset.from_pandas(test)

    # Rename label column to the conventional 'label'
    if "sentiment" in train_ds.column_names:
        train_ds = train_ds.rename_column("sentiment", "label")
    if "sentiment" in test_ds.column_names:
        test_ds = test_ds.rename_column("sentiment", "label")

    # Create validation split from train
    split = train_ds.train_test_split(test_size=0.1, seed=42)

    # Assemble DatasetDict with validation
    dataset = DatasetDict({
        "train": split["train"],
        "validation": split["test"],
        "test": test_ds
    })

    return dataset
