"""
Contains functionality for uploading data
"""
import os
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
        dataset (DatasetDict): A DatasetDict containing 'train' and 'test' datasets
    """
    train = pd.read_csv(os.path.join("..", train_dir, train_file ), sep=sep)
    test = pd.read_csv(os.path.join("..", test_dir, test_file), sep=sep)


    def create_label(rating):
        """
        Convert numeric rating (1 - 10) into binary label: 0 = negative, 1 = positive.
        """
        return 1 if rating > 5 else 0

    train["sentiment"] = train["rating"].apply(create_label)
    test["sentiment"]  = test["rating"].apply(create_label)

    # Keep only required columns
    train = train[["review", "sentiment"]]
    test  = test[["review", "sentiment"]]

    # Convert to Hugging Face datasets
    train_ds = Dataset.from_pandas(train)
    test_ds  = Dataset.from_pandas(test)

    # Create DatasetDict
    dataset = DatasetDict({
        "train": train_ds,
        "test": test_ds
    })

    return dataset
