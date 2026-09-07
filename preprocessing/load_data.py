from pathlib import Path

import pandas as pd


def load_csv(file_path: Path) -> pd.DataFrame:
    """
    Load a DAGHAR CSV file.

    Parameters
    ----------
    file_path : Path
        Path to train.csv, validation.csv or test.csv.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"\nDataset file was not found:\n{file_path}\n\n"
            "Make sure the DAGHAR UCI dataset is inside:\n"
            "data/processed/daghar/standardized_view/UCI/"
        )

    dataframe = pd.read_csv(file_path)

    print(f"Loaded: {file_path.name}")
    print(f"Shape: {dataframe.shape}")

    return dataframe


def load_daghar_splits(
    train_file: Path,
    val_file: Path,
    test_file: Path,
):
    """
    Load the three DAGHAR dataset partitions.
    """

    train_df = load_csv(train_file)
    val_df = load_csv(val_file)
    test_df = load_csv(test_file)

    return train_df, val_df, test_df