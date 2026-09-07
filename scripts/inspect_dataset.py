import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


from config import (
    TRAIN_FILE,
    VAL_FILE,
    TEST_FILE,
    LABEL_COLUMN,
)

from preprocessing.load_data import load_daghar_splits
from preprocessing.preprocess import dataframe_to_numpy


def inspect_split(name, dataframe):

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print("\nShape:")
    print(dataframe.shape)

    print("\nFirst 10 columns:")
    print(
        dataframe.columns[:10].tolist()
    )

    print("\nLast 10 columns:")
    print(
        dataframe.columns[-10:].tolist()
    )

    if LABEL_COLUMN in dataframe.columns:

        print("\nActivity codes:")
        print(
            sorted(
                dataframe[LABEL_COLUMN]
                .unique()
                .tolist()
            )
        )

        print("\nClass distribution:")
        print(
            dataframe[
                LABEL_COLUMN
            ].value_counts().sort_index()
        )

    X, y = dataframe_to_numpy(
        dataframe
    )

    print("\nConverted X shape:")
    print(X.shape)

    print("\nConverted y shape:")
    print(y.shape)


def main():

    print("\nDAGHAR UCI HAR DATASET INSPECTION")

    train_df, val_df, test_df = load_daghar_splits(
        TRAIN_FILE,
        VAL_FILE,
        TEST_FILE,
    )

    inspect_split(
        "TRAIN SET",
        train_df,
    )

    inspect_split(
        "VALIDATION SET",
        val_df,
    )

    inspect_split(
        "TEST SET",
        test_df,
    )


if __name__ == "__main__":
    main()