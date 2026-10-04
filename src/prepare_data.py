import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


DATA_PATH = Path("data/processed/assignment_pairs_features.csv")
OUTPUT_DIR = Path("data/processed")

FEATURE_COLUMNS = [
    "jaccard_unigram",
    "shared_word_count",
    "shared_phrase_count",
    "longest_copied_stretch"
]


def prepare_data():
    df = pd.read_csv(DATA_PATH)

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df["label"]
    )

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["label"]
    )

    columns = [
        "pair_id",
        "label"
    ] + FEATURE_COLUMNS

    train_df = train_df[columns]
    validation_df = validation_df[columns]
    test_df = test_df[columns]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(
        OUTPUT_DIR / "train_features.csv",
        index=False
    )

    validation_df.to_csv(
        OUTPUT_DIR / "validation_features.csv",
        index=False
    )

    test_df.to_csv(
        OUTPUT_DIR / "test_features.csv",
        index=False
    )

    print("Data preparation completed.")

    print("\nDataset sizes:")
    print(f"Total:      {len(df)}")
    print(f"Training:   {len(train_df)}")
    print(f"Validation: {len(validation_df)}")
    print(f"Testing:    {len(test_df)}")

    print("\nTraining label distribution:")
    print(train_df["label"].value_counts().sort_index())

    print("\nValidation label distribution:")
    print(validation_df["label"].value_counts().sort_index())

    print("\nTesting label distribution:")
    print(test_df["label"].value_counts().sort_index())

    print("\nUnique pair IDs:")
    print(
        len(set(train_df["pair_id"]) |
            set(validation_df["pair_id"]) |
            set(test_df["pair_id"]))
    )


if __name__ == "__main__":
    prepare_data()