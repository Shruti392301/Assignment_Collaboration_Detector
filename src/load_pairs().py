import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


DATA_PATH = Path("data/raw/assignment_pairs_raw.csv")


def load_pairs(file_path=DATA_PATH):
    df = pd.read_csv(file_path)

    required_columns = [
        "pair_id",
        "topic",
        "text_a",
        "text_b",
        "label",
        "similarity_type"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    df = df.dropna(subset=["text_a", "text_b", "label"])

    df["text_a"] = df["text_a"].astype(str)
    df["text_b"] = df["text_b"].astype(str)
    df["label"] = df["label"].astype(int)

    return df


def split_pairs(df, random_state=42):
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=random_state,
        stratify=df["label"]
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=random_state,
        stratify=temp_df["label"]
    )

    return train_df, val_df, test_df


if __name__ == "__main__":
    df = load_pairs()

    train_df, val_df, test_df = split_pairs(df)

    print("Total pairs:", len(df))
    print("Training pairs:", len(train_df))
    print("Validation pairs:", len(val_df))
    print("Testing pairs:", len(test_df))

    print("\nTraining labels:")
    print(train_df["label"].value_counts())

    print("\nValidation labels:")
    print(val_df["label"].value_counts())

    print("\nTesting labels:")
    print(test_df["label"].value_counts())