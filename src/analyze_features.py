import pandas as pd


DATA_PATH = "data/processed/assignment_pairs_features.csv"


def analyze_features():
    df = pd.read_csv(DATA_PATH)

    feature_columns = [
        "jaccard_unigram",
        "shared_word_count",
        "shared_phrase_count",
        "longest_copied_stretch"
    ]

    print("Dataset shape:")
    print(df.shape)

    print("\nLabel distribution:")
    print(df["label"].value_counts().sort_index())

    print("\nAverage feature values by label:")
    print(
        df.groupby("label")[feature_columns]
        .mean()
        .round(4)
    )

    print("\nMinimum feature values by label:")
    print(
        df.groupby("label")[feature_columns]
        .min()
        .round(4)
    )

    print("\nMaximum feature values by label:")
    print(
        df.groupby("label")[feature_columns]
        .max()
        .round(4)
    )


if __name__ == "__main__":
    analyze_features()