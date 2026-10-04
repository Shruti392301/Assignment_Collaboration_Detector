import pandas as pd
from pathlib import Path

from text_features import get_lexical_features


RAW_PATH = Path("data/raw/assignment_pairs_raw.csv")
PROCESSED_DIR = Path("data/processed")
OUTPUT_PATH = PROCESSED_DIR / "assignment_pairs_features.csv"


def create_features():
    df = pd.read_csv(RAW_PATH)

    feature_rows = []

    for _, row in df.iterrows():
        features = get_lexical_features(
            row["text_a"],
            row["text_b"]
        )

        feature_rows.append(features)

    feature_df = pd.DataFrame(feature_rows)

    result = pd.concat(
        [
            df,
            feature_df
        ],
        axis=1
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    result.to_csv(OUTPUT_PATH, index=False)

    print("Feature extraction completed.")
    print(f"Original rows: {len(df)}")
    print(f"Processed rows: {len(result)}")
    print(f"Output file: {OUTPUT_PATH}")

    print("\nFeature columns:")
    print(feature_df.columns.tolist())

    print("\nFirst 5 rows:")
    print(
        result[
            [
                "pair_id",
                "label",
                "jaccard_unigram",
                "shared_word_count",
                "shared_phrase_count",
                "longest_copied_stretch"
            ]
        ].head()
    )


if __name__ == "__main__":
    create_features()