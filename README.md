## Project Folder Structure

```text
Assignment-Collaboration-Detector/
│
├── data/
│   ├── raw/
│   │   └── assignment_pairs_raw.csv
│   │
│   ├── processed/
│   │   └── assignment_pairs_cleaned.csv
│   │
│   └── sample/
│       └── sample_pairs.csv
│
├── notebooks/
│   ├── 01_dataset_creation.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_lexical_features.ipynb
│   ├── 04_semantic_features.ipynb
│   └── 05_model_training.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── lexical_features.py
│   ├── semantic_features.py
│   ├── style_features.py
│   ├── feature_builder.py
│   └── model.py
│
├── app.py
├── requirements.txt
└── README.md
```
