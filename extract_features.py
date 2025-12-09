import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_and_clean_dataset(include_age: bool):
    """Load Excel data, clean features, and return train/test splits with ages."""
    train_df = pd.read_excel("train.xlsx")
    test_df = pd.read_excel("test.xlsx")

    # Rebuild train/test: combine, shuffle with fixed seed, and split 60/40.
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_df = full_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    split_index = int(len(full_df) * 0.6)
    train_df = full_df.iloc[:split_index].copy()
    test_df = full_df.iloc[split_index:].copy()

    # Clean string columns that have ">" in the recomputed train/test splits
    num_str_cols = ["AFP", "CA125", "CA19-9"]
    for col in num_str_cols:
        for df in (train_df, test_df):
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].str.replace(">", "", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop label, subject id and ca72 columns
    base_feature_cols = [c for c in train_df.columns if c not in ["TYPE", "SUBJECT_ID", "CA72-4"]]
    means = train_df[base_feature_cols].mean()

    train_imputed = train_df.copy()
    test_imputed = test_df.copy()

    train_imputed[base_feature_cols] = train_imputed[base_feature_cols].fillna(means)
    test_imputed[base_feature_cols] = test_imputed[base_feature_cols].fillna(means)

    if not include_age:
        base_feature_cols = [c for c in base_feature_cols if c not in ["Age", "Menopause"]]

    train_features = train_imputed[base_feature_cols]
    train_labels = 1 - train_imputed["TYPE"]

    test_features = test_imputed[base_feature_cols]
    test_labels = 1 - test_imputed["TYPE"]

    train_age = train_imputed["Age"]
    test_age = test_imputed["Age"]

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(train_features),
        columns=train_features.columns,
        index=train_features.index,
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(test_features),
        columns=test_features.columns,
        index=test_features.index,
    )

    return X_train_scaled, train_labels, train_age, X_test_scaled, test_labels, test_age


def make_age_stratified_splits(include_age: bool):
    """Build per-age-group train/test subsets from the cleaned dataset."""
    (
        train_features,
        train_labels,
        train_age,
        test_features,
        test_labels,
        test_age,
    ) = load_and_clean_dataset(include_age)

    dataset = {}
    bins = [0, 30, 40, 50, 60, float('inf')]
    labels = ['<30', '30-39', '40-49', '50-59', '60+']
    train_bins = pd.cut(train_age, bins, False, labels)
    test_bins = pd.cut(test_age, bins, False, labels)

    for label in labels:
        train_mask = train_bins == label
        test_mask = test_bins == label
        data = {
            'train': (train_features.loc[train_mask], train_labels.loc[train_mask]),
            'test': (test_features.loc[test_mask], test_labels.loc[test_mask])
        }
        dataset[label] = data

    return dataset


# Backwards-compatible aliases
get_cleaned_train_test = load_and_clean_dataset
get_split_age_datasets = make_age_stratified_splits


if __name__ == "__main__":
    load_and_clean_dataset(False)
