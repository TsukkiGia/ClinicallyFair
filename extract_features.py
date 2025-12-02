import pandas as pd

def get_cleaned_train_test(include_age: bool):
    train_df = pd.read_excel('train.xlsx')
    test_df = pd.read_excel('test.xlsx')

    # Clean string columns that have ">"
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

    return train_features, train_labels, train_age, test_features, test_labels, test_age

def get_split_age_datasets(young_age: int = 40, old_age: int = 60):
    train_features, train_labels, train_age, test_features, test_labels, test_age = get_cleaned_train_test(False)
    young_mask = train_age < young_age
    mid_mask = (train_age >= young_age) & (train_age <= old_age)
    old_mask = train_age > old_age

    young_mask_test = test_age < young_age
    mid_mask_test = (test_age >= young_age) & (test_age <= old_age)
    old_mask_test = test_age > old_age

    return {
        "young": {
            "train": (train_features.loc[young_mask], train_labels.loc[young_mask]),
            "test": (test_features.loc[young_mask_test], test_labels.loc[young_mask_test])
        },
        "mid": {
            "train": (train_features.loc[mid_mask], train_labels.loc[mid_mask]),
            "test": (test_features.loc[mid_mask_test], test_labels.loc[mid_mask_test])
        },
        "old": {
            "train": (train_features.loc[old_mask], train_labels.loc[old_mask]),
            "test": (test_features.loc[old_mask_test], test_labels.loc[old_mask_test])
        },
    }



if __name__ == "__main__":
    get_cleaned_train_test(False)
