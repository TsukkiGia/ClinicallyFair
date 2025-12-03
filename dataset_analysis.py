import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def age_histogram(df: pd.DataFrame):
    plt.figure(figsize=(8,5))
    plt.hist(df, bins=20)
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.title("Age Distribution (Histogram)")
    plt.tight_layout()
    plt.savefig('age_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()

def get_logreg_feature_importance(columns, model, output_path):
    coefficients = model.coef_[0] 
    df = pd.DataFrame(
        {
            "feature": columns,
            "importance": coefficients
        }
    ).sort_values("importance")
    plt.figure(figsize=(10, 14))
    plt.barh(df["feature"], df["importance"], color=np.where(df["importance"] > 0, "crimson", "steelblue"))
    plt.xlabel("Coefficient Value")
    plt.title(f"Signed Feature Importance")
    plt.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return df

def plot_feature_importance_heatmap(model_importances, output_path):
    """Plot heatmap of signed logistic regression coefficients across models."""

    importance_series = {}
    for model_name, importance in model_importances.items():
        if isinstance(importance, pd.DataFrame):
            series = importance.set_index("feature")["importance"]
        elif isinstance(importance, pd.Series):
            series = importance
        else:
            series = pd.Series(importance)
        importance_series[model_name] = series

    combined_df = pd.DataFrame(importance_series).fillna(0)
    order = combined_df.abs().max(axis=1).sort_values(ascending=False).index
    combined_df = combined_df.loc[order]

    plt.figure(figsize=(12, max(6, len(combined_df) * 0.35)))
    sns.heatmap(
        combined_df,
        cmap="RdBu_r",
        center=0,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Coefficient"}
    )
    plt.title("Feature Importance Heatmap (Signed Logistic Coefficients)")
    plt.xlabel("Model")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def cancer_prevalence_fixed_bins(df):
    bins = [0, 30, 40, 50, 60, float("inf")]
    labels = ["<30", "30-39", "40-49", "50-59", "60+"]

    df = df.copy()
    df["age_group"] = pd.cut(df["Age"], bins=bins, labels=labels, right=False)

    out = (
        df.groupby("age_group")["TYPE"]
        .agg(
            prevalence=lambda x: (x == 1).mean(),
            count="count"
        )
        .reset_index()
    )

    plt.figure(figsize=(7,4))
    plt.bar(out["age_group"], out["prevalence"], color="crimson")
    plt.title("Cancer Prevalence by Age Group")
    plt.xlabel("Age Group")
    plt.ylabel("Cancer Prevalence (Proportion)")
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig('cancer_prevalence_by_age_group.png', dpi=300, bbox_inches='tight')
    plt.close()

    return out
