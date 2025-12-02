import pandas as pd
import matplotlib as plt
import numpy as np
import seaborn as sns

def age_histogram(df: pd.DataFrame):
    plt.figure(figsize=(8,5))
    plt.hist(df, bins=20)
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.title("Age Distribution (Histogram)")
    plt.show()

def get_logreg_feature_importance(df, model):
    coefficients = model.coef_[0] 
    df = pd.DataFrame(
        {
            "features": df.columns,
            "importance": coefficients
        }
    ).sort_values("importance")
    plt.figure(figsize=(10, 14))
    plt.barh(df["feature"], df["importance"], color=np.where(df["importance"] > 0, "crimson", "steelblue"))
    plt.xlabel("Coefficient Value")
    plt.title(f"Signed Feature Importance")
    plt.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

def plot_correlation_with_age(df, feature_col):
    data = df[[feature_col, "Age"]].dropna()

    corr = data[feature_col].corr(data["Age"], method="pearson")

    # Plot
    plt.figure(figsize=(7,5))
    sns.regplot(
        x="Age", 
        y=feature_col, 
        data=data, 
        scatter_kws={"alpha": 0.5}, 
        line_kws={"color": "red"}
    )
    
    plt.title(f"Correlation Between Age and {feature_col}\nPearson r = {corr}")
    plt.xlabel("Age")
    plt.ylabel(feature_col)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return corr