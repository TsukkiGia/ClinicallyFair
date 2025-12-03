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


def plot_model_metrics_by_age(model_metric_data, metrics, age_groups, output_path, focus_groups=None, title="Model Metrics by Age Group"):
    """Render a tabular view of metrics by age group for each model."""

    if not model_metric_data:
        print("No model metric data provided for table plot.")
        return

    focus_groups = focus_groups or {}
    model_names = list(model_metric_data.keys())

    # Build data matrix and mask
    data_matrix = []
    mask_matrix = []
    for model in model_names:
        metric_entry = model_metric_data.get(model, {})
        allowed_groups = focus_groups.get(model)
        row_values = []
        mask_row = []
        for metric in metrics:
            metric_values = metric_entry.get(metric, {})
            for age_group in age_groups:
                value = metric_values.get(age_group)
                if allowed_groups is not None and age_group not in allowed_groups:
                    row_values.append(np.nan)
                    mask_row.append(True)
                else:
                    row_values.append(value)
                    is_missing = value is None or (isinstance(value, float) and np.isnan(value))
                    mask_row.append(is_missing)
        data_matrix.append(row_values)
        mask_matrix.append(mask_row)

    col_labels = ["Model"] + [f"{metric}\n{age}" for metric in metrics for age in age_groups]
    cell_text = []
    for model, row in zip(model_names, data_matrix):
        formatted = [model]
        for value in row:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                formatted.append("-")
            else:
                formatted.append(f"{value:.3f}")
        cell_text.append(formatted)

    fig_height = max(4, len(model_names) * 0.6)
    fig, ax = plt.subplots(figsize=(max(8, len(col_labels) * 1.2), fig_height))
    ax.axis('off')

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    for i in range(len(model_names)):
        for j in range(1, len(col_labels)):
            cell = table[(i + 1, j)]
            if mask_matrix[i][j - 1]:
                cell.set_facecolor('lightgray')
            else:
                cell.set_facecolor('white')

    plt.title(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Model metrics table saved as '{output_path}'")
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
