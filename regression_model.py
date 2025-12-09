
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import math
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

from extract_features import get_cleaned_train_test, get_split_age_datasets
from dataset_analysis import (
    get_logreg_feature_importance,
    plot_feature_importance_heatmap,
    plot_model_metrics_by_age,
)
from visualise_data import (
    plot_personalized_accuracies_combined_negatives,
    plot_accuracies_negatives,
    plot_accuracy_comparison,
    plot_accuracy_comparison_general_personalised,
    plot_decoupled_vs_general_negatives,
)


param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
logistic_model = LogisticRegression(max_iter = 250, C = 1/10, warm_start = True)
AGE_LABELS = ['<30', '30-39', '40-49', '50-59', '60+']
AGE_BINS = [0, 30, 40, 50, 60, float('inf')]
METRIC_ORDER = ["Test Accuracy", "Test AUC", "FP", "TP", "TN"]
FIGURE_DIR = "figures"


def figure_path(filename: str) -> str:
    """Return the full path for a figure stored in the shared directory."""
    os.makedirs(FIGURE_DIR, exist_ok=True)
    return os.path.join(FIGURE_DIR, filename)


def get_best_model(train_features, train_labels):
    """Run grid search to select the best-regularized logistic regression model."""
    logistic_model = LogisticRegression(max_iter=2500, C=1/10, warm_start=True)
    
    grid_search = GridSearchCV(
        estimator=logistic_model,
        param_grid=param_grid,
        scoring='accuracy',  # roc_auc, f1
        cv=5,
        verbose=0
    )

    # Fit the Grid Search to training data
    grid_search.fit(train_features, train_labels)
    print(f"The best C value is: {grid_search.best_params_['C']}")
    best_model = grid_search.best_estimator_
    return best_model


def _sanitize_label(label):
    """Convert an age-group label into a filesystem-safe string."""
    return (label.replace("<", "lt")
                 .replace(">", "gt")
                 .replace("+", "plus")
                 .replace(" ", "_")
                 .replace("-", "_")
                 .lower())


def calculate_personalized_accuracies(age_datasets, print_results=True, return_models=False, importance_prefix=None):
    """Train per-age-group models and return their test accuracies (and models if requested)."""
    personalized_accuracies = {}
    evaluation_data = {}

    age_order = ['<30', '30-39', '40-49', '50-59', '60+']
    group_keys = [label for label in age_order if label in age_datasets]
    group_keys.extend([label for label in age_datasets if label not in group_keys])

    if print_results:
        print("\n" + "="*60)
        print("TRAINING PERSONALIZED MODELS FOR EACH AGE GROUP")
        print("="*60)

    for group_key in group_keys:
        if print_results:
            print(f"\n{group_key}:")
            print("-" * 40)

        train_features_group, train_labels_group = age_datasets[group_key]["train"]
        test_features_group, test_labels_group = age_datasets[group_key]["test"]

        train_count = len(train_features_group)
        test_count = len(test_features_group)

        if print_results:
            print(f"  Train samples: {train_count}")
            print(f"  Test samples: {test_count}")

        if train_count == 0 or test_count == 0:
            if print_results:
                print(f"  Skipping {group_key} - insufficient data")
            continue

        personalized_model = get_best_model(train_features_group, train_labels_group)
        personalized_acc = personalized_model.score(test_features_group, test_labels_group)
        personalized_accuracies[group_key] = personalized_acc

        feature_names = train_features_group.columns.tolist()
        if importance_prefix:
            output_path = figure_path(f"{importance_prefix}_{_sanitize_label(group_key)}.png")
            get_logreg_feature_importance(feature_names, personalized_model, output_path)

        if return_models:
            evaluation_data[group_key] = {
                "model": personalized_model,
                "test_features": test_features_group,
                "test_labels": test_labels_group,
                "feature_names": feature_names,
                "coefficients": personalized_model.coef_[0],
                "train_sample_size": train_count
            }

        if print_results:
            print(f"  Personalized Model Accuracy: {personalized_acc:.4f}")

    if return_models:
        return personalized_accuracies, evaluation_data
    return personalized_accuracies


def calculate_accuracy_across_ages(model, test_features, test_labels, test_age, print_results=True):
    """Compute test accuracy for each predefined age bin for a general model."""
    # Create age groups
    test_age_groups = pd.cut(test_age, bins=AGE_BINS, labels=AGE_LABELS, right=False)
    
    # Get predictions for all test data
    predictions = model.predict(test_features)
    
    if print_results:
        # Calculate accuracy for each age bracket
        print("\n" + "="*50)
        print("Accuracy by Age Group:")
        print("="*50)
    
    accuracies = {}
    for age_label in AGE_LABELS:
        # Get indices for this age group
        age_mask = test_age_groups == age_label
        
        if age_mask.sum() == 0:
            if print_results:
                print(f"{age_label:8s}: No samples in this group")
            continue
        
        # Calculate accuracy for this age group
        group_labels = test_labels[age_mask]
        group_predictions = predictions[age_mask]
        group_accuracy = accuracy_score(group_labels, group_predictions)
        
        accuracies[age_label] = group_accuracy
        sample_count = age_mask.sum()
        
        if print_results:
            print(f"{age_label:8s}: {group_accuracy:.4f} (n={sample_count})")
    
    if print_results:
        print("="*50 + "\n")
    
    return accuracies


def _mcnemar_exact_p_value(n01, n10):
    """Compute a two-sided exact McNemar p-value using the binomial distribution."""
    discordant = n01 + n10
    if discordant == 0:
        return np.nan

    k = min(n01, n10)
    p_tail = 0.0
    for i in range(0, k + 1):
        p_tail += math.comb(discordant, i)
    p_tail *= 0.5 ** discordant
    p_value = 2.0 * p_tail
    return min(1.0, p_value)


def _mcnemar_from_prediction_vectors(y_true, y_pred_a, y_pred_b, alpha=0.05):
    """Compute McNemar contingency table and stats from two prediction vectors."""
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)

    correct_a = y_pred_a == y_true
    correct_b = y_pred_b == y_true

    n01 = int(np.sum(~correct_a & correct_b))  # A wrong, B correct
    n10 = int(np.sum(correct_a & ~correct_b))  # A correct, B wrong
    n00 = int(np.sum(~correct_a & ~correct_b))
    n11 = int(np.sum(correct_a & correct_b))

    discordant = n01 + n10
    if discordant == 0:
        p_value = np.nan
        statistic = np.nan
        significant = False
    else:
        p_value = _mcnemar_exact_p_value(n01, n10)
        statistic = ((abs(n01 - n10) - 1) ** 2) / discordant
        significant = bool(p_value < alpha)

    return {
        "n01": n01,
        "n10": n10,
        "n00": n00,
        "n11": n11,
        "discordant": discordant,
        "statistic": statistic,
        "p_value": p_value,
        "significant@0.05": significant,
    }


def calculate_mcnemar_across_ages(
    model_a,
    model_b,
    test_features_a,
    test_features_b,
    test_labels,
    test_age,
    label_a="Model A",
    label_b="Model B",
    print_results=True,
):
    """
    Perform McNemar tests comparing two models within each age bin.

    Returns a dict keyed by age label; values contain the contingency counts,
    McNemar test statistic (with continuity correction), and p-value.
    """
    test_age_groups = pd.cut(test_age, bins=AGE_BINS, labels=AGE_LABELS, right=False)

    if print_results:
        print("\n" + "=" * 50)
        print(f"McNemar Test by Age Group ({label_a} vs {label_b})")
        print("=" * 50)

    results = {}
    for age_label in AGE_LABELS:
        age_mask = test_age_groups == age_label
        n_samples = age_mask.sum()

        if n_samples == 0:
            if print_results:
                print(f"{age_label:8s}: No samples in this group")
            continue

        y_true = test_labels[age_mask]
        y_pred_a = model_a.predict(test_features_a[age_mask])
        y_pred_b = model_b.predict(test_features_b[age_mask])

        stats = _mcnemar_from_prediction_vectors(y_true, y_pred_a, y_pred_b)
        results[age_label] = stats

        if print_results:
            p_value = stats["p_value"]
            if np.isnan(p_value):
                print(
                    f"{age_label:8s}: Models identical on all {n_samples} samples "
                    f"(no discordant pairs), McNemar test undefined"
                )
            else:
                sig_label = "YES" if stats["significant@0.05"] else "no"
                print(
                    f"{age_label:8s}: n01={stats['n01']:3d}, n10={stats['n10']:3d}, "
                    f"discordant={stats['discordant']:3d}, "
                    f"chi2={stats['statistic']:6.3f}, p={p_value:.4g}, "
                    f"significant@0.05={sig_label}"
                )

    if print_results:
        print("=" * 50 + "\n")

    return results

def _compute_roc_metrics(model, test_features, test_labels):
    """Compute ROC curve points and AUC for a model if both classes are present."""
    unique_classes = np.unique(test_labels)
    if len(unique_classes) < 2:
        print("Skipping ROC curve - only one class present in labels.")
        return None

    probas = model.predict_proba(test_features)[:, 1]
    fpr, tpr, _ = roc_curve(test_labels, probas)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


def plot_roc_curves(model_eval_data, title, filename):
    """Plot ROC curves for one or more evaluated models and save to disk."""
    if not model_eval_data:
        print("No model evaluation data provided for ROC plot.")
        return

    plt.figure(figsize=(8, 6))
    plotted = False

    skipped_labels = []

    for entry in model_eval_data:
        label = entry.get("label", "Model")
        metrics = _compute_roc_metrics(
            entry["model"], entry["test_features"], entry["test_labels"]
        )

        if metrics is None:
            skipped_labels.append(label)
            continue

        fpr, tpr, roc_auc = metrics
        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{label} (AUC = {roc_auc:.3f})"
        )
        plotted = True

    if not plotted:
        print("Unable to plot ROC curve(s) - insufficient data.")
        plt.close()
        return

    if skipped_labels:
        print("ROC curves skipped for:", ", ".join(skipped_labels))

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Chance Level')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"ROC plot saved as '{filename}'")
    plt.close()


def compute_metrics_by_age(model, test_features, test_labels, test_age):
    """Compute accuracy/AUC/confusion counts per age bin for a given model."""
    if len(test_labels) == 0:
        return {}

    test_age_groups = pd.cut(test_age, bins=AGE_BINS, labels=AGE_LABELS, right=False)
    predictions = model.predict(test_features)
    probabilities = model.predict_proba(test_features)[:, 1]

    metrics_by_age = {metric: {} for metric in METRIC_ORDER}

    for age_label in AGE_LABELS:
        mask = test_age_groups == age_label
        if mask.sum() == 0:
            continue
        stats = _metrics_from_predictions(
            test_labels[mask], predictions[mask], probabilities[mask]
        )
        for metric_name in METRIC_ORDER:
            metrics_by_age[metric_name][age_label] = stats.get(metric_name)

    return metrics_by_age


def _metrics_from_predictions(labels, predictions, probabilities=None):
    """Return accuracy, AUC (if available), and confusion matrix counts."""
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)

    accuracy = accuracy_score(labels, predictions)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

    auc_value = np.nan
    if probabilities is not None and len(np.unique(labels)) == 2:
        try:
            auc_value = roc_auc_score(labels, probabilities)
        except ValueError:
            pass

    return {
        "Test Accuracy": accuracy,
        "Test AUC": auc_value,
        "FN": fn,
        "FP": fp,
        "TP": tp,
        "TN": tn
    }


def compute_model_metrics(model, test_features, test_labels):
    """Compute overall test accuracy, AUC, and confusion counts for a model."""
    if len(test_labels) == 0:
        return None
    predictions = model.predict(test_features)
    probabilities = model.predict_proba(test_features)[:, 1]
    return _metrics_from_predictions(test_labels, predictions, probabilities)


def compute_decoupled_metrics(evaluation_data):
    """Aggregate metrics across multiple personalized models evaluated on their splits."""
    all_predictions = []
    all_labels = []
    all_probabilities = []

    for data in evaluation_data.values():
        test_features = data.get("test_features")
        test_labels = data.get("test_labels")
        model = data.get("model")
        if test_features is None or test_labels is None or len(test_labels) == 0:
            continue
        preds = model.predict(test_features)
        probs = model.predict_proba(test_features)[:, 1]
        all_predictions.append(preds)
        all_labels.append(test_labels)
        all_probabilities.append(probs)

    if not all_labels:
        print("No evaluation data available for decoupled metrics.")
        return None

    predictions = np.concatenate(all_predictions)
    labels = np.concatenate(all_labels)
    probabilities = np.concatenate(all_probabilities)
    return _metrics_from_predictions(labels, predictions, probabilities)


def get_sample_sizes_by_age_group(test_age):
    """Extract sample sizes for each of the fixed age groups."""
    age_labels = ['<30', '30-39', '40-49', '50-59', '60+']
    age_bins = [0, 30, 40, 50, 60, float('inf')]
    
    # Create age groups
    test_age_groups = pd.cut(test_age, bins=age_bins, labels=age_labels, right=False)
    
    # Count samples in each group
    sample_sizes = {}
    for age_label in age_labels:
        age_mask = test_age_groups == age_label
        sample_sizes[age_label] = age_mask.sum()
    
    return sample_sizes


def calculate_mcnemar_personalized_vs_generic_by_age(
    generic_model,
    generic_test_features,
    generic_test_labels,
    personalized_eval_data,
    label_generic="Generic - Without Age",
    label_personalized_prefix="Personalized - Without Age",
    print_results=True,
):
    """
    Perform McNemar tests comparing the generic model (without age)
    against each personalized (decoupled) model on that age group's test patients.
    """
    results = {}

    if print_results:
        print("\n" + "=" * 50)
        print(f"McNemar Test by Age Group ({label_generic} vs {label_personalized_prefix})")
        print("=" * 50)

    for age_label in AGE_LABELS:
        data = personalized_eval_data.get(age_label)
        if not data:
            continue

        X_personal = data["test_features"]
        y_group = data["test_labels"]
        if len(y_group) == 0:
            if print_results:
                print(f"{age_label:8s}: No samples in this group")
            continue

        idx = X_personal.index
        # Align generic features/labels to the same patients
        X_generic = generic_test_features.loc[idx]
        y_true = generic_test_labels.loc[idx]

        # Sanity check: labels should match between views
        if not np.array_equal(np.asarray(y_true), np.asarray(y_group)):
            print(f"Warning: label mismatch for age group {age_label}, skipping McNemar comparison.")
            continue

        y_pred_generic = generic_model.predict(X_generic)
        y_pred_personal = data["model"].predict(X_personal)

        stats = _mcnemar_from_prediction_vectors(y_true, y_pred_generic, y_pred_personal)
        results[age_label] = stats

        if print_results:
            p_value = stats["p_value"]
            if np.isnan(p_value):
                print(
                    f"{age_label:8s}: Models identical on all {len(y_true)} samples "
                    f"(no discordant pairs), McNemar test undefined"
                )
            else:
                sig_label = "YES" if stats["significant@0.05"] else "no"
                print(
                    f"{age_label:8s}: n01={stats['n01']:3d}, n10={stats['n10']:3d}, "
                    f"discordant={stats['discordant']:3d}, "
                    f"chi2={stats['statistic']:6.3f}, p={p_value:.4g}, "
                    f"significant@0.05={sig_label}"
                )

    if print_results:
        print("=" * 50 + "\n")

    return results


if __name__ == "__main__":
    # ==================== EXPERIMENT 1: WITHOUT AGE ====================

    metrics_summary = {}
    model_metric_data = {}

    # Build a single canonical train/test split (with age feature),
    # then derive the no-age version from the same split so that
    # McNemar comparisons use exactly the same patients.
    train_features_with_age, train_labels_with_age, train_age_with, test_features_with_age, test_labels_with_age, test_age_with = get_cleaned_train_test(True)

    drop_cols = [c for c in ["Age", "Menopause"] if c in train_features_with_age.columns]
    train_features_no_age = train_features_with_age.drop(columns=drop_cols)
    test_features_no_age = test_features_with_age.drop(columns=drop_cols)

    train_labels_no_age = train_labels_with_age
    test_labels_no_age = test_labels_with_age
    train_age_no = train_age_with
    test_age_no = test_age_with

    # ==================== EXPERIMENT 1: WITHOUT AGE ====================

    # Train model
    model_without_age = get_best_model(train_features_no_age, train_labels_no_age)

    # Get feature importance
    importance_df_no_age = get_logreg_feature_importance(
        train_features_no_age.columns,
        model_without_age,
        figure_path("feature_importance_general_without_age.png"),
    )
    
    # Report model metrics
    gm_no_age_metrics = compute_model_metrics(model_without_age, test_features_no_age, test_labels_no_age)
    metrics_summary["Generic Model - Without Age"] = gm_no_age_metrics

    # Get accuracies across age groups
    accuracies_without_age = calculate_accuracy_across_ages(
        model_without_age, test_features_no_age, test_labels_no_age, test_age_no, print_results=True
    )
    model_metric_data["Generic - Without Age"] = compute_metrics_by_age(
        model_without_age, test_features_no_age, test_labels_no_age, test_age_no
    )
    # ==================== EXPERIMENT 2: WITH AGE ====================

    # Train model
    model_with_age = get_best_model(train_features_with_age, train_labels_with_age)
    importance_df_with_age = get_logreg_feature_importance(
        train_features_with_age.columns,
        model_with_age,
        figure_path("feature_importance_general_with_age.png"),
    )
    
    # Report model metrics
    gm_with_age_metrics = compute_model_metrics(model_with_age, test_features_with_age, test_labels_with_age)
    metrics_summary["Generic Model - With Age"] = gm_with_age_metrics

    # Get accuracies across age groups
    accuracies_with_age = calculate_accuracy_across_ages(
        model_with_age, test_features_with_age, test_labels_with_age, test_age_with, print_results=True
    )
    model_metric_data["Generic - With Age"] = compute_metrics_by_age(
        model_with_age, test_features_with_age, test_labels_with_age, test_age_with
    )

    # ==================== MCNEMAR COMPARISON (GENERAL MODELS) ====================

    mcnemar_results_by_age = calculate_mcnemar_across_ages(
        model_without_age,
        model_with_age,
        test_features_no_age,
        test_features_with_age,
        test_labels_no_age,
        test_age_no,
        label_a="Generic - Without Age",
        label_b="Generic - With Age",
        print_results=True,
    )
    
    
    # ==================== COMPARISON ====================
    
    # Plot comparison
    plot_accuracy_comparison(accuracies_without_age, accuracies_with_age)
    plot_accuracies_negatives(accuracies_without_age, accuracies_with_age)

    # ROC curves for the two general models
    general_model_eval = [
        {
            "label": "General - Without Age",
            "model": model_without_age,
            "test_features": test_features_no_age,
            "test_labels": test_labels_no_age
        },
        {
            "label": "General - With Age",
            "model": model_with_age,
            "test_features": test_features_with_age,
            "test_labels": test_labels_with_age
        }
    ]
    plot_roc_curves(
        general_model_eval,
        "ROC Curves - General Models",
        figure_path("roc_general_models.png"),
    )
    
    
    # ==================== EXPERIMENT 3: PERSONALIZED MODELS BY AGE ====================
    
    # Get age-split datasets (without age as feature)
    age_datasets_no_age = get_split_age_datasets(False)
    
    # Train personalized models for each age group (without age)
    personalized_accuracies_no_age, personalized_eval_no_age = calculate_personalized_accuracies(
        age_datasets_no_age,
        print_results=True,
        return_models=True,
        importance_prefix="feature_importance_personalized_no_age"
    )
    
    # Get age-split datasets (WITH age as feature)
    age_datasets_with_age = get_split_age_datasets(True)
    
    # Train personalized models for each age group (with age)
    print("\n" + "="*60)
    print("TRAINING PERSONALIZED MODELS WITH AGE FEATURE")
    print("="*60)
    personalized_accuracies_with_age, personalized_eval_with_age = calculate_personalized_accuracies(
        age_datasets_with_age,
        print_results=True,
        return_models=True,
        importance_prefix="feature_importance_personalized_with_age"
    )

    # Plot combined personalized model accuracies (diverging plot)
    plot_personalized_accuracies_combined_negatives(personalized_accuracies_no_age, personalized_accuracies_with_age)

    # Plot comparison of general vs personalized models
    plot_accuracy_comparison_general_personalised(accuracies_without_age, accuracies_with_age, personalized_accuracies_no_age)

    # Diverging plot: generic-without-age vs decoupled models by age group
    plot_decoupled_vs_general_negatives(accuracies_without_age, personalized_accuracies_no_age)

    # Plot ROC curves for personalized models (without and with age)
    personalized_eval_entries_no_age = [
        {**data, "label": f"No Age - {label}"}
        for label, data in personalized_eval_no_age.items()
    ]
    plot_roc_curves(
        personalized_eval_entries_no_age,
        "ROC Curves - Personalized Models (No Age)",
        figure_path("roc_personalized_no_age.png"),
    )

    personalized_eval_entries_with_age = [
        {**data, "label": f"With Age - {label}"}
        for label, data in personalized_eval_with_age.items()
    ]
    plot_roc_curves(
        personalized_eval_entries_with_age,
        "ROC Curves - Personalized Models (With Age)",
        figure_path("roc_personalized_with_age.png"),
    )

    # ==================== MCNEMAR: GENERIC VS PERSONALIZED (BY AGE) ====================

    mcnemar_personalized_vs_generic = calculate_mcnemar_personalized_vs_generic_by_age(
        model_without_age,
        test_features_no_age,
        test_labels_no_age,
        personalized_eval_no_age,
        label_generic="Generic - Without Age",
        label_personalized_prefix="Personalized - Without Age",
        print_results=True,
    )

    focus_groups = {
        "Generic - Without Age": set(AGE_LABELS),
        "Generic - With Age": set(AGE_LABELS),
        "Personalized - Without Age": set(AGE_LABELS),
    }

    personalized_metrics_by_age = {metric: {} for metric in METRIC_ORDER}
    for label, data in personalized_eval_no_age.items():
        stats = compute_model_metrics(
            data["model"], data["test_features"], data["test_labels"]
        )
        if not stats:
            continue
        for metric in METRIC_ORDER:
            personalized_metrics_by_age[metric][label] = stats.get(metric)

    model_metric_data["Personalized - Without Age"] = personalized_metrics_by_age

    decoupled_metrics = compute_decoupled_metrics(personalized_eval_no_age)
    metrics_summary["Decoupled Models - Without Age"] = decoupled_metrics

    importance_heatmap_data = {
        "Generic - Without Age": importance_df_no_age.set_index("feature")["importance"],
        "Generic - With Age": importance_df_with_age.set_index("feature")["importance"],
    }

    for label, data in personalized_eval_no_age.items():
        feature_names = data.get("feature_names")
        coefficients = data.get("coefficients")
        importance_heatmap_data[f"Decoupled - {label}"] = pd.Series(coefficients, index=feature_names)

    plot_feature_importance_heatmap(
        importance_heatmap_data,
        figure_path("feature_importance_heatmap.png"),
    )

    metrics_df = pd.DataFrame(metrics_summary)
    desired_rows = ["Test Accuracy", "Test AUC", "FP", "TP", "TN"]
    metrics_df = metrics_df.reindex(desired_rows)
    print("\n" + "="*60)
    print("MODEL METRIC SUMMARY")
    print("="*60)
    print(metrics_df)
    metrics_df.to_csv('model_metrics_summary.csv')
    print("Metric summary saved as 'model_metrics_summary.csv'")

    age_groups = AGE_LABELS

    # Accuracy table
    plot_model_metrics_by_age(
        model_metric_data,
        ["Test Accuracy"],
        age_groups,
        figure_path("model_metrics_accuracy_by_age.png"),
        focus_groups=focus_groups,
        title="Model Accuracy by Age Group",
    )

    # AUC table
    plot_model_metrics_by_age(
        model_metric_data,
        ["Test AUC"],
        age_groups,
        figure_path("model_metrics_auc_by_age.png"),
        focus_groups=focus_groups,
        title="Model AUC by Age Group",
    )

    # Confusion-matrix count tables (FP, TP, TN)
    plot_model_metrics_by_age(
        model_metric_data,
        ["FP"],
        age_groups,
        figure_path("model_metrics_fp_by_age.png"),
        focus_groups=focus_groups,
        title="Model FP Counts by Age Group",
    )

    plot_model_metrics_by_age(
        model_metric_data,
        ["TP"],
        age_groups,
        figure_path("model_metrics_tp_by_age.png"),
        focus_groups=focus_groups,
        title="Model TP Counts by Age Group",
    )

    plot_model_metrics_by_age(
        model_metric_data,
        ["TN"],
        age_groups,
        figure_path("model_metrics_tn_by_age.png"),
        focus_groups=focus_groups,
        title="Model TN Counts by Age Group",
    )

    # Get sample sizes for the 6 age groups (using test_age from experiment 1/2)
    sample_sizes = get_sample_sizes_by_age_group(test_age_with)
    print("\n" + "-"*60)
    print("Sample sizes for 6 age groups:")
    print("-"*60)
    for age_label, size in sample_sizes.items():
        print(f"  {age_label}: {size}")
