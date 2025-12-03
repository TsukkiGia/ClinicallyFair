
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score, confusion_matrix
np.random.seed(42)

from extract_features import get_cleaned_train_test, get_split_age_datasets
from dataset_analysis import get_logreg_feature_importance, plot_feature_importance_heatmap
from visualise_data import plot_personalized_accuracies_combined_negatives, plot_accuracies_negatives, plot_accuracy_comparison, plot_accuracy_comparison_general_personalised


param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
logistic_model = LogisticRegression(max_iter = 250, C = 1/10, warm_start = True)

def get_basic_model(train_features, train_labels):
    best_model = logistic_model.fit(train_features, train_labels)
    return best_model

def get_best_model(train_features, train_labels):
    """Searches for the best regularisation metric to use"""
    logistic_model = LogisticRegression(max_iter=250, C=1/10, warm_start=True)
    
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
    return (label.replace("<", "lt")
                 .replace(">", "gt")
                 .replace("+", "plus")
                 .replace(" ", "_")
                 .replace("-", "_")
                 .lower())

def calculate_personalized_accuracies(age_datasets, print_results=True, return_models=False, importance_prefix=None):
    """Train personalized models for each detailed age bin from get_split_age_datasets."""
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
            output_path = f"{importance_prefix}_{_sanitize_label(group_key)}.png"
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
    """Calculate and optionally display accuracy for each age bracket for the general model"""
    # Define age brackets
    age_labels = ['<30', '30-39', '40-49', '50-59', '60+']
    age_bins = [0, 30, 40, 50, 60, float('inf')]
    
    # Create age groups
    test_age_groups = pd.cut(test_age, bins=age_bins, labels=age_labels, right=False)
    
    # Get predictions for all test data
    predictions = model.predict(test_features)
    
    if print_results:
        # Calculate accuracy for each age bracket
        print("\n" + "="*50)
        print("Accuracy by Age Group:")
        print("="*50)
    
    accuracies = {}
    for age_label in age_labels:
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

def _compute_roc_metrics(model, test_features, test_labels):
    """Compute ROC curve metrics if both classes are present."""
    unique_classes = np.unique(test_labels)
    if len(unique_classes) < 2:
        print("Skipping ROC curve - only one class present in labels.")
        return None

    probas = model.predict_proba(test_features)[:, 1]
    fpr, tpr, _ = roc_curve(test_labels, probas)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


def plot_roc_curves(model_eval_data, title, filename):
    """Plot ROC curves for one or more models."""
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


def compute_general_model_metrics(model, test_features, test_labels):
    """Compute standard metrics for a single model/test set."""
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
    """Extract sample sizes for each of the 6 age groups"""
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


if __name__ == "__main__":
    # ==================== EXPERIMENT 1: WITHOUT AGE ====================
    print("="*60)
    print("EXPERIMENT 1: MODEL WITHOUT AGE AS A FEATURE")
    print("="*60)

    metrics_summary = {}

    # Load data without age
    train_features_no_age, train_labels_no_age, train_age_no, test_features_no_age, test_labels_no_age, test_age = get_cleaned_train_test(False)
    
    # Train model
    model_without_age = get_best_model(train_features_no_age, train_labels_no_age)
    importance_df_no_age = get_logreg_feature_importance(
        train_features_no_age.columns,
        model_without_age,
        "feature_importance_general_without_age.png"
    )
    
    # Report overall accuracy
    accuracy_no_age = model_without_age.score(test_features_no_age, test_labels_no_age)
    print(f"Overall Test Set No Age Accuracy: {accuracy_no_age:.4f}")

    gm_no_age_metrics = compute_general_model_metrics(model_without_age, test_features_no_age, test_labels_no_age)
    metrics_summary["Generic Model - Without Age"] = gm_no_age_metrics

    # Get accuracies across age groups
    accuracies_without_age = calculate_accuracy_across_ages(
        model_without_age, test_features_no_age, test_labels_no_age, test_age, print_results=True
    )
    
    
    # ==================== EXPERIMENT 2: WITH AGE ====================
    print("\n" + "="*60)
    print("EXPERIMENT 2: MODEL WITH AGE AS A FEATURE")
    print("="*60)
    
    # Load data with age
    train_features_with_age, train_labels_with_age, train_age_with, test_features_with_age, test_labels_with_age, test_age = get_cleaned_train_test(True)
    
    # Train model
    model_with_age = get_best_model(train_features_with_age, train_labels_with_age)
    importance_df_with_age = get_logreg_feature_importance(
        train_features_with_age.columns,
        model_with_age,
        "feature_importance_general_with_age.png"
    )
    
    # Report overall accuracy
    accuracy_with_age = model_with_age.score(test_features_with_age, test_labels_with_age)
    print(f"Overall Test Set With Age Accuracy: {accuracy_with_age:.4f}")

    gm_with_age_metrics = compute_general_model_metrics(model_with_age, test_features_with_age, test_labels_with_age)
    
    metrics_summary["Generic Model - With Age"] = gm_with_age_metrics

    # Get accuracies across age groups
    accuracies_with_age = calculate_accuracy_across_ages(
        model_with_age, test_features_with_age, test_labels_with_age, test_age, print_results=True
    )
    
    
    # ==================== COMPARISON ====================
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"Overall Accuracy WITHOUT Age: {accuracy_no_age:.4f}")
    print(f"Overall Accuracy WITH Age:    {accuracy_with_age:.4f}")
    print(f"Difference (With - Without):  {accuracy_with_age - accuracy_no_age:+.4f}")
    print("="*60 + "\n")
    
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
    plot_roc_curves(general_model_eval, "ROC Curves - General Models", "roc_general_models.png")
    
    
    # ==================== EXPERIMENT 3: PERSONALIZED MODELS BY AGE ====================
    print("\n" + "="*60)
    print("EXPERIMENT 3: PERSONALIZED MODELS FOR EACH AGE GROUP")
    print("="*60)
    
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

    # Plot ROC curves for personalized models (without and with age)
    personalized_eval_entries_no_age = [
        {**data, "label": f"No Age - {label}"}
        for label, data in personalized_eval_no_age.items()
    ]
    plot_roc_curves(
        personalized_eval_entries_no_age,
        "ROC Curves - Personalized Models (No Age)",
        "roc_personalized_no_age.png"
    )

    personalized_eval_entries_with_age = [
        {**data, "label": f"With Age - {label}"}
        for label, data in personalized_eval_with_age.items()
    ]
    plot_roc_curves(
        personalized_eval_entries_with_age,
        "ROC Curves - Personalized Models (With Age)",
        "roc_personalized_with_age.png"
    )

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

    plot_feature_importance_heatmap(importance_heatmap_data, "feature_importance_heatmap.png")

    metrics_df = pd.DataFrame(metrics_summary)
    desired_rows = ["Test Accuracy", "Test AUC", "FN", "FP", "TP", "TN"]
    metrics_df = metrics_df.reindex(desired_rows)
    print("\n" + "="*60)
    print("MODEL METRIC SUMMARY")
    print("="*60)
    print(metrics_df)
    metrics_df.to_csv('model_metrics_summary.csv')
    print("Metric summary saved as 'model_metrics_summary.csv'")

    # Get sample sizes for the 6 age groups (using test_age from experiment 1/2)
    sample_sizes = get_sample_sizes_by_age_group(test_age)
    print("\n" + "-"*60)
    print("Sample sizes for 6 age groups:")
    print("-"*60)
    for age_label, size in sample_sizes.items():
        print(f"  {age_label}: {size}")
