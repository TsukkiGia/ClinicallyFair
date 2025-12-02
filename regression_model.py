
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# from tqdm import tqdm
import warnings
import copy
# import seaborn as sns
# import requests
import os
import urllib
from collections import namedtuple
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
np.random.seed(42)

from extract_features import get_cleaned_train_test


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


def calculate_accuracy_across_ages(model, test_features, test_labels, test_age, print_results=True):
    """Calculate and optionally display accuracy for each age bracket"""
    # Define age brackets
    age_labels = ['<30', '30-39', '40-49', '50-59', '60-69', '70+']
    age_bins = [0, 30, 40, 50, 60, 70, 120]
    
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


def plot_accuracy_comparison(accuracies_without_age, accuracies_with_age):
    """Plot comparison of accuracies across age groups for both models"""
    age_labels = ['<30', '30-39', '40-49', '50-59', '60-69', '70+']
    
    # Extract accuracies for plotting (only for age groups that exist in both)
    acc_without = [accuracies_without_age.get(label, None) for label in age_labels]
    acc_with = [accuracies_with_age.get(label, None) for label in age_labels]
    
    # Filter out None values and corresponding labels
    filtered_labels = []
    filtered_without = []
    filtered_with = []
    
    for i, label in enumerate(age_labels):
        if acc_without[i] is not None and acc_with[i] is not None:
            filtered_labels.append(label)
            filtered_without.append(acc_without[i])
            filtered_with.append(acc_with[i])
    
    # Create bar plot
    x = np.arange(len(filtered_labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, filtered_without, width, label='Without Age', alpha=0.8)
    bars2 = ax.bar(x + width/2, filtered_with, width, label='With Age', alpha=0.8)
    
    ax.set_xlabel('Age Group', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Model Accuracy Across Age Groups: With vs Without Age as Feature', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(filtered_labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('accuracy_comparison_by_age.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'accuracy_comparison_by_age.png'")
    plt.show()


def plot_accuracies_negatives(accuracies_without_age, accuracies_with_age):
    """Plot diverging bar chart showing accuracy changes when age is included.
    Positive changes (improvements) go right, negative changes (decreases) go left."""
    
    age_labels = ['<30', '30-39', '40-49', '50-59', '60-69', '70+']
    
    # Calculate differences (with_age - without_age)
    differences = []
    valid_labels = []
    
    for label in age_labels:
        if label in accuracies_without_age and label in accuracies_with_age:
            diff = accuracies_with_age[label] - accuracies_without_age[label]
            differences.append(diff)
            valid_labels.append(label)
    
    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color bars based on positive/negative
    colors = ['green' if d > 0 else 'red' for d in differences]
    
    y_positions = np.arange(len(valid_labels))
    bars = ax.barh(y_positions, differences, color=colors, alpha=0.7, edgecolor='black')
    
    # Customize plot
    ax.set_yticks(y_positions)
    ax.set_yticklabels(valid_labels)
    ax.set_xlabel('Change in Accuracy (With Age - Without Age)', fontsize=12)
    ax.set_ylabel('Age Group', fontsize=12)
    ax.set_title('Impact of Including Age on Model Accuracy by Age Group', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=1.5, linestyle='-')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, diff) in enumerate(zip(bars, differences)):
        width = bar.get_width()
        label_x = width + (0.005 if width > 0 else -0.005)
        ha = 'left' if width > 0 else 'right'
        ax.text(label_x, bar.get_y() + bar.get_height()/2., 
                f'{diff:+.4f}',
                ha=ha, va='center', fontsize=10, fontweight='bold')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Accuracy Increases'),
        Patch(facecolor='red', alpha=0.7, label='Accuracy Decreases')
    ]
    ax.legend(handles=legend_elements, loc='best')
    
    plt.tight_layout()
    plt.savefig('accuracy_changes_by_age.png', dpi=300, bbox_inches='tight')
    print("Diverging plot saved as 'accuracy_changes_by_age.png'")
    plt.show()

if __name__ == "__main__":
    # ==================== EXPERIMENT 1: WITHOUT AGE ====================
    print("="*60)
    print("EXPERIMENT 1: MODEL WITHOUT AGE AS A FEATURE")
    print("="*60)
    
    # Load data without age
    train_features_no_age, train_labels_no_age, test_features_no_age, test_labels_no_age, test_age = get_cleaned_train_test(False)
    
    # Train model
    model_without_age = get_best_model(train_features_no_age, train_labels_no_age)
    
    # Report overall accuracy
    accuracy_no_age = model_without_age.score(test_features_no_age, test_labels_no_age)
    print(f"Overall Test Set No Age Accuracy: {accuracy_no_age:.4f}")
    
    # Get accuracies across age groups
    accuracies_without_age = calculate_accuracy_across_ages(
        model_without_age, test_features_no_age, test_labels_no_age, test_age, print_results=True
    )
    
    
    # ==================== EXPERIMENT 2: WITH AGE ====================
    print("\n" + "="*60)
    print("EXPERIMENT 2: MODEL WITH AGE AS A FEATURE")
    print("="*60)
    
    # Load data with age
    train_features_with_age, train_labels_with_age, test_features_with_age, test_labels_with_age, test_age = get_cleaned_train_test(True)
    
    # Train model
    model_with_age = get_best_model(train_features_with_age, train_labels_with_age)
    
    # Report overall accuracy
    accuracy_with_age = model_with_age.score(test_features_with_age, test_labels_with_age)
    print(f"Overall Test Set With Age Accuracy: {accuracy_with_age:.4f}")
    
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
