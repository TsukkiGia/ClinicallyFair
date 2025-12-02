
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

from extract_features import get_cleaned_train_test, get_split_age_datasets


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


def get_sample_sizes_by_age_group(test_age):
    """Extract sample sizes for each of the 6 age groups"""
    age_labels = ['<30', '30-39', '40-49', '50-59', '60-69', '70+']
    age_bins = [0, 30, 40, 50, 60, 70, 120]
    
    # Create age groups
    test_age_groups = pd.cut(test_age, bins=age_bins, labels=age_labels, right=False)
    
    # Count samples in each group
    sample_sizes = {}
    for age_label in age_labels:
        age_mask = test_age_groups == age_label
        sample_sizes[age_label] = age_mask.sum()
    
    return sample_sizes


def combine_age_groups_to_three(accuracies_dict, sample_sizes_dict):
    """Combine 6 age groups into 3: <40, 40-50, >50
    Using weighted average based on sample sizes"""
    # Map fine-grained groups to coarse groups
    mapping = {
        '<30': 'young',
        '30-39': 'young',
        '40-49': 'mid',
        '50-59': 'old',
        '60-69': 'old',
        '70+': 'old'
    }
    
    # Collect accuracies and sample sizes by coarse group
    combined_accs = {'young': [], 'mid': [], 'old': []}
    combined_sizes = {'young': [], 'mid': [], 'old': []}
    
    for fine_label, coarse_label in mapping.items():
        if fine_label in accuracies_dict and fine_label in sample_sizes_dict:
            combined_accs[coarse_label].append(accuracies_dict[fine_label])
            combined_sizes[coarse_label].append(sample_sizes_dict[fine_label])
    
    # Calculate weighted average for each group
    result = {}
    for group in combined_accs.keys():
        if combined_accs[group] and combined_sizes[group]:
            accs = np.array(combined_accs[group])
            sizes = np.array(combined_sizes[group])
            
            # Weighted average: sum(accuracy * size) / sum(size)
            if sizes.sum() > 0:
                result[group] = np.sum(accs * sizes) / sizes.sum()
    
    return result


def calculate_personalized_accuracies(age_datasets, print_results=True):
    """Train personalized models for each age group (young, mid, old) and return accuracies"""
    personalized_accuracies = {}
    
    age_group_names = {
        'young': 'Young (<40)',
        'mid': 'Mid (40-50)',
        'old': 'Old (>50)'
    }
    
    if print_results:
        print("\n" + "="*60)
        print("TRAINING PERSONALIZED MODELS FOR EACH AGE GROUP")
        print("="*60)
    
    for group_key, group_name in age_group_names.items():
        if print_results:
            print(f"\n{group_name}:")
            print("-" * 40)
        
        # Get train and test data for this age group
        train_features_group, train_labels_group = age_datasets[group_key]["train"]
        test_features_group, test_labels_group = age_datasets[group_key]["test"]
        
        if print_results:
            print(f"  Train samples: {len(train_features_group)}")
            print(f"  Test samples: {len(test_features_group)}")
        
        if len(train_features_group) == 0 or len(test_features_group) == 0:
            if print_results:
                print(f"  Skipping {group_name} - insufficient data")
            continue
        
        # Train personalized model for this age group (without age as feature)
        personalized_model = get_best_model(train_features_group, train_labels_group)
        personalized_acc = personalized_model.score(test_features_group, test_labels_group)
        personalized_accuracies[group_key] = personalized_acc
        
        if print_results:
            print(f"  Personalized Model Accuracy: {personalized_acc:.4f}")
    
    return personalized_accuracies


def plot_personalized_accuracies(personalized_accuracies):
    """Plot bar chart showing accuracies of personalized models for each age group"""
    age_groups = ['Young (<40)', 'Mid (40-50)', 'Old (>50)']
    group_keys = ['young', 'mid', 'old']
    
    # Extract accuracies
    accuracies = [personalized_accuracies.get(key, 0) for key in group_keys]
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_positions = np.arange(len(age_groups))
    bars = ax.bar(x_positions, accuracies, color=['#2E86AB', '#A23B72', '#F18F01'], 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Customize plot
    ax.set_xlabel('Age Group', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Personalized Model Accuracies by Age Group\n(Models Trained Without Age Feature)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(age_groups)
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add horizontal line at mean accuracy
    mean_acc = np.mean(accuracies)
    ax.axhline(y=mean_acc, color='red', linestyle='--', linewidth=2, alpha=0.7, 
               label=f'Mean: {mean_acc:.4f}')
    ax.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('personalized_accuracies.png', dpi=300, bbox_inches='tight')
    print("\nPersonalized accuracies plot saved as 'personalized_accuracies.png'")
    plt.show()


def plot_personalized_accuracies_combined_negatives(accuracies_no_age, accuracies_with_age):
    """Plot diverging bar chart comparing personalized models with and without age feature.
    Shows difference on positive/negative scale."""
    
    age_groups = ['Young (<40)', 'Mid (40-50)', 'Old (>50)']
    group_keys = ['young', 'mid', 'old']
    
    # Calculate differences (with_age - no_age)
    differences = []
    for key in group_keys:
        if key in accuracies_no_age and key in accuracies_with_age:
            diff = accuracies_with_age[key] - accuracies_no_age[key]
            differences.append(diff)
        else:
            differences.append(0)
    
    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Color bars based on positive/negative
    colors = ['green' if d > 0 else 'red' if d < 0 else 'gray' for d in differences]
    
    y_positions = np.arange(len(age_groups))
    bars = ax.barh(y_positions, differences, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Customize plot
    ax.set_yticks(y_positions)
    ax.set_yticklabels(age_groups, fontsize=11)
    ax.set_xlabel('Change in Accuracy (With Age - Without Age)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Age Group', fontsize=12, fontweight='bold')
    ax.set_title('Impact of Including Age in Personalized Models\n(Positive = With Age Better, Negative = Without Age Better)', 
                 fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=2, linestyle='-')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (bar, diff) in enumerate(zip(bars, differences)):
        width = bar.get_width()
        label_x = width + (0.005 if width > 0 else -0.005)
        ha = 'left' if width > 0 else 'right'
        ax.text(label_x, bar.get_y() + bar.get_height()/2., 
                f'{diff:+.4f}',
                ha=ha, va='center', fontsize=11, fontweight='bold')
    
    # Add accuracy values as text annotations on the left side
    for i, key in enumerate(group_keys):
        no_age_acc = accuracies_no_age.get(key, 0)
        with_age_acc = accuracies_with_age.get(key, 0)
        ax.text(-0.15, i, 
                f'No Age: {no_age_acc:.3f}\nWith Age: {with_age_acc:.3f}', 
                ha='right', va='center', fontsize=9, 
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.6, edgecolor='black'))
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='With Age Better', edgecolor='black'),
        Patch(facecolor='red', alpha=0.7, label='Without Age Better', edgecolor='black'),
        Patch(facecolor='gray', alpha=0.7, label='No Change', edgecolor='black')
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('personalized_with_vs_without_age.png', dpi=300, bbox_inches='tight')
    print("\nCombined personalized plot saved as 'personalized_with_vs_without_age.png'")
    plt.show()


# def plot_young_mid_old_accuracies_negatives(baseline_accuracies, comparison_accuracies, 
#                                             baseline_label, comparison_label, filename):
#     """Plot diverging bar chart comparing two sets of accuracies for 3 age groups.
#     Positive changes (comparison better) go right, negative (baseline better) go left."""
    
#     age_groups = ['Young (<40)', 'Mid (40-50)', 'Old (>50)']
#     group_keys = ['young', 'mid', 'old']
    
#     # Calculate differences (comparison - baseline)
#     differences = []
    
#     for key in group_keys:
#         if key in baseline_accuracies and key in comparison_accuracies:
#             diff = comparison_accuracies[key] - baseline_accuracies[key]
#             differences.append(diff)
#         else:
#             differences.append(0)
    
#     # Create horizontal bar chart
#     fig, ax = plt.subplots(figsize=(10, 6))
    
#     # Color bars based on positive/negative
#     colors = ['green' if d > 0 else 'red' for d in differences]
    
#     y_positions = np.arange(len(age_groups))
#     bars = ax.barh(y_positions, differences, color=colors, alpha=0.7, edgecolor='black')
    
#     # Customize plot
#     ax.set_yticks(y_positions)
#     ax.set_yticklabels(age_groups)
#     ax.set_xlabel(f'Change in Accuracy ({comparison_label} - {baseline_label})', fontsize=12)
#     ax.set_ylabel('Age Group', fontsize=12)
#     ax.set_title(f'{comparison_label} vs {baseline_label} Accuracy by Age Group', 
#                  fontsize=14, fontweight='bold')
#     ax.axvline(x=0, color='black', linewidth=1.5, linestyle='-')
#     ax.grid(axis='x', alpha=0.3)
    
#     # Add value labels on bars
#     for i, (bar, diff) in enumerate(zip(bars, differences)):
#         width = bar.get_width()
#         label_x = width + (0.005 if width > 0 else -0.005)
#         ha = 'left' if width > 0 else 'right'
#         ax.text(label_x, bar.get_y() + bar.get_height()/2., 
#                 f'{diff:+.4f}',
#                 ha=ha, va='center', fontsize=10, fontweight='bold')
    
#     # Add accuracy values as text annotations on the left side
#     for i, key in enumerate(group_keys):
#         baseline_acc = baseline_accuracies.get(key, 0)
#         comparison_acc = comparison_accuracies.get(key, 0)
#         ax.text(-0.15, i, 
#                 f'{baseline_label[:10]}: {baseline_acc:.3f}\n{comparison_label[:10]}: {comparison_acc:.3f}', 
#                 ha='right', va='center', fontsize=9, 
#                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.5))
    
#     # Legend
#     from matplotlib.patches import Patch
#     legend_elements = [
#         Patch(facecolor='green', alpha=0.7, label=f'{comparison_label} Better'),
#         Patch(facecolor='red', alpha=0.7, label=f'{baseline_label} Better')
#     ]
#     ax.legend(handles=legend_elements, loc='best')
    
#     plt.tight_layout()
#     plt.savefig(filename, dpi=300, bbox_inches='tight')
#     print(f"Plot saved as '{filename}'")
#     plt.show()


if __name__ == "__main__":
    # ==================== EXPERIMENT 1: WITHOUT AGE ====================
    print("="*60)
    print("EXPERIMENT 1: MODEL WITHOUT AGE AS A FEATURE")
    print("="*60)
    
    # Load data without age
    train_features_no_age, train_labels_no_age, train_age_no, test_features_no_age, test_labels_no_age, test_age = get_cleaned_train_test(False)
    
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
    train_features_with_age, train_labels_with_age, train_age_with, test_features_with_age, test_labels_with_age, test_age = get_cleaned_train_test(True)
    
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
    
    
    # ==================== EXPERIMENT 3: PERSONALIZED MODELS BY AGE ====================
    print("\n" + "="*60)
    print("EXPERIMENT 3: PERSONALIZED MODELS FOR EACH AGE GROUP")
    print("="*60)
    
    # Get age-split datasets (without age as feature)
    age_datasets_no_age = get_split_age_datasets(young_age=40, old_age=50, include_age=False)
    
    # Train personalized models for each age group (without age)
    personalized_accuracies_no_age = calculate_personalized_accuracies(
        age_datasets_no_age, print_results=True
    )
    
    # Get age-split datasets (WITH age as feature)
    age_datasets_with_age = get_split_age_datasets(young_age=40, old_age=50, include_age=True)
    
    # Train personalized models for each age group (with age)
    print("\n" + "="*60)
    print("TRAINING PERSONALIZED MODELS WITH AGE FEATURE")
    print("="*60)
    personalized_accuracies_with_age = calculate_personalized_accuracies(
        age_datasets_with_age, print_results=True
    )
    
    # Plot combined personalized model accuracies (diverging plot)
    plot_personalized_accuracies_combined_negatives(personalized_accuracies_no_age, personalized_accuracies_with_age)
    
    # Get sample sizes for the 6 age groups (using test_age from experiment 1/2)
    sample_sizes = get_sample_sizes_by_age_group(test_age)
    print("\n" + "-"*60)
    print("Sample sizes for 6 age groups:")
    print("-"*60)
    for age_label, size in sample_sizes.items():
        print(f"  {age_label}: {size}")
    
    # # Combine the 6 age groups from Experiment 1 (without age) into 3 groups
    # print("\n" + "-"*60)
    # print("Combining general model (no age) accuracies into 3 groups (weighted):")
    # print("-"*60)
    # general_no_age_3groups = combine_age_groups_to_three(accuracies_without_age, sample_sizes)
    # for group_key in ['young', 'mid', 'old']:
    #     if group_key in general_no_age_3groups:
    #         print(f"  {group_key}: {general_no_age_3groups[group_key]:.4f}")
    
    # # Combine the 6 age groups from Experiment 2 (with age) into 3 groups
    # print("\n" + "-"*60)
    # print("Combining general model (with age) accuracies into 3 groups (weighted):")
    # print("-"*60)
    # general_with_age_3groups = combine_age_groups_to_three(accuracies_with_age, sample_sizes)
    # for group_key in ['young', 'mid', 'old']:
    #     if group_key in general_with_age_3groups:
    #         print(f"  {group_key}: {general_with_age_3groups[group_key]:.4f}")
    
    
    # # ==================== COMPARISON 1: Personalized vs General (both without age) ====================
    # print("\n" + "="*60)
    # print("COMPARISON 1: Personalized (No Age) vs General (No Age)")
    # print("="*60)
    # for group_key in ['young', 'mid', 'old']:
    #     if group_key in personalized_accuracies_no_age and group_key in general_no_age_3groups:
    #         pers_acc = personalized_accuracies_no_age[group_key]
    #         gen_acc = general_no_age_3groups[group_key]
    #         diff = pers_acc - gen_acc
    #         print(f"{group_key:6s}: Personalized={pers_acc:.4f}, General={gen_acc:.4f}, Diff={diff:+.4f}")
    # print("="*60 + "\n")
    
    # # Plot comparison 1
    # plot_young_mid_old_accuracies_negatives(
    #     baseline_accuracies=general_no_age_3groups,
    #     comparison_accuracies=personalized_accuracies_no_age,
    #     baseline_label='General (No Age)',
    #     comparison_label='Personalized (No Age)',
    #     filename='personalized_vs_general_no_age.png'
    # )
    
    
    # # ==================== COMPARISON 2: Personalized (no age) vs General (with age) ====================
    # print("\n" + "="*60)
    # print("COMPARISON 2: Personalized (No Age) vs General (With Age)")
    # print("="*60)
    # for group_key in ['young', 'mid', 'old']:
    #     if group_key in personalized_accuracies_no_age and group_key in general_with_age_3groups:
    #         pers_acc = personalized_accuracies_no_age[group_key]
    #         gen_with_age_acc = general_with_age_3groups[group_key]
    #         diff = pers_acc - gen_with_age_acc
    #         print(f"{group_key:6s}: Personalized={pers_acc:.4f}, General(w/Age)={gen_with_age_acc:.4f}, Diff={diff:+.4f}")
    # print("="*60 + "\n")
    
    # # Plot comparison 2
    # plot_young_mid_old_accuracies_negatives(
    #     baseline_accuracies=general_with_age_3groups,
    #     comparison_accuracies=personalized_accuracies_no_age,
    #     baseline_label='General (With Age)',
    #     comparison_label='Personalized (No Age)',
    #     filename='personalized_no_age_vs_general_with_age.png'
    # )
