import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from extract_features import get_cleaned_train_test

# Get features with age included
train_features, train_labels, train_age, test_features, test_labels, test_age = get_cleaned_train_test(True)



def plot_cancer_age_distribution():
    # Combine features and labels into a single DataFrame
    train_data = train_features.copy()
    train_data['cancer'] = train_labels

    # Create age groups for better visualization
    train_data['age_group'] = pd.cut(
        train_data['Age'], 
        bins=[0, 30, 40, 50, 60, float('inf')],
        labels=['<30', '30-39', '40-49', '50-59', '60+']
    )

    # Plot distribution of age groups by cancer status
    plt.figure(figsize=(10, 6))
    sns.countplot(x='age_group', hue='cancer', data=train_data)
    plt.title('Distribution of Age Groups by Cancer Status')
    plt.xlabel('Age Group')
    plt.ylabel('Count')
    plt.legend(labels=['No Cancer', 'Cancer'])
    plt.tight_layout()
    plt.savefig('age_distribution_by_cancer.png', dpi=300, bbox_inches='tight')

    print(f"\nDataset shape: {train_data.shape}")
    print(f"\nCancer distribution:\n{train_data['cancer'].value_counts()}")
    print(f"\nAge group distribution:\n{train_data['age_group'].value_counts().sort_index()}")

def plot_personalized_accuracies_combined_negatives(accuracies_no_age, accuracies_with_age):
    """Plot diverging bar chart comparing personalized models with and without age feature.
    Shows difference on positive/negative scale."""

    if not accuracies_no_age or not accuracies_with_age:
        print("Insufficient personalized accuracy data to compare.")
        return

    age_order = ['<30', '30-39', '40-49', '50-59', '60+']
    group_keys = [label for label in age_order if label in accuracies_no_age and label in accuracies_with_age]
    group_keys.extend([
        label for label in accuracies_no_age
        if label in accuracies_with_age and label not in group_keys
    ])

    if not group_keys:
        print("No overlapping age groups between personalized accuracy runs.")
        return

    age_groups = group_keys

    differences = [accuracies_with_age[key] - accuracies_no_age[key] for key in group_keys]

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['green' if d > 0 else 'red' if d < 0 else 'gray' for d in differences]

    y_positions = np.arange(len(age_groups))
    bars = ax.barh(y_positions, differences, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(age_groups, fontsize=11)
    ax.set_xlabel('Change in Accuracy (With Age - Without Age)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Age Group', fontsize=12, fontweight='bold')
    ax.set_title('Impact of Including Age in Personalized Models\n(Positive = With Age Better, Negative = Without Age Better)', 
                 fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=2, linestyle='-')
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    for i, (bar, diff) in enumerate(zip(bars, differences)):
        width = bar.get_width()
        label_x = width + (0.005 if width > 0 else -0.005)
        ha = 'left' if width > 0 else 'right'
        ax.text(label_x, bar.get_y() + bar.get_height()/2., 
                f'{diff:+.4f}',
                ha=ha, va='center', fontsize=11, fontweight='bold')

    for i, key in enumerate(group_keys):
        no_age_acc = accuracies_no_age.get(key, 0)
        with_age_acc = accuracies_with_age.get(key, 0)
        ax.text(-0.15, i, 
                f'No Age: {no_age_acc:.3f}\nWith Age: {with_age_acc:.3f}', 
                ha='right', va='center', fontsize=9, 
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.6, edgecolor='black'))

    
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='With Age Better', edgecolor='black'),
        Patch(facecolor='red', alpha=0.7, label='Without Age Better', edgecolor='black'),
        Patch(facecolor='gray', alpha=0.7, label='No Change', edgecolor='black')
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=10)

    plt.tight_layout()
    plt.savefig('personalized_with_vs_without_age.png', dpi=300, bbox_inches='tight')
    print("\nCombined personalized plot saved as 'personalized_with_vs_without_age.png'")
    plt.close()


def plot_accuracy_comparison(accuracies_without_age, accuracies_with_age):
    """Plot comparison of accuracies across age groups for general models with age and without age"""
    age_labels = ['<30', '30-39', '40-49', '50-59', '60+']
    
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
    plt.savefig('general_models_accuracy_comparison_by_age.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'accuracy_comparison_by_age.png'")
    plt.close()


def plot_accuracy_comparison_general_personalised(general_accuracies_without_age, general_accuracies_with_age, personalized_accuracies_no_age):
    """Plot comparison of accuracies across age groups for:
    - General model without age
    - General model with age  
    - Personalized models without age
    """
    age_labels = ['<30', '30-39', '40-49', '50-59', '60+']
    
    # Extract accuracies for plotting (only for age groups that exist in all three)
    acc_general_without = [general_accuracies_without_age.get(label, None) for label in age_labels]
    acc_general_with = [general_accuracies_with_age.get(label, None) for label in age_labels]
    acc_personalized = [personalized_accuracies_no_age.get(label, None) for label in age_labels]
    
    # Filter out None values and corresponding labels
    filtered_labels = []
    filtered_general_without = []
    filtered_general_with = []
    filtered_personalized = []
    
    for i, label in enumerate(age_labels):
        if (acc_general_without[i] is not None and 
            acc_general_with[i] is not None and 
            acc_personalized[i] is not None):
            filtered_labels.append(label)
            filtered_general_without.append(acc_general_without[i])
            filtered_general_with.append(acc_general_with[i])
            filtered_personalized.append(acc_personalized[i])
    
    # Create bar plot with 3 bars per age group
    x = np.arange(len(filtered_labels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 7))
    bars1 = ax.bar(x - width, filtered_general_without, width, label='General - Without Age', alpha=0.8, color='#1f77b4')
    bars2 = ax.bar(x, filtered_general_with, width, label='General - With Age', alpha=0.8, color='#ff7f0e')
    bars3 = ax.bar(x + width, filtered_personalized, width, label='Personalized - Without Age', alpha=0.8, color='#2ca02c')
    
    ax.set_xlabel('Age Group', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Model Accuracy Comparison: General vs Personalized Models', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(filtered_labels)
    ax.legend(loc='best', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.2])
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('general_vs_personalized_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'general_vs_personalized_accuracy_comparison.png'")
    plt.close()


def plot_accuracies_negatives(accuracies_without_age, accuracies_with_age):
    """Plot diverging bar chart showing accuracy changes when age is included.
    Positive changes (improvements) go right, negative changes (decreases) go left."""
    
    age_labels = ['<30', '30-39', '40-49', '50-59', '60+']
    
    # Calculate differences (with_age - without_age)
    differences = []
    valid_labels = []
    zero_change_labels = []
    
    for label in age_labels:
        if label in accuracies_without_age and label in accuracies_with_age:
            diff = accuracies_with_age[label] - accuracies_without_age[label]
            differences.append(diff)
            valid_labels.append(label)
            if np.isclose(diff, 0):
                zero_change_labels.append(label)

    if not differences:
        print("No age groups have overlapping accuracy results to plot.")
        return
    
    if zero_change_labels:
        print("No accuracy change detected for:", ", ".join(zero_change_labels))
    
    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color bars based on positive/negative (gray for zero change)
    colors = []
    for diff in differences:
        if np.isclose(diff, 0):
            colors.append('gray')
        elif diff > 0:
            colors.append('green')
        else:
            colors.append('red')
    
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
    
    # Legend
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Accuracy Increases'),
        Patch(facecolor='red', alpha=0.7, label='Accuracy Decreases')
    ]
    ax.legend(handles=legend_elements, loc='best')
    
    plt.tight_layout()
    plt.savefig('accuracy_changes_by_age.png', dpi=300, bbox_inches='tight')
    print("Diverging plot saved as 'accuracy_changes_by_age.png'")
    plt.close()


if __name__ == "__main__":
    plot_cancer_age_distribution()