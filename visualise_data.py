import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# from tqdm import tqdm
import warnings
import copy
import seaborn as sns
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

if __name__ == "__main__":
    plot_cancer_age_distribution()