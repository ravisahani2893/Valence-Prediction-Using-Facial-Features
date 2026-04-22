import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error


# ============================================================================
# Exploratory Data Analysis (EDA) for the "Semaine" dataset
# ============================================================================

# ============================================================================
# 1. Load train, test and validation datasets
# ============================================================================

train_data = pd.read_csv('visual_train_features.csv')
val_data   = pd.read_csv('visual_val_features.csv')
test_data  = pd.read_csv('visual_test_features.csv')

# ============================================================================
# 2. Display basic information about the datasets
# ============================================================================
print("Train Data:")
print(train_data.info())
print(train_data.head())

print("\nValidation Data:")
print(val_data.info())
print(val_data.head())

print("\nTest Data:")
print(test_data.info())
print(test_data.head())

# ============================================================================
# 3. Checking shape of the datasets
# ============================================================================

# Observation: 
# 1. Train dataset has 70,090, validation dataset has 33,349 and test dataset has 15,765 samples.
# 2. Each dataset contains 422 feature variables along with the target variable 'unused_target_label'.
print(f"Train Data Shape: {train_data.shape}")
print(f"Validation Data Shape: {val_data.shape}")
print(f"Test Data Shape: {test_data.shape}")

# ============================================================================
# 4. Checking for missing and duplicates values
# ============================================================================


# ============================================================================
# 4.1 Checking for missing values in the Train dataset
# ============================================================================

# Checking for missing values in the train dataset
nan_counts = train_data.isnull().sum()

# Filtering columns that have missing values
nan_cols = nan_counts[nan_counts > 0]

# Observation: There are no missing values in the train dataset.
print(f"Number of columns with missing values in train dataset: {len(nan_cols)}")
print(f"Columns with missing values in training dataset: {nan_cols.index.tolist()}")

#============================================================================


# ============================================================================
# 4.2 Checking for missing values in the Validation dataset
# ============================================================================

# Checking for missing values in the train dataset
val_nan_counts = val_data.isnull().sum()

# Filtering columns that have missing values
val_nan_cols = val_nan_counts[val_nan_counts > 0]

# Observation: There are no missing values in the train dataset.
print(f"Number of columns with missing values in validation dataset: {len(val_nan_cols)}")
print(f"Columns with missing values in validation dataset: {val_nan_cols.index.tolist()}")

#============================================================================


# ============================================================================
# 4.3 Checking for missing values in the Test dataset
# ============================================================================

# Checking for missing values in the test dataset
test_nan_counts = test_data.isnull().sum()

# Filtering columns that have missing values
test_nan_cols = test_nan_counts[test_nan_counts > 0]

# Observation: There are no missing values in the train dataset.
print(f"Number of columns with missing values in test dataset: {len(test_nan_cols)}")
print(f"Columns with missing values in test dataset: {test_nan_cols.index.tolist()}")

#============================================================================

# ============================================================================
# 5. Checking for duplicate rows in the Train dataset
# ============================================================================

# Checking for duplicate rows in the train dataset. 
# Observation: There are no duplicate rows in the train dataset.
duplicate_count = train_data.duplicated().sum()
print(f"Number of duplicate rows in train dataset: {duplicate_count}")


# ============================================================================
# 5. Checking statistics analysis of the target variable 
# ============================================================================

# Identifying the target variable column in the train dataset.
target_column = [column for column in train_data.columns if 'target' in column.lower() or 'label' in column.lower()][0]
print(f"Identified target variable column: {target_column}")

# Extracting the target variable from the train dataset.
train_data_target = train_data[target_column]

# Displaying statistics of the target variable in the train dataset.
print(f"\n--- Train dataset target variable statistics ---")
print(f"Mean:   {train_data_target.mean():.6f}")
print(f"Std:    {train_data_target.std():.6f}")
print(f"Var:    {train_data_target.var():.6f}")
print(f"Min:    {train_data_target.min():.6f}")
print(f"Max:    {train_data_target.max():.6f}")
print(f"Skew:   {train_data_target.skew():.4f}")
print(train_data_target.quantile([0.25, 0.5, 0.75]))

# Extracting the target variable from the validation dataset.
val_data_target = val_data[target_column]

# Displaying statistics of the target variable in the validation dataset.
print(f"\n--- Validation dataset target variable statistics ---")
print(f"Mean:   {val_data_target.mean():.6f}")
print(f"Std:    {val_data_target.std():.6f}")
print(f"Var:    {val_data_target.var():.6f}")
print(f"Min:    {val_data_target.min():.6f}")
print(f"Max:    {val_data_target.max():.6f}")
print(f"Skew:   {val_data_target.skew():.4f}")
print(val_data_target.quantile([0.25, 0.5, 0.75]))

# Extracting the target variable from the test dataset.
test_data_target = test_data[target_column]

# Displaying statistics of the target variable in the test dataset.
print(f"\n--- Test dataset target variable statistics ---")
print(f"Mean:   {test_data_target.mean():.6f}")
print(f"Std:    {test_data_target.std():.6f}")
print(f"Var:    {test_data_target.var():.6f}")
print(f"Min:    {test_data_target.min():.6f}")
print(f"Max:    {test_data_target.max():.6f}")
print(f"Skew:   {test_data_target.skew():.4f}")
print(test_data_target.quantile([0.25, 0.5, 0.75]))

# 1. Train dataset:
#    - Mean:   0.134550, Std: 0.190693, Var: 0.036364, Min: -0.315208, Max: 0.470252, Skew: -0.6349
# 2. Validation dataset:
#    - Mean:   -0.094996, Std: 0.279004, Var: 0.077843, Min: -0.422096, Max: 0.497379, Skew: 0.7869
# 3. Test dataset:
#    - Mean:   -0.059549, Std: 0.141452, Var: 0.020009, Min: -0.270847, Max: 0.213010, Skew: 0.2442

# Key Observation
# The training set is positively biased (mean +0.135) with the opposite skew direction to the validation (mean -0.094996) and test sets (mean  -0.059549). This indicates that each split contains participants with different emotional baselines, which means models trained on the training data with positive mean distribution will tend to overpredict valence on the more negative validation and test sets.



# ============================================================================
# 6. Checking distribution of valence in histogram
# ============================================================================
# Extracting train target data
train_target = train_data[target_column]
os.makedirs('report', exist_ok=True)
plt.figure(figsize=(10, 5))
sns.histplot(train_target, bins=50, kde=True)
plt.title('Distribution of Valence (Train)')
plt.xlabel('Valence')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('report/eda_valence_distribution.png', dpi=150)
plt.show()

# ============================================================================
# 6. Checking distribution of valence in various graphs
# ============================================================================
# Boxplot for outliers
plt.figure(figsize=(10, 3))
sns.boxplot(x=train_target)
plt.title('Boxplot of Valence (Train)')
plt.xlabel('Valence')
plt.tight_layout()
plt.savefig('report/eda_valence_boxplot.png', dpi=150)
plt.show()


# Compare train vs validation target distributions
plt.figure(figsize=(10, 5))
sns.histplot(train_target, bins=50, kde=True, label='Train', alpha=0.5)
sns.histplot(val_data[target_column], bins=50, kde=True, label='Validation', alpha=0.5, color='orange')
plt.title('Train vs Validation Valence Distribution')
plt.xlabel('Valence')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.savefig('report/eda_train_vs_val_distribution.png', dpi=150)
plt.show()

# Compare train vs test target distributions
plt.figure(figsize=(10, 5))
sns.histplot(train_target, bins=50, kde=True, label='Train', alpha=0.5)
sns.histplot(test_data[target_column], bins=50, kde=True, label='Test', alpha=0.5, color='orange')
plt.title('Train vs Test Valence Distribution')
plt.xlabel('Valence')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.savefig('report/eda_train_vs_test_distribution.png', dpi=150)
plt.show()


# ============================================================================
# 6. Feature trend analysis
# ============================================================================

# Extract all feature columns and exclude target column
feature_cols = [c for c in train_data.columns if c != target_column]
print(f"\nTotal feature columns: {len(feature_cols)}")

# Compute correlation of all features with target (with sign, not absolute)
correlations_data = train_data[feature_cols].corrwith(train_target).sort_values()

# Negative trend: features that DECREASES as valence INCREASES
negative_trend = correlations_data[correlations_data < 0].head(20)
print("=== Top 20 Negative Trend Features ===")
print(negative_trend)

# Positive trend: features that INCREASE as valence INCREASES
positive_trend = correlations_data[correlations_data > 0].sort_values(ascending=False).head(20)
print("\n=== Top 20 Positive Trend Features ===")
print(positive_trend)

# Count feature with positive and negative trend
print(f"\nFeatures with positive trend: {(correlations_data > 0).sum()}")
print(f"Features with negative trend: {(correlations_data < 0).sum()}")


# Top 10 from each direction
top_negative = correlations_data.head(10)
top_positive = correlations_data.sort_values(ascending=False).head(10)

# Side-by-side plot
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Negative trend (left panel)
top_negative.sort_values().plot(kind='barh', ax=axes[0], color='red')
axes[0].set_title('Top 10 Negative Correlation with Valence')
axes[0].set_xlabel('Pearson Correlation')
axes[0].axvline(x=0, color='black', linewidth=0.8)

# Positive trend (right panel)
top_positive.sort_values().plot(kind='barh', ax=axes[1], color='green')
axes[1].set_title('Top 10 Positive Correlation with Valence')
axes[1].set_xlabel('Pearson Correlation')
axes[1].axvline(x=0, color='black', linewidth=0.8)

plt.suptitle('Feature Trends: Negative vs Positive Correlation with Valence', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('report/eda_top10_positive_negative_trends.png', dpi=150)
plt.show()


# Evaluate Inter-feature correlation heatmap (top 20 features only) 
# Observation:
# The heatmap shows that several of the top correlated features are also highly correlated with each other, 
# indicating potential multicollinearity. This suggests redundancy in features, which may affect linear models.
# Dimensionality reduction techniques such as PCA may help in handling this issue.
top_20_features = correlations_data.head(20).index.tolist()
corr_matrix = train_data[top_20_features].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, vmin=-1, vmax=1)
plt.title('Correlation Heatmap - Top 20 Features')
plt.tight_layout()
plt.savefig('report/eda_top20_heatmap.png', dpi=150)
plt.show()

# Constant or near-zero-variance columns
feature_stds = train_data[feature_cols].std()
nzv_cols = feature_stds[feature_stds < 1e-8].index.tolist()
print(f"Near-zero-variance columns to drop: {len(nzv_cols)}")


fig, axes = plt.subplots(1, 2, figsize=(18, 5))

# Left panel: Train vs Validation
sns.histplot(train_target, bins=50, kde=True, label='Train', alpha=0.5, ax=axes[0])
sns.histplot(val_data[target_column], bins=50, kde=True, label='Validation', 
             alpha=0.5, color='orange', ax=axes[0])
axes[0].set_title('Train vs Validation Valence Distribution')
axes[0].set_xlabel('Valence')
axes[0].set_ylabel('Frequency')
axes[0].legend()

# Right panel: Train vs Test
sns.histplot(train_target, bins=50, kde=True, label='Train', alpha=0.5, ax=axes[1])
sns.histplot(test_data[target_column], bins=50, kde=True, label='Test', 
             alpha=0.5, color='green', ax=axes[1])
axes[1].set_title('Train vs Test Valence Distribution')
axes[1].set_xlabel('Valence')
axes[1].set_ylabel('Frequency')
axes[1].legend()
plt.tight_layout()
plt.savefig('report/eda_combined_distribution.png', dpi=150, bbox_inches='tight')
plt.show()



# ============================================================================
# Baseline Modelling
# ============================================================================

# Extracting target variable from dataset
target_variable = [column for column in train_data.columns if 'target' in column.lower() or 'label' in column.lower()][0]

# Extracting feature variable excluding target variable
feature_cols = [c for c in train_data.columns if c != target_variable]
print(f"\nTotal feature columns: {len(feature_cols)}")

# Splitting training, test and validation data 
y_train      = train_data[target_variable].values
y_validation = val_data[target_variable].values
y_test       = test_data[target_variable].values


training_x = train_data[feature_cols].values
validation_x = val_data[feature_cols].values
test_x       = test_data[feature_cols].values

# Standardise features
train_x_means = np.mean(training_x, axis=0)
train_x_stds  = np.std(training_x, axis=0)

# Z-score standardisation
train_x_std      = (training_x - train_x_means) / train_x_stds
validation_x_std = (validation_x - train_x_means) / train_x_stds
test_x_std       = (test_x - train_x_means) / train_x_stds

# Standardise target using TRAIN statistic data

y_train_mean = y_train.mean()
y_train_sd   = y_train.std()
print(f"\nTarget: mean={y_train_mean:.6f}, sd={y_train_sd:.6f}")

baseline_rmse = root_mean_squared_error(y_train, np.full(y_train.shape, y_train_mean))
baseline_val_rmse = root_mean_squared_error(y_validation, np.full(y_validation.shape, y_train_mean))
baseline_test_rmse = root_mean_squared_error(y_test, np.full(y_test.shape, y_train_mean))
print(f"\n--- Baseline: Predict Mean ---")
print(f"Train RMSE: {baseline_rmse:.6f}")
print(f"Val   RMSE: {baseline_val_rmse:.6f}")
print(f"Test   RMSE: {baseline_test_rmse:.6f}")