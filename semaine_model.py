from tensorflow.python.feature_column.feature_column import linear_model
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load training, testing and validation dataset from CSV files. 
train_data = pd.read_csv('visual_test_features.csv')
test_data = pd.read_csv('visual_train_features.csv')
val_data = pd.read_csv('visual_val_features.csv')

print(train_data.head())


# Check shape of train data. 
# Observation: The shape of train data is (70090, 423) which indicates that there are 70,090 samples and 423 features in the training dataset. It indicates that training dataset is relatively large and may require feature selection or dimensionality reduction technique model training to improve performance and reduce overfitting.
print(train_data.shape)

# Check first few rows of test data
print(train_data.head())

# Check for missing values and duplicates in test data
print(train_data.isnull().sum())
print(train_data.duplicated().sum())

# Initialise target column variable
target_variable = 'unused_target_label'


# Target variable is 'unused_target_label', which we will use for computing variance, visualation and correlation.
target = train_data[target_variable]

# Compute mean value of target variable. Observation: Mean value of target variable is -0.05954900858970294, which indicates that on average the valence of the samples is slightly negative. This suggests that dataset have more negative valence samples compare to positive samples. Overall, it provides information that the user seems more likely to be annoyed compare to being delighted.
target.mean() 

# Compute variance of target variable. Observation: Variance of target variable is 0.02 which indicates that the valence values are spread moderately around the mean. A higher variance would indicate that the valence values are more spread out, while a lower variance would suggest that they are more closely clustered around the mean. In this case, a variance of 0.02 suggests that there is some variability in the target variable, but they are not extremely dispersed.
target.var()

# Compute standard deviation of target variable. Observation: Standard deviation of target variable is 0.1414, which indicates that the valence values deviate from the mean by approximately 0.1414 on average. This suggests that there is some variability in the valence values, but they are not extremely dispersed. 
target.std() 

# Compute Quantile values of target variable. 
# Observation: The quantile values of the target variable are as follows:
# 25% quantile (Q1): -0.174424, which means that 25% of the valence values are less than or equal to -0.174424,
# 50% quantile (Q2 or median): -0.078553, which means that 50% of the valence values are less than or equal to -0.078553,
# 75% quantile (Q3): 0.056986, which means that 75% of the valence values are less than or equal to 0.056986.
target.quantile([0.25, 0.5, 0.75])


# Plot boxplot of target variable to check for outliers. The boxplot will show the median, quartiles, and potential outliers in the valence variable, which can help us understand the spread and identify any extreme values that may affect our analysis or modeling.
plt.figure(figsize=(10,6))
sns.boxplot(x=target)
plt.title('Boxplot of Valence')
plt.xlabel('Valence')
plt.show()



# Check if there are any constant columns in the test data. Constant columns have only one unique value across all samples, which means they do not provide any useful information for modeling and can be safely removed. We check for constant columns by looking for columns where the number of unique values is equal to 1, excluding the target variable 'unused_target_label' since it is not a feature but a target label.
low_variance_columns = [column for column in train_data.columns if train_data[column].nunique() == 1 and column != 'unused_target_label']

# Print the number of constant columns found in test dataset. Observation: There are 0 constant columns in the test dataset, which means that all features have more than one unique value and can provide useful information for modeling.
print(f"Constant columns: {len(low_variance_columns)}")


# Correlation analysis to identify features that are highly correlated with target variable 'unused_target_label'.
# We will compute the Pearson correlation coefficient between each feature and the target variable, and then sort the features based on the absolute value of their correlation coefficients in descending order.
# This allows us to identify which features have the strongest linear relationship with the target variable, which can be useful for feature selection and understanding the underlying patterns in the data.
correlations = train_data.corr()[target.name].drop(target.name).abs().sort_values(ascending=False)

# Print top 20 features that are highly correlated with the target variable. Observation: The top 20 features that are highly correlated with the target variable 'unused_target_label' are as follows:
print(correlations.head(20))

# Select top 20 features based on absolute correlation with target variable. We will use these top features for further analysis and modeling, as they are likely to have the most significant impact on the target variable and can help improve the performance of our machine learning models.
top_features = correlations.abs().sort_values(ascending=False)[1:21].index
top_features

# Select features with absolute correlation greater than 0.05 with the target variable.
# This threshold is chosen to identify features that have a moderate to strong linear relationship with the target variable, which can be useful for feature selection and improving model performance.
# Observation: There are 294 features with an absolute correlation greater than 0.05 with the target variable 'unused_target_label', which suggests that there are many features that have a moderate to strong linear relationship with the target variable and can potentially be useful for modeling.
selected_features = correlations[abs(correlations) > 0.05].index
print(f"Number of features with absolute correlation greater than 0.05: {len(selected_features)}")
selected_features

# Feature selection using Principal Component Analysis (PCA) to reduce the dimensionality of the dataset while retaining as much variance as possible. PCA transforms the original features into a new set of uncorrelated features called principal components, which are ordered by the amount of variance they explain in the data. This can help improve model performance and reduce overfitting by eliminating redundant and less informative features.

# For PCA, we will first standardize the features to have a mean of 0 and a standard deviation of 1, which is important for PCA to work effectively since it is sensitive to the scale of the features. After standardization, we will apply PCA to select the top principal components that explain 95% of the variance in the data, which allows us to retain most of the information while reducing the number of features.
scaler = StandardScaler()
list_features=train_data.drop(target_variable, axis=1)
scaled_features = scaler.fit_transform(list_features)
pca = PCA(n_components=0.95)  # Retain 95% of variance
pca_features = pca.fit_transform(scaled_features)
pca.n_components_

# The shape of the PCA components indicates the number of principal components that were retained after applying PCA.
# In this case, the shape is (83,422), which means total 83 features were selected which have high variance 95% of the variance in the data.
pca.components_.shape

# Explained variance ratio of the PCA components indicates the proportion of variance in the original data that is by each principal component (ex. PC1, PC2 etc). The explained variance ratio helps us understand how much information (variance) is retained by each principal component and allows us to determine the number of components to retain based on the desired level of variance preservation. 
pca.explained_variance_ratio_

# Check shape of explained variance ratio. Observation: The shape of the explained variance ratio is (83,), which indicates that there are 83 PCA components that were retained after applying PCA. Each component captures a certain amount of variance in the original data. 
print(pca.explained_variance_ratio_.shape)

# List of features after applying PCA. The PCA components are the new features that are created by the PCA transformation, and they are linear combinations of the original features. Each component represents a direction in the feature space that captures a certain amount of variance in the data. The number of components retained is determined by the amount of variance we want to preserve, which in this case is 95%. The resulting PCA components can be used as input features for machine learning models, and they often lead to improved performance by reducing dimensionality and eliminating noise from the data.
pca_df = pd.DataFrame(pca_features, columns=[f'PC{i+1}' for i in range(pca.n_components_)])



# Baseline Model Training, Validating and Testing using PCA features.
# We will train a simple linear regression model using the PCA features as input and the target variable 'unused_target_label' as the output. We will evaluate the model's performance on the validation and test datasets to assess its generalization ability. The baseline model will serve as a reference point for comparing the performance of more complex models that we may develop later.



# Separate features and target variable for training the model. We will use the PCA features as input (X_train) and the target variable 'unused_target_label' as output (y_train) for training our baseline model.
X_train = train_data.drop(target_variable, axis=1)

# Separate target variable from training dataset. The target variable 'unused_target_label' is the output that we want to predict using our machine learning model, and it represents the valence of the samples in the dataset.
Y_train = train_data[target_variable]

# Check shape of X_train and Y_train dataset. 
# Observation 1: X_train has a shape of (70090, 422). X_train dataset contains 70,090 samples and 422 features (after dropping the target variable). 
# Observation 2: Y_train has a shape of (70090,), which indicates that it contains 70,090 samples of the target variable 'unused_target_label'. The number of samples in X_train and Y_train are the same, which is expected since they correspond to the same dataset.
print(X_train.shape)
print(Y_train.shape)

# Similarly, we will separate features and target variable for validation dataset. We will use the PCA features as input (X_val) and the target variable 'unused_target_label' as output (Y_val) for validating our baseline model.
X_val = val_data.drop(target_variable, axis=1)
Y_val = val_data[target_variable]

# Check shape of X_val and Y_val dataset.
# Observation 1: X_val has a shape of (33349, 422).
# Observation 2: Y_val has a shape of (33349,).
print(X_val.shape)
print(Y_val.shape)

# Similarly, we will separate features and target variable for test dataset. We will use the PCA features as input (X_test) and the target variable 'unused_target_label' as output (Y_test) for testing our baseline model.
X_test = test_data.drop(target_variable, axis=1)
Y_test = test_data[target_variable]

# Check shape of X_test and Y_test dataset.
# Observation 1: X_test has a shape of (15765, 422).
# Observation 2: Y_test has a shape of (15765,).
print(X_test.shape)
print(Y_test.shape)


# PCA Pipeline for training, testing and validation dataset.

# Scaling: Standardize the features to have a mean of 0 and a standard deviation of 1, which is important for PCA to work effectively since it is sensitive to the scale of the features.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Check shape of scaled datasets. Observation: The shapes of the scaled datasets are as follows:
# X_train_scaled has a shape of (70090, 422).
# X_val_scaled has a shape of (33349, 422).
# X_test_scaled has a shape of (15765, 422).
print(f' X_train_scaled shape: {X_train_scaled.shape} X_val_scaled shape: {X_val_scaled.shape} X_test_scaled shape: {X_test_scaled.shape}')


# Apply PCA to select top principal components that explain 95% of the variance in the dataset.
# We will fit PCA on the training data and then apply the same transformation to the validation and test datasets to ensure that they are transformed in the same way as the training data.
pca = PCA(n_components=0.95) # Retain 95% of variance
pca.fit(X_train_scaled)
X_train_pca = pca.transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Check number of components retained after applying PCA.

print(f"Number of PCA components retained: {pca.n_components_}")
# X_train_pca, X_val_pca and X_test_pca are the transformed datasets after applying PCA, and their shapes indicate the number of samples and the number of principal components retained. 
# Observation 1: X_train_pca has a shape of (70090, 115).
# Observation 2: X_val_pca has a shape of (33349, 115).
# Observation 3: X_test_pca has a shape of (15765, 115).
# Conclusion: After applying PCA, we have reduced the number of features from 422 to 115 while retaining 95% of the variance in the data. This dimensionality reduction can help improve the performance of our machine learning models by eliminating redundant and less informative features, and it can also reduce the risk of overfitting.
print(f' X_train shape: {X_train_pca.shape} X_val shape: {X_val_pca.shape} X_test shape: {X_test_pca.shape}')

###########################---------Baseline Model Training, Validating and Testing using PCA features----------###########################
# Baseline model training, validating and testing using PCA features.
# Step 1: we will train a simple Linear Regression model using the PCA features as input and the target variable 'unused_target_label' as output. 
# Step 2: We will evaluate the model's performance on the validation and test datasets. The baseline model will serve as a reference point for comparing the performance with Deep Neural Network model.

# Initialise Linear Regression model. Linear Regression is a simple and widely used algorithm for regression tasks that models the relationship between the input features and the target variable by fitting a linear equation to the observed data. 
linear_model = LinearRegression()
# Train linear regression model using X_train_pca and Y_train dataset.
linear_model.fit(X_train_pca, Y_train)

# Predict target variable for validation dataset using the trained linear regression model.
y_val_pred = linear_model.predict(X_val_pca)

# Evaluate the performance of the linear regression model on the validation dataset using Mean Squared Error (MSE) and R-squared (R2) metrics. 
# MSE measures the average squared difference between the predicted and actual values, while R2 indicates the proportion of variance in the target variable that is explained by the model.
mse_val = mean_squared_error(Y_val, y_val_pred)
r2_val = r2_score(Y_val, y_val_pred)
print(f"Validation MSE: {mse_val}")
print(f"Validation R2 Score: {r2_val}")

# Predict target variable for test dataset using the trained linear regression model.
y_test_pred = linear_model.predict(X_test_pca)
# Evaluate the performance of the linear regression model on the test dataset using Mean Squared Error (MSE) and R-squared (R2) metrics.
mse_test = mean_squared_error(Y_test, y_test_pred)
r2_test = r2_score(Y_test, y_test_pred)
print(f"Test MSE: {mse_test}")
print(f"Test R2 Score: {r2_test}")


# Simple Mean base line model for comparison. We will compute the mean of the target variable in the training dataset and use it as a constant prediction for all samples in the validation and test datasets. This mean baseline model serves as a reference point to evaluate the performance of our linear regression model, and it helps us understand whether our model is performing better than a simple average-based approach.
mean_baseline = Y_train.mean()
y_val_baseline = [mean_baseline] * len(Y_val)
y_test_baseline = [mean_baseline] * len(Y_test)
mse_val_baseline = mean_squared_error(Y_val, y_val_baseline)
r2_val_baseline = r2_score(Y_val, y_val_baseline)
mse_test_baseline = mean_squared_error(Y_test, y_test_baseline)
r2_test_baseline = r2_score(Y_test, y_test_baseline)
print(f"Mean Baseline Validation MSE: {mse_val_baseline}")
print(f"Mean Baseline Validation R2 Score: {r2_val_baseline}")
print(f"Mean Baseline Test MSE: {mse_test_baseline}")
print(f"Mean Baseline Test R2 Score: {r2_test_baseline}")


# Observations and Conclusion:
# Observation 1: Test MSE: 0.1363860450256454 and Test R2 Score: -5.816821306880774
# Observation 2: Validation MSE: 0.0822741830332232 and Validation R2 Score: -0.056949831192588896
# Observation 3: Mean Baseline Test MSE: 0.0576818623015181 and Mean Baseline Test R2 Score: -1.883043847217761
# Observation 4: Mean Baseline Validation MSE: 0.13053288407960736 and Mean Baseline Validation R2 Score: -0.6769137620887751
# Conclusion: 
# Point 1: The linear regression model has a relatively high MSE and a negative R2 score on both the validation and test datasets, which indicates that the model is not performing well in capturing the relationship between the PCA features and the target variable 'unused_target_label'.
# Point 2: The negative R2 score suggests that the model is performing worse than a simple mean-based model, which indicates that there may be a need for more complex models or additional feature engineering to improve performance.



###########################---------Baseline Model Training, Validating and Testing with all features----------###########################


# 1. We will train a simple Linear Regression model using all the features (without PCA) as input and the target variable 'unused_target_label' as output. 
# 2. We will evaluate the model's performance on the validation and test datasets to assess its generalisation ability. 
# 3. The baseline model with all features will serve as a reference point for comparing the performance with the PCA-based model and Deep Neural Network model.

# Initialise Linear Regression model.
linear_model_all_features = LinearRegression()
# Train linear regression model using all features from X_train and Y_train dataset.

# Before training model with all features, we will check the shape of X_train dataset to ensure that it contains the expected number of samples and features. This is important to confirm that we are using the correct dataset for training and that there are no issues with the data loading or preprocessing steps.
# Observation: The shape of X_train is (70090, 422) which is expected since train sample has 70,090 samples and 422 features (after dropping the target variable).
print(X_train.shape)

# Similarly, we will check the shape of X_val dataset to ensure that it contains the expected number of samples for the target variable. This is important to confirm that we are using the correct target variable for training and that there are no issues with the data loading or preprocessing steps.
# Observation: The shape of X_val is (33349, 422) which is expected since validation sample has 33,349 samples and 422 features (after dropping the target variable).
print(X_val.shape)

# Similarly, we will check the shape of X_test dataset to ensure that it contains the expected number of samples for the target variable. This is important to confirm that we are using the correct target variable for training and that there are no issues with the data loading or preprocessing steps.
# Observation: The shape of X_test is (15765, 422) which is expected since test sample has 15,765 samples and 422 features (after dropping the target variable).
print(X_test.shape)

linear_model_all_features.fit(X_train, Y_train)
# Predict target variable for validation dataset using the trained linear regression model with all features.

y_val_pred_all_features = linear_model_all_features.predict(X_val)
# Evaluate the performance of the linear regression model with all features on the validation dataset using Mean Squared Error (MSE) and R-squared (R2) metrics.
mse_val_all_features = mean_squared_error(Y_val, y_val_pred_all_features)
r2_val_all_features = r2_score(Y_val, y_val_pred_all_features)
print(f"Validation MSE with all features: {mse_val_all_features}")
print(f"Validation R2 Score with all features: {r2_val_all_features}")
# Predict target variable for test dataset using the trained linear regression model with all features.
y_test_pred_all_features = linear_model_all_features.predict(X_test)
# Evaluate the performance of the linear regression model with all features on the test dataset using Mean Squared Error (MSE) and R-squared (R2) metrics.
mse_test_all_features = mean_squared_error(Y_test, y_test_pred_all_features)
r2_test_all_features = r2_score(Y_test, y_test_pred_all_features)
print(f"Test MSE with all features: {mse_test_all_features}")
print(f"Test R2 Score with all features: {r2_test_all_features}")

# Observations and Conclusion:
# Observation 1: Validation MSE with all features: 0.08689797661020171 and Validation R2 Score with all features: -0.11635021246022603
# Observation 2: Test MSE with all features: 0.17075342379945194 and Test R2 Score with all features: -7.534565082227248
# Conclusion:
# Point 1: The linear regression model with all features has a relatively high MSE and a negative R2 score on both the validation and test datasets, which indicates that the model is not performing well in capturing the relationship between the features and the target variable 'unused_target_label'.
# Point 2: The performance of the linear regression model with all features is worse than the PCA-based model, which suggests that the PCA transformation has helped in improving the model's performance by reducing dimensionality and eliminating redundant and less informative features. This highlights the importance of feature selection and dimensionality reduction techniques in improving the performance of machine learning models, especially when dealing with high-dimensional datasets.



###########################---------Baseline Model Training, Validating and Testing with all features----------###########################
