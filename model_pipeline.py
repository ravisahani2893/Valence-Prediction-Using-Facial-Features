import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.decomposition import PCA
from sklearn.metrics import root_mean_squared_error
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras import regularizers
import numpy as np
import os



# Load training, testing and validation dataset from CSV files. 
train_data = pd.read_csv('visual_train_features.csv')
test_data = pd.read_csv('visual_test_features.csv')
val_data = pd.read_csv('visual_val_features.csv')

# ============================================================================
# 1. DATA PIPELINE
# ============================================================================

# Extract target variable column
target_variable = [column for column in train_data.columns if 'target' in column.lower() or 'label' in column.lower()][0]

# Extracting feature variable excluding target variable
feature_cols = [c for c in train_data.columns if c != target_variable]
print(f"\nTotal feature columns: {len(feature_cols)}")

# Extract train, test and validation target data
y_train      = train_data[target_variable].values.astype(np.float32)
y_validation = val_data[target_variable].values.astype(np.float32)
y_test       = test_data[target_variable].values.astype(np.float32)


# Extract train, test and validation features data
train_x      = train_data[feature_cols].values.astype(np.float32)
validation_x = val_data[feature_cols].values.astype(np.float32)
test_x       = test_data[feature_cols].values.astype(np.float32)


# Standardise features using TRAIN statistics
train_x_means = np.mean(train_x, axis=0)
train_x_stds  = np.std(train_x, axis=0)

# Guard against division by zero for any remaining zero-std columns
train_x_stds[train_x_stds == 0] = 1.0


# Feature scaling: Z-score standardisation of training, validation and testing data, 
train_x_std      = (train_x - train_x_means) / train_x_stds
validation_x_std = (validation_x - train_x_means) / train_x_stds
test_x_std       = (test_x - train_x_means) / train_x_stds

# Calculate training data mean
y_train_mean = y_train.mean()
y_train_sd   = y_train.std()

# ============================================================================
# 2. LINEAR REGRESSION (ALL FEATURES)
# ============================================================================

# Initialise LinearRegression for model training
lr_model = LinearRegression()
lr_model.fit(train_x_std, y_train)

# Make prediction on training dataset using the trained linear regression model.
y_train_pred = lr_model.predict(train_x_std)

# Make prediction on validation dataset using the trained linear regression model.
y_val_pred = lr_model.predict(validation_x_std)

# Make prediction on test dataset using the trained linear regression model.
y_test_pred = lr_model.predict(test_x_std)


# Calculate the Root Mean Squared Error (RMSE) for the training, validation and test datasets to evaluate the performance of the linear regression model. RMSE is a commonly used metric for regression tasks that measures the average magnitude of the errors between the predicted and actual values, with lower values indicating better performance.
train_rmse = root_mean_squared_error(y_train, y_train_pred)
val_rmse   = root_mean_squared_error(y_validation, y_val_pred)
test_rmse  = root_mean_squared_error(y_test, y_test_pred)
print(f"Train RMSE: {train_rmse}")
print(f"Validation RMSE: {val_rmse}")
print(f"Test RMSE: {test_rmse}")



# ============================================================================
# 3. RIDGE REGRESSION (ALL FEATURES)
# ============================================================================
# Ridge adds L2 penalty to stabilise the solution with 400+ correlated features.
# Alpha=1.0 is a moderate default; alpha=1000 would over-regularise.
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(train_x_std, y_train)

# Predictions
ridge_train_pred = ridge_model.predict(train_x_std)
ridge_val_pred   = ridge_model.predict(validation_x_std)
ridge_test_pred  = ridge_model.predict(test_x_std)

# Evaluation
ridge_train_rmse = root_mean_squared_error(y_train, ridge_train_pred)
ridge_val_rmse   = root_mean_squared_error(y_validation, ridge_val_pred)
ridge_test_rmse  = root_mean_squared_error(y_test, ridge_test_pred)

print(f"\n--- Ridge Regression (alpha=1.0, all features) ---")
print(f"Train RMSE: {ridge_train_rmse:.6f}")
print(f"Val   RMSE: {ridge_val_rmse:.6f}")
print(f"Test  RMSE: {ridge_test_rmse:.6f}")

# ============================================================================
# 4. PCA DIMENSIONALITY REDUCTION + LINEAR REGRESSION
# ============================================================================

# PCA
pca = PCA(n_components=0.95)

train_x_pca      = pca.fit_transform(train_x_std)
validation_x_pca = pca.transform(validation_x_std)
test_x_pca       = pca.transform(test_x_std)

print(f"\n--- PCA: {train_x_std.shape[1]} features -> {train_x_pca.shape[1]} components (95% variance) ---")

# Linear Regression on PCA features
lr_pca_model = LinearRegression()
lr_pca_model.fit(train_x_pca, y_train)

# Predictions
lr_pca_train_pred = lr_pca_model.predict(train_x_pca)
lr_pca_val_pred   = lr_pca_model.predict(validation_x_pca)
lr_pca_test_pred  = lr_pca_model.predict(test_x_pca)

# Evaluation
lr_pca_train_rmse = root_mean_squared_error(y_train, lr_pca_train_pred)
lr_pca_val_rmse   = root_mean_squared_error(y_validation, lr_pca_val_pred)
lr_pca_test_rmse  = root_mean_squared_error(y_test, lr_pca_test_pred)

print(f"Train RMSE: {lr_pca_train_rmse:.6f}")
print(f"Val   RMSE: {lr_pca_val_rmse:.6f}")
print(f"Test  RMSE: {lr_pca_test_rmse:.6f}")



# ============================================================================
# 5. MLP EXPERIMENT 1 - DEEPER NETWORK WITH DROPOUT (ALL FEATURES)
# ============================================================================

# Architecture: 256 -> 128 -> 64 -> 32 -> 1
# Motivation: Experiment 1 used a small network with weak regularisation.
# This experiment increases capacity (256-unit first layer, 4 hidden layers)
# to capture more complex nonlinear relationships between facial action units,
# head pose, and gaze features. Dropout (0.3) is added after the first layer
# to control the increased overfitting risk from the larger parameter count.
# The deeper architecture allows hierarchical feature learning: early layers
# can learn basic correlations within feature families (e.g., AU combinations),
# while later layers combine these into higher-level affect patterns.
model_2 = Sequential([
    keras.layers.Dense(256, activation='relu', kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.L2(1e-5)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.L2(1e-5)),
    keras.layers.Dense(64, activation='relu', kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.L2(1e-5)),
    keras.layers.Dense(32, activation='relu', kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.L2(1e-5)),
    keras.layers.Dense(1)
])


# Compile Model - Using Mean Squared Error (MSE) as loss and Adam optimiser
model_2.compile(
    loss='mse',
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=['mae']
)
model_2.summary()


# Callbacks: Stop training if validation loss does not improve
# Reduce learning rate if validation loss plateaus
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
]
# ----------------------------------------------------------------------------
# Model Training
# ----------------------------------------------------------------------------
print('\n--- Training MLP Experiment 2 (deeper network) ---')

history_2 = model_2.fit(
    train_x_std, y_train,
    epochs=500,
    batch_size=128,
    validation_data=(validation_x_std, y_validation),
    callbacks=callbacks,
    verbose=1
)
training_epochs = len(history_2.history['loss'])
print(f"Total training epoch {training_epochs}")

# ----------------------------------------------------------------------------
# Predictions
# ----------------------------------------------------------------------------
mlp2_train_pred = model_2.predict(train_x_std).ravel()
mlp2_val_pred   = model_2.predict(validation_x_std).ravel()
mlp2_test_pred  = model_2.predict(test_x_std).ravel()

# ----------------------------------------------------------------------------
# Evaluation using RMSE
# ----------------------------------------------------------------------------

mlp2_train_rmse = root_mean_squared_error(y_train, mlp2_train_pred)
mlp2_val_rmse   = root_mean_squared_error(y_validation, mlp2_val_pred)
mlp2_test_rmse  = root_mean_squared_error(y_test, mlp2_test_pred)

print("\n--- MLP Experiment Results ---")
print(f"Train RMSE: {mlp2_train_rmse:.6f}")
print(f"Val   RMSE: {mlp2_val_rmse:.6f}")
print(f"Test  RMSE: {mlp2_test_rmse:.6f}")

# Save full model (architecture + weights)
os.makedirs("models", exist_ok=True)
model_2.save("models/mlp_deep_neural_network_mode.h5")


# ============================================================================
# 6. MLP EXPERIMENT 2 - SMALLER NEURAL NETWORK  WITH Noise
# ============================================================================


# Model definition. 
model = Sequential([
    keras.layers.GaussianNoise(0.01),
    keras.layers.Dense(8, activation='relu', kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.L2(1e-5)),
    keras.layers.Dense(4, activation='relu', kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.L2(1e-5)),
    keras.layers.Dense(1)
])

model.compile(
    loss='mse',
    optimizer=keras.optimizers.Adam(learning_rate=0.001)
)

# Early stopping based on validation loss
callback = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

print('Training:')
history = model.fit(
    train_x_std, y_train,
    epochs=500,
    batch_size=128,
    validation_data=(validation_x_std, y_validation),
    callbacks=[callback],
    verbose=1
)

training_epochs = len(history.history['loss'])
print(f"Total training epoch {training_epochs}")

# Predictions
model_train_prediction = model.predict(train_x_std).ravel()
model_val_pred   = model.predict(validation_x_std).ravel()
model_test_pred  = model.predict(test_x_std).ravel()

# RMSE
model_train_prediction_rmse = root_mean_squared_error(y_train, model_train_prediction)
model_val_pred_rmse   = root_mean_squared_error(y_validation, model_val_pred)
model_test_pred_rmse  = root_mean_squared_error(y_test, model_test_pred)

print("\n--- MLP Experiment Results ---")
print(f"Train RMSE: {model_train_prediction_rmse:.6f}")
print(f"Val   RMSE: {model_val_pred_rmse:.6f}")
print(f"Test  RMSE: {model_test_pred_rmse:.6f}")

# Save full model (architecture + weights)
os.makedirs("models", exist_ok=True)
model.save("models/mlp_simple_neural_network_mode.h5")