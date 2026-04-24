import os
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.metrics import root_mean_squared_error

# ============================================================================
# 1. LOAD DATA
# ============================================================================
train_data = pd.read_csv('visual_train_features.csv')
test_data  = pd.read_csv('visual_test_features.csv')

# ============================================================================
# 2. IDENTIFY TARGET & FEATURES
# ============================================================================
target_variable = [col for col in train_data.columns 
                   if 'target' in col.lower() or 'label' in col.lower()][0]

feature_cols = [c for c in train_data.columns if c != target_variable]

# ============================================================================
# 3. EXTRACT DATA
# ============================================================================
train_x = train_data[feature_cols].values.astype(np.float32)
test_x  = test_data[feature_cols].values.astype(np.float32)

y_train = train_data[target_variable].values.astype(np.float32)
y_test  = test_data[target_variable].values.astype(np.float32)

# ============================================================================
# 4. SCALE DATA (IMPORTANT: USE TRAIN STATS)
# ============================================================================
train_x_mean = np.mean(train_x, axis=0)
train_x_std  = np.std(train_x, axis=0)

# Avoid division by zero
train_x_std[train_x_std == 0] = 1.0

# Scale test data using TRAIN statistics
test_x_scaled = (test_x - train_x_mean) / train_x_std

# ============================================================================
# 5. LOAD AND EVALUATE DEEP NEURAL NETWORK
# ============================================================================

model_path = "models/mlp_deep_neural_network_mode.h5"

print("\n--- Deep Neural Network Evaluation on Test Data ---\n")

if not os.path.exists(model_path):
    print(f"Model file not found -> {model_path}")
else:
    # Load model
    model = keras.models.load_model(model_path, compile=False)

    # Predict
    y_pred = model.predict(test_x_scaled).ravel()

    # Evaluate
    rmse = root_mean_squared_error(y_test, y_pred)

    print(f"Deep Neural Network Test RMSE: {rmse:.6f}")