import os
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.metrics import root_mean_squared_error

# ============================================================================
# 1. LOAD DATA
# ============================================================================
train_data = pd.read_csv('visual_train_features.csv')
val_data   = pd.read_csv('visual_val_features.csv')
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

# avoid division by zero
train_x_std[train_x_std == 0] = 1.0

# scale test data using TRAIN stats
test_x_scaled = (test_x - train_x_mean) / train_x_std

# ============================================================================
# 5. LOAD AND EVALUATE MODELS
# ============================================================================

models = {
    "Deep Neural Network": "models/mlp_deep_neural_network_mode.h5",
    "Simple Neural Network": "models/mlp_simple_neural_network_mode.h5"
}

print("\n--- Model Evaluation on Test Data ---\n")

for model_name, model_path in models.items():
    
    if not os.path.exists(model_path):
        print(f"{model_name}: Model file not found -> {model_path}")
        continue

    # Load model
    model = model = keras.models.load_model(model_path, compile=False)

    # Predict
    y_pred = model.predict(test_x_scaled).ravel()

    # Evaluate
    rmse = root_mean_squared_error(y_test, y_pred)

    print(f"{model_name} Test RMSE: {rmse:.6f}")

