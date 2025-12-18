import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import joblib
import shap

# ===========================
# Load dataset
# ===========================
df = pd.read_excel("ECF_2.xlsx")

target = "risk_score"
X = df.drop(columns=[target, "person_id"])
y = df[target].astype(float)

# Identify column types
cat_cols = X.select_dtypes(include=["object"]).columns
num_cols = X.select_dtypes(exclude=["object"]).columns

print("Categorical columns:", list(cat_cols))
print("Numeric columns:", list(num_cols))

# ===========================
# Preprocessing (Sklearn)
# ===========================
preprocess = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("numeric", StandardScaler(), num_cols)
    ]
)

# Fit + transform data
X_processed = preprocess.fit_transform(X)
X_processed = np.array(X_processed, dtype="float32")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# Convert to float32 (TensorFlow requirement)
X_train = np.array(X_train, dtype="float32")
X_test = np.array(X_test, dtype="float32")

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


# ===========================
# Neural Network Architecture
# ===========================
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(1, activation='linear')   # regression output
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

model.summary()

# ===========================
# Training with early stopping
# ===========================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# ===========================
# Evaluate model
# ===========================
preds = model.predict(X_test).flatten()
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print("\n===========================")
print("       MODEL RESULTS        ")
print("===========================")
print("MAE :", mae)
print("RMSE:", rmse)
print("R²  :", r2)


# ==========================================================
# TRAINING CURVES
# ==========================================================
plt.figure(figsize=(10,5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid()
plt.show()

# ==========================================================
# SHAP VALUES
# ==========================================================
# explainer = shap.KernelExplainer(
#     model,
#     X_test[np.random.choice(X_test.shape[0], 50, replace=False)]
# )

# X_to_explain = X_test[:500]
# shap_values = explainer.shap_values(X_to_explain)

# print(shap_values)
# print(shap_values[0])
# print(X_to_explain)
# print(X_to_explain[0])

# shap.summary_plot(shap_values, X_to_explain)


# ==========================================================
# FEATURE IMPORTANCE FROM FIRST LAYER
# ==========================================================
print("\n==============================")
print(" EXTRACTING FEATURE IMPORTANCE")
print("==============================\n")

# Get encoded feature names
ohe = preprocess.named_transformers_["categorical"]
ohe_features = list(ohe.get_feature_names_out(cat_cols))
all_features = ohe_features + list(num_cols)

print(f"Total encoded features: {len(all_features)}")

# Get first layer weights
first_layer = model.layers[0]
kernel, bias = first_layer.get_weights()

# Importance per encoded feature
encoded_importance = {}

for i, feat in enumerate(all_features):
    w = kernel[i]
    encoded_importance[feat] = np.sum(np.abs(w))

encoded_df = pd.DataFrame({
    "encoded_feature": list(encoded_importance.keys()),
    "importance": list(encoded_importance.values())
}).sort_values(by="importance", ascending=False)

print("\n===== TOP 20 ENCODED FEATURE IMPORTANCES =====\n")
print(encoded_df.head(20))


# ==========================================================
# AGGREGATE IMPORTANCE BACK TO ORIGINAL COLUMNS
# ==========================================================
original_importance = {}

# numeric → direct mapping
for col in num_cols:
    original_importance[col] = encoded_importance[col]

# categorical → sum all one-hot columns
for col in cat_cols:
    col_features = [f for f in encoded_importance if f.startswith(col + "_")]
    original_importance[col] = sum(encoded_importance[f] for f in col_features)

original_df = pd.DataFrame({
    "original_feature": list(original_importance.keys()),
    "importance": list(original_importance.values())
}).sort_values(by="importance", ascending=False)

print("\n===== TRUE FEATURE IMPORTANCE (ORIGINAL COLUMNS) =====\n")
print(original_df)


# ==========================================================
# REAL vs PREDICTED TABLE
# ==========================================================
results_df = pd.DataFrame({
    "Real": y_test.values,
    "Predicted": preds,
    "Absolute_Error": np.abs(y_test.values - preds),
})

print("\nFirst 20 predictions:\n")
print(results_df.head(20))


# ==========================================================
# SAVE MODEL
# ==========================================================
model.save("risk_score_model.keras")
joblib.dump(preprocess, "preprocess.pkl")

original_df.to_csv("feature_importance_original.csv", index=False)
encoded_df.to_csv("feature_importance_encoded.csv", index=False)

print("\nSaved model and preprocessing pipeline.")
print("Saved feature importance CSV files.")

