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


# ==========================================================
# 1. LOAD DATA
# ==========================================================
df = pd.read_excel("ECF_2 copy.xlsx")

TARGET = "risk_score"

X = df.drop(columns=[TARGET, "person_id"], errors="ignore")
y = df[TARGET].astype(float)

cat_cols = X.select_dtypes(include=["object"]).columns
num_cols = X.select_dtypes(exclude=["object"]).columns

print("Categorical columns:", list(cat_cols))
print("Numeric columns:", list(num_cols))


# ==========================================================
# 2. PREPROCESSING PIPELINE
# ==========================================================
preprocess = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("numeric", StandardScaler(), num_cols),
    ]
)

X_processed = preprocess.fit_transform(X)
X_processed = np.array(X_processed, dtype="float32")

X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


# ==========================================================
# 3. BUILD MODEL (Deep Learning MLP)
# ==========================================================
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)

model.summary()


# ==========================================================
# 4. TRAIN MODEL
# ==========================================================
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


# ==========================================================
# 5. EVALUATE MODEL
# ==========================================================
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
# 6. PLOT TRAINING CURVES
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
# 7. FEATURE IMPORTANCE FROM FIRST LAYER
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
# 8. AGGREGATE IMPORTANCE BACK TO ORIGINAL COLUMNS
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
# 9. SAVE ARTIFACTS
# ==========================================================
model.save("risk_score_model.keras")
joblib.dump(preprocess, "preprocess.pkl")

original_df.to_csv("feature_importance_original.csv", index=False)
encoded_df.to_csv("feature_importance_encoded.csv", index=False)

print("\nSaved model and preprocessing pipeline.")
print("Saved feature importance CSV files.")
