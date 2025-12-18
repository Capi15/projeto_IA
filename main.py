import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ===========================
# Load dataset
# ===========================
df = pd.read_excel("ECF_2.xlsx")

target = "risk_score"
X = df.drop(columns=[target, "person_id", "is_high_risk", "monthly_premium", "avg_claim_amount", "annual_premium"], errors='ignore')
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

# Split BEFORE preprocessing to keep original data for analysis
X_train_orig, X_test_orig, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit + transform data
preprocess.fit(X_train_orig)
X_train = preprocess.transform(X_train_orig)
X_test = preprocess.transform(X_test_orig)

# Convert to float32 (TensorFlow requirement)
X_train = np.array(X_train, dtype="float32")
X_test = np.array(X_test, dtype="float32")

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

# --- NON-NORMALIZED ---
mae = mean_absolute_error(y_test, preds)
mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)
medae = median_absolute_error(y_test, preds)

# --- NORMALIZED (Escala relativa/percentual) ---
y_max = y_test.max()
y_min = y_test.min()
y_range = y_max - y_min
y_mean = y_test.mean()

# NRMSE (Normalized RMSE): Erro como % do range total dos dados
nrmse_range = rmse / y_range 
# CV-RMSE (Coef. of Variation): Erro como % da média
cv_rmse = rmse / y_mean
# R2 (Qualidade do ajuste)
r2 = r2_score(y_test, preds)

# MAPE (Mean Absolute Percentage Error)
# Nao se pode usar por causa de valores zero
#mape = np.mean(np.abs((y_test - preds) / (y_test + 1e-10))) * 100

# WMAPE: Erro Absoluto Total / Soma Total dos Valores Reais
sum_abs_error = np.sum(np.abs(y_test - preds))
sum_actual = np.sum(y_test)

wmape = (sum_abs_error / sum_actual) * 100

print(f"{'Métrica':<25} | {'Valor':<10} | {'Interpretação'}")
print("-" * 65)
print(f"{'MAE (Erro Médio Abs)':<25} | {mae:.4f}     | Erras em média {mae:.2f} pontos no score")
print(f"{'RMSE (Erro Quadrático)':<25} | {rmse:.4f}     | Penaliza erros grandes (ex: casos graves)")
print(f"{'MedAE (Erro Mediano)':<25} | {medae:.4f}     | Erro típico (ignora outliers extremos)")
print(f"{'CV-RMSE (% da Média)':<25} | {cv_rmse*100:.2f}%     | Erro relativo à média")
print(f"{'R² Score':<25} | {r2:.4f}     | O modelo explica {r2*100:.1f}% da variância")
print("-" * 65)
print(f"{'NRMSE (% do Range)':<25} | {nrmse_range*100:.2f}%     | Erro relativo à escala min-max")
print(f"{'WMAPE (Erro Total)':<25} | {wmape:.2f}%     | Erro relativo à soma total")
print("-" * 65)

# --- ANÁLISE DE RESÍDUOS ---
residuals = y_test - preds

plt.figure(figsize=(14, 5))
    
# Plot 1: Real vs Previsto
plt.subplot(1, 2, 1)
plt.scatter(y_test, preds, alpha=0.5, color='royalblue')
plt.plot([y_min, y_max], [y_min, y_max], 'r--', lw=2) # Linha perfeita
plt.xlabel("Valor Real (Risk Score)")
plt.ylabel("Valor Previsto")
plt.title("Real vs Previsto (Ideal = Linha Vermelha)")
plt.grid(True, alpha=0.3)

# Plot 2: Distribuição dos Erros
plt.subplot(1, 2, 2)
sns.histplot(residuals, kde=True, color='purple', bins=30)
plt.axvline(0, color='r', linestyle='--')
plt.xlabel("Erro (Resíduo)")
plt.title("Distribuição dos Erros (Ideal = Centrado no 0)")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n=======================================================")
print(" 2. PERFORMANCE POR GRUPO (CATEGORICAL ANALYSIS)")
print("=======================================================")
print("Isto mostra se o modelo é tendencioso (Bias Analysis)\n")

# Juntar previsões ao dataframe original para análise
analysis_df = X_test_orig.copy()
analysis_df['Real'] = y_test.values
analysis_df['Predicted'] = preds
analysis_df['Abs_Error'] = np.abs(analysis_df['Real'] - analysis_df['Predicted'])

for col in cat_cols:
    if col in analysis_df.columns:
        print(f"--- Análise por: {col.upper()} ---")
        # Agrupar e calcular métricas
        group_metrics = analysis_df.groupby(col).agg(
            Count=('Real', 'count'),
            MAE=('Abs_Error', 'mean'),
            Mean_Real=('Real', 'mean'),
            Mean_Pred=('Predicted', 'mean')
        ).sort_values(by='MAE', ascending=False)
        
        print(group_metrics)
        print("-" * 50)
        print("\n")


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
# background_summary = shap.kmeans(X_train, 50) 

# explainer = shap.KernelExplainer(model.predict, background_summary)

# shap_values = explainer.shap_values(X_test[:100])

# ohe = preprocess.named_transformers_["categorical"]
# feature_names = list(ohe.get_feature_names_out(cat_cols)) + list(num_cols)

# if isinstance(shap_values, list):
#     shap_values = shap_values[0]

# shap.summary_plot(shap_values, X_test[:100], feature_names=feature_names)

# ==========================================================
# FEATURE IMPORTANCE FROM FIRST LAYER
# ==========================================================
# print("\n==============================")
# print(" EXTRACTING FEATURE IMPORTANCE")
# print("==============================\n")

# # Get encoded feature names
# ohe = preprocess.named_transformers_["categorical"]
# ohe_features = list(ohe.get_feature_names_out(cat_cols))
# all_features = ohe_features + list(num_cols)

# print(f"Total encoded features: {len(all_features)}")

# # Get first layer weights
# first_layer = model.layers[0]
# kernel, bias = first_layer.get_weights()

# # Importance per encoded feature
# encoded_importance = {}

# for i, feat in enumerate(all_features):
#     w = kernel[i]
#     encoded_importance[feat] = np.sum(np.abs(w))

# encoded_df = pd.DataFrame({
#     "encoded_feature": list(encoded_importance.keys()),
#     "importance": list(encoded_importance.values())
# }).sort_values(by="importance", ascending=False)

# print("\n===== TOP 20 ENCODED FEATURE IMPORTANCES =====\n")
# print(encoded_df.head(20))


# ==========================================================
# AGGREGATE IMPORTANCE BACK TO ORIGINAL COLUMNS
# ==========================================================
# original_importance = {}

# # numeric → direct mapping
# for col in num_cols:
#     original_importance[col] = encoded_importance[col]

# # categorical → sum all one-hot columns
# for col in cat_cols:
#     col_features = [f for f in encoded_importance if f.startswith(col + "_")]
#     original_importance[col] = sum(encoded_importance[f] for f in col_features)

# original_df = pd.DataFrame({
#     "original_feature": list(original_importance.keys()),
#     "importance": list(original_importance.values())
# }).sort_values(by="importance", ascending=False)

# print("\n===== TRUE FEATURE IMPORTANCE (ORIGINAL COLUMNS) =====\n")
# print(original_df)


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

#original_df.to_csv("feature_importance_original.csv", index=False)
#encoded_df.to_csv("feature_importance_encoded.csv", index=False)

print("\nSaved model and preprocessing pipeline.")
print("Saved feature importance CSV files.")

