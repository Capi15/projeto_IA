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
# 1. CARREGAR E LIMPAR DADOS
# ===========================
df = pd.read_excel("ECF_2.xlsx")

# Remoção de Data Leakage (Fuga de Informação)
# MOTIVO: Removemos colunas como 'monthly_premium' ou 'total_claims_paid' porque
# elas são *consequências* do risco e não *causas*. Se as mantivesses, o modelo
# iria "adivinhar" o risco baseado no quanto a pessoa pagou, o que é inútil para
# novos clientes que ainda não pagaram nada.
target = "annual_medical_cost"

# Lista de colunas a remover
cols_to_drop = [
    "risk_score", 
    "person_id", 
    # --- DATA LEAKAGE (SPOILERS) ---
    "risk_score",                # Calculado COM BASE no custo esperado
    "is_high_risk",              # Derivado do risco/custo
    "monthly_premium",           # Calculado com base no custo esperado
    "annual_premium",            # Calculado com base no custo esperado
    "avg_claim_amount",          # Derivado dos custos já incorridos
    "claims_count",              # Fortemente correlacionado com custo
    "total_claims_paid",         # É literalmente o custo já materializado
    "copay"                      # Calculado com base no risco/custo
]

# Definir X e y
# errors='ignore' garante que o código não falha se a coluna já não existir
X = df.drop(columns=cols_to_drop, errors='ignore')
y = df[target].astype(float)

# Identificar tipos de colunas automaticamente
cat_cols = X.select_dtypes(include=["object"]).columns
num_cols = X.select_dtypes(exclude=["object"]).columns

print("Categorical columns:", list(cat_cols))
print("Numeric columns:", list(num_cols))

# ===========================
# 2. DIVISÃO DOS DADOS (SPLIT)
# ===========================
# [MOTIVO IMPORTANTE] Fazemos o Split ANTES do Pré-processamento.
# Porquê? Se calculares a média (StandardScaler) com todos os dados antes de dividir,
# a média do conjunto de TESTE vai influenciar o TREINO. Isso chama-se "Data Leakage".
# O correto é o modelo não saber absolutamente NADA sobre os dados de teste.
X_train_orig, X_test_orig, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===========================
# 3. PRÉ-PROCESSAMENTO (PREPROCESSING)
# ===========================
# Definimos as regras de transformação
preprocess = ColumnTransformer(
    transformers=[
        # OneHotEncoder: Transforma texto ("Male", "Female") em números (0, 1).
        # handle_unknown="ignore": Se no futuro aparecer uma categoria nova que não existia
        # no treino, o código não rebenta, apenas ignora (essencial para produção).
        ("categorical", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        
        # StandardScaler: Coloca todos os números na mesma escala (média 0, desvio 1).
        # MOTIVO: Redes Neuronais funcionam muito mal se misturares números pequenos (Idade: 30)
        # com números grandes (Salário: 50000). A escala ajuda a rede a aprender mais rápido.
        ("numeric", StandardScaler(), num_cols)
    ]
)

# [CRÍTICO] .fit apenas no TREINO
preprocess.fit(X_train_orig) 

# Aplicar a transformação (.transform) nos dois
X_train = preprocess.transform(X_train_orig)
X_test = preprocess.transform(X_test_orig)

# [MOTIVO TÉCNICO] Converter para float32
# O Python usa float64 por defeito (muita precisão), mas as placas gráficas (GPU) e o TensorFlow
# preferem float32. Ocupa metade da memória e é mais rápido, sem perder qualidade relevante.
X_train = np.array(X_train, dtype="float32")
X_test = np.array(X_test, dtype="float32")

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


# ===========================
# 4. ARQUITETURA DA REDE NEURONAL
# ===========================

# Arquitetura em "Funil": Começa larga (256) e vai estreitando (128 -> 64 -> 1).
# Isso força a rede a resumir a informação e extrair apenas os padrões importantes.
model = Sequential([
    # Camada de Entrada + Primeira Oculta
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    # [MOTIVO] Dropout: Desliga aleatoriamente 30% dos neurónios a cada passagem.
    # Serve para evitar "Overfitting" (o modelo decorar os dados). Obriga a rede a ser robusta.
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    # Camada de Saída: 1 neurónio apenas.
    # Activation='linear': Como é uma REGRESSÃO (prever um valor contínuo), não queremos
    # limitar a saída (como sigmoid que limita entre 0 e 1). Queremos o valor real.
    Dense(1, activation='linear')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), # Adam é o "standard" da indústria
    loss='mse',    # Mean Squared Error: Tenta minimizar o erro ao quadrado (penaliza muito os erros grandes)
    metrics=['mae'] # Mean Absolute Error: Mais fácil para humanos lerem
)

model.summary()


# ===========================
# 5. TREINO
# ===========================

# EarlyStopping: O "fiscal" do treino.
# Se o 'val_loss' (erro na validação) não melhorar durante 20 épocas (patience), o treino para.
# restore_best_weights=True: Garante que no fim, ficamos com a melhor versão do modelo, e não a última.
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_split=0.2, # Usa 20% do treino para ir validando enquanto aprende
    epochs=200,
    batch_size=32, # Atualiza os pesos a cada 32 linhas de dados
    callbacks=[early_stop],
    verbose=1
)

# ===========================
# 6. AVALIAÇÃO DO MODELO
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

# [MOTIVO] WMAPE vs MAPE
# O MAPE tradicional explode se tiveres zeros nos dados (Divisão por zero).
# O WMAPE (Weighted MAPE) resolve isso somando todos os erros e dividindo pela soma total dos valores.
# É a forma mais segura de calcular o "Erro Percentual" em dados financeiros ou de saúde.
sum_abs_error = np.sum(np.abs(y_test - preds))
sum_actual = np.sum(y_test)
wmape = (sum_abs_error / sum_actual) * 100

print(f"{'Métrica':<25} | {'Valor':<10} | {'Interpretação'}")
print("-" * 65)
print(f"{'MAE (Erro Médio Abs)':<25} | ${mae:.2f}     | Erras em média ${mae:.2f} no custo anual")
print(f"{'RMSE (Erro Quadrático)':<25} | ${rmse:.2f}     | Penaliza erros grandes (ex: casos muito caros)")
print(f"{'MedAE (Erro Mediano)':<25} | ${medae:.2f}     | Erro típico (ignora outliers extremos)")
print(f"{'CV-RMSE (% da Média)':<25} | {cv_rmse*100:.2f}%     | Erro relativo à média")
print(f"{'R² Score':<25} | {r2:.4f}     | O modelo explica {r2*100:.1f}% da variância")
print("-" * 65)
print(f"{'NRMSE (% do Range)':<25} | {nrmse_range*100:.2f}%     | Erro relativo à escala min-max")
print(f"{'WMAPE (Erro Total)':<25} | {wmape:.2f}%     | Erro relativo à soma total")
print("-" * 65)

# --- GRÁFICOS DE RESÍDUOS ---
# Importante para ver se o modelo tem "vícios".
# Se o histograma não estiver centrado no zero, o modelo está "biased" (enviesado).
residuals = y_test - preds

plt.figure(figsize=(14, 5))

# Plot 1: Real vs Previsto
plt.subplot(1, 2, 1)
plt.scatter(y_test, preds, alpha=0.5, color='royalblue')
plt.plot([y_min, y_max], [y_min, y_max], 'r--', lw=2)
plt.xlabel("Valor Real (Annual Medical Cost $)")
plt.ylabel("Valor Previsto ($)")
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

# --- ANÁLISE POR GRUPO (FAIRNESS) ---
print("\n=======================================================")
print(" 2. PERFORMANCE POR GRUPO (CATEGORICAL ANALYSIS)")
print("=======================================================")
print("Isto mostra se o modelo é tendencioso (Bias Analysis)\n")
# [MOTIVO] Esta parte é crucial na saúde.
# Verifica se o modelo funciona tão bem para Homens como para Mulheres, 
# ou se falha mais em zonas Rurais. Ajuda a detetar preconceitos no algoritmo.

# Nota: Usamos X_test_orig porque X_test já está transformado em números (ninguém entende).
analysis_df = X_test_orig.copy()
analysis_df['Real'] = y_test.values
analysis_df['Predicted'] = preds
analysis_df['Abs_Error'] = np.abs(analysis_df['Real'] - analysis_df['Predicted'])

for col in cat_cols:
    if col in analysis_df.columns:
        print(f"--- Análise por: {col.upper()} ---")
        # Agrupar e calcular métricas
        group_metrics = analysis_df.groupby(col).agg(
            Count=('Real', 'count'),          # Quantos exemplos temos?
            MAE=('Abs_Error', 'mean'),        # Qual o erro médio neste grupo?
            Mean_Real=('Real', 'mean'),       # Qual o risco médio real deste grupo?
        ).sort_values(by='MAE', ascending=False)
        
        print(group_metrics)
        print("-" * 50)
        print("\n")


# ==========================================================
# CURVAS DE TREINO
# ==========================================================
# Serve para ver se houve Overfitting.
# Se a linha "Validation Loss" começar a subir enquanto a "Training Loss" desce,
# o modelo começou a decorar em vez de aprender.
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
# GUARDAR TUDO
# ==========================================================

# Guardar o modelo treinado
model.save("cost_prediction_model.keras")

# [MUITO IMPORTANTE] Guardar o preprocessador
# Sem este ficheiro 'preprocess_cost.pkl', o modelo é inútil no futuro, 
# porque não saberás como transformar os dados novos da mesma forma que o treino.
joblib.dump(preprocess, "preprocess_cost.pkl")

#original_df.to_csv("feature_importance_original.csv", index=False)
#encoded_df.to_csv("feature_importance_encoded.csv", index=False)

print("\nSaved model and preprocessing pipeline.")
print("Saved feature importance CSV files.")

