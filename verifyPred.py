import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# ==========================================
# 1. CONFIGURAÇÕES
# ==========================================
FOLDER = "models/v1/"
MODEL_FILE = FOLDER + "risk_score_model.keras"
PREPROCESS_FILE = FOLDER + "preprocess.pkl"
DATA_FILE = "ECF_2.xlsx"  # O teu ficheiro Excel

# ==========================================
# 2. CARREGAR DADOS E MODELOS
# ==========================================
print("A carregar modelo e dados...")

# Carregar o Excel
df = pd.read_excel(DATA_FILE)

# Selecionar apenas as primeiras 20 linhas
df_subset = df.head(20).copy()

# Carregar o Modelo treinado e o Pipeline de pré-processamento
try:
    model = tf.keras.models.load_model(MODEL_FILE)
    preprocess = joblib.load(PREPROCESS_FILE)
except FileNotFoundError as e:
    print(f"\nERRO: Não encontrei os ficheiros necessários. ({e})")
    print("Certifica-te que 'risk_score_model.keras' e 'preprocess.pkl' estão na pasta.")
    exit()

# ==========================================
# 3. PREPARAR OS DADOS (IGUAL AO TREINO)
# ==========================================
# Definir as colunas a remover (devem ser AS MESMAS que removeste no treino)
target_col = "risk_score"
cols_to_drop = ["risk_score", "person_id", "is_high_risk"]

X_raw = df_subset.drop(columns=cols_to_drop, errors='ignore')

# Transformar os dados usando o "preprocess" carregado
# Isto garante que a normalização é idêntica à do treino
X_processed = preprocess.transform(X_raw)
X_processed = np.array(X_processed, dtype="float32")

# ==========================================
# 4. FAZER PREVISÕES
# ==========================================
print("A calcular previsões para as primeiras 10 pessoas...")
predictions = model.predict(X_processed).flatten()

# ==========================================
# 5. COMPARAR RESULTADOS
# ==========================================
# Criar uma tabela final para comparação
results = pd.DataFrame({
    "Person_ID": df_subset["person_id"],
    "Real_Score": df_subset["risk_score"],
    "Pred_Score": predictions,
    "Erro_Absoluto": np.abs(df_subset["risk_score"] - predictions),
    "Real_High_Risk_Flag": df_subset.get("is_high_risk", "N/A") 
})

# Formatação para leitura fácil
pd.options.display.float_format = '{:.4f}'.format

print("\n=======================================================")
print("   COMPARAÇÃO: REAL vs PREVISTO (Top 10)")
print("=======================================================")
print(results)

# Pequena análise rápida
avg_error = results["Erro_Absoluto"].mean()
print(f"\nErro Médio nestas 10 linhas: {avg_error:.4f}")



# -------------------------------------------------------------------
# ==========================================
# 1. CONFIGURAÇÕES
# ==========================================
FOLDER = "models/v2/"
MODEL_FILE = FOLDER + "risk_score_model.keras"
PREPROCESS_FILE = FOLDER + "preprocess.pkl"
DATA_FILE = "ECF_2.xlsx"  # O teu ficheiro Excel

# ==========================================
# 2. CARREGAR DADOS E MODELOS
# ==========================================
print("A carregar modelo e dados...")

# Carregar o Excel
df = pd.read_excel(DATA_FILE)

# Selecionar apenas as primeiras 10 linhas
df_subset = df.head(20).copy()

# Carregar o Modelo treinado e o Pipeline de pré-processamento
try:
    model = tf.keras.models.load_model(MODEL_FILE)
    preprocess = joblib.load(PREPROCESS_FILE)
except FileNotFoundError as e:
    print(f"\nERRO: Não encontrei os ficheiros necessários. ({e})")
    print("Certifica-te que 'risk_score_model.keras' e 'preprocess.pkl' estão na pasta.")
    exit()

# ==========================================
# 3. PREPARAR OS DADOS (IGUAL AO TREINO)
# ==========================================
# Definir as colunas a remover (devem ser AS MESMAS que removeste no treino)
target_col = "risk_score"
cols_to_drop = ["risk_score", "person_id", "is_high_risk", "monthly_premium", "avg_claim_amount", "mental_health"]

X_raw = df_subset.drop(columns=cols_to_drop, errors='ignore')

# Transformar os dados usando o "preprocess" carregado
# Isto garante que a normalização é idêntica à do treino
X_processed = preprocess.transform(X_raw)
X_processed = np.array(X_processed, dtype="float32")

# ==========================================
# 4. FAZER PREVISÕES
# ==========================================
print("A calcular previsões para as primeiras 10 pessoas...")
predictions = model.predict(X_processed).flatten()

# ==========================================
# 5. COMPARAR RESULTADOS
# ==========================================
# Criar uma tabela final para comparação
results = pd.DataFrame({
    "Person_ID": df_subset["person_id"],
    "Real_Score": df_subset["risk_score"],
    "Pred_Score": predictions,
    "Erro_Absoluto": np.abs(df_subset["risk_score"] - predictions),
    "Real_High_Risk_Flag": df_subset.get("is_high_risk", "N/A") 
})

# Formatação para leitura fácil
pd.options.display.float_format = '{:.4f}'.format

print("\n=======================================================")
print("   COMPARAÇÃO: REAL vs PREVISTO (Top 10)")
print("=======================================================")
print(results)

# Pequena análise rápida
avg_error = results["Erro_Absoluto"].mean()
print(f"\nErro Médio nestas 10 linhas: {avg_error:.4f}")


# -------------------------------------------------------------------
# ==========================================
# 1. CONFIGURAÇÕES
# ==========================================
FOLDER = "models/v3/"
MODEL_FILE = FOLDER + "risk_score_model.keras"
PREPROCESS_FILE = FOLDER + "preprocess.pkl"
DATA_FILE = "ECF_2.xlsx"  # O teu ficheiro Excel

# ==========================================
# 2. CARREGAR DADOS E MODELOS
# ==========================================
print("A carregar modelo e dados...")

# Carregar o Excel
df = pd.read_excel(DATA_FILE)

# Selecionar apenas as primeiras 10 linhas
df_subset = df.head(20).copy()

# Carregar o Modelo treinado e o Pipeline de pré-processamento
try:
    model = tf.keras.models.load_model(MODEL_FILE)
    preprocess = joblib.load(PREPROCESS_FILE)
except FileNotFoundError as e:
    print(f"\nERRO: Não encontrei os ficheiros necessários. ({e})")
    print("Certifica-te que 'risk_score_model.keras' e 'preprocess.pkl' estão na pasta.")
    exit()

# ==========================================
# 3. PREPARAR OS DADOS (IGUAL AO TREINO)
# ==========================================
# Definir as colunas a remover (devem ser AS MESMAS que removeste no treino)
target_col = "risk_score"
cols_to_drop = ["risk_score", "person_id", "is_high_risk", "monthly_premium", "avg_claim_amount", "annual_premium"]

X_raw = df_subset.drop(columns=cols_to_drop, errors='ignore')

# Transformar os dados usando o "preprocess" carregado
# Isto garante que a normalização é idêntica à do treino
X_processed = preprocess.transform(X_raw)
X_processed = np.array(X_processed, dtype="float32")

# ==========================================
# 4. FAZER PREVISÕES
# ==========================================
print("A calcular previsões para as primeiras 10 pessoas...")
predictions = model.predict(X_processed).flatten()

# ==========================================
# 5. COMPARAR RESULTADOS
# ==========================================
# Criar uma tabela final para comparação
results = pd.DataFrame({
    "Person_ID": df_subset["person_id"],
    "Real_Score": df_subset["risk_score"],
    "Pred_Score": predictions,
    "Erro_Absoluto": np.abs(df_subset["risk_score"] - predictions),
    "Real_High_Risk_Flag": df_subset.get("is_high_risk", "N/A") 
})

# Formatação para leitura fácil
pd.options.display.float_format = '{:.4f}'.format

print("\n=======================================================")
print("   COMPARAÇÃO: REAL vs PREVISTO (Top 10)")
print("=======================================================")
print(results)

# Pequena análise rápida
avg_error = results["Erro_Absoluto"].mean()
print(f"\nErro Médio nestas 10 linhas: {avg_error:.4f}")