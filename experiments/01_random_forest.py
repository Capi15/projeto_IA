"""
================================================================================
 EXPERIMENTO 01: RANDOM FOREST REGRESSOR
================================================================================
 Descrição: Modelo de ensemble baseado em árvores de decisão.
            Cada árvore vota e a média das previsões é o resultado final.

 Vantagens:
   - Não precisa de normalização dos dados
   - Lida bem com features categóricas (após encoding)
   - Menos propenso a overfitting que uma única árvore
   - Fornece importância das features nativamente

 Desvantagens:
   - Mais lento que redes neuronais para datasets muito grandes
   - Pode não capturar relações muito complexas

 Hipótese: Random Forest pode ter melhor performance que Deep Learning
           em datasets tabulares com muitas features categóricas.
================================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
import matplotlib.pyplot as plt
import joblib
import warnings
import time
warnings.filterwarnings('ignore')

print("=" * 70)
print(" EXPERIMENTO 01: RANDOM FOREST REGRESSOR")
print(" Dataset: 100k linhas | Sem Data Leakage")
print("=" * 70)

# ============================================================================
# 1. CARREGAR DADOS
# ============================================================================
print("\n[1/6] Carregando dados...")
df = pd.read_excel("ECF_2.xlsx")
print(f"   → Dataset: {df.shape[0]:,} linhas × {df.shape[1]} colunas")

# ============================================================================
# 2. CONFIGURAR TARGET E REMOVER LEAKAGE
# ============================================================================
print("\n[2/6] Removendo data leakage...")

target = "annual_medical_cost"

cols_to_drop = [
    target, "person_id",
    "risk_score", "is_high_risk",
    "annual_premium", "monthly_premium", "copay",
    "claims_count", "avg_claim_amount", "total_claims_paid",
    "policy_changes_last_2yrs", "provider_quality",
]

X = df.drop(columns=cols_to_drop, errors='ignore')
y = df[target].astype(float)

print(f"   → Features: {X.shape[1]}")

# ============================================================================
# 3. FEATURE ENGINEERING (mesmo do Deep Learning)
# ============================================================================
print("\n[3/6] Feature Engineering...")

smoker_map = {'Current': 1.0, 'Former': 0.5, 'Never': 0.0}
if X['smoker'].dtype == 'object':
    X['smoker_num'] = X['smoker'].map(smoker_map).fillna(0)
else:
    X['smoker_num'] = X['smoker']

X['risk_proxy'] = (
    0.30 * (X['chronic_count'] / max(X['chronic_count'].max(), 1)) +
    0.25 * (X['age'] / 100) +
    0.20 * (X['bmi'] / 50).clip(0, 1) +
    0.15 * X['smoker_num'] +
    0.10 * (X['medication_count'] / max(X['medication_count'].max(), 1))
).clip(0, 1)

X['utilization_score'] = (
    0.6 * np.log1p(X['hospitalizations_last_3yrs']) +
    0.4 * np.log1p(X['visits_last_year'])
)

X['procedure_intensity'] = (
    X['proc_surgery_count'] * 3 +
    X['proc_imaging_count'] * 1.5 +
    X['proc_lab_count'] +
    X['proc_consult_count'] +
    X['proc_physio_count']
)

X['hospital_severity'] = (
    X['days_hospitalized_last_3yrs'] / 
    X['hospitalizations_last_3yrs'].replace(0, 1)
)

X['age_chronic_interaction'] = X['age'] * X['chronic_count']
X['bmi_chronic_interaction'] = X['bmi'] * X['chronic_count']

severe_conditions = ['cardiovascular_disease', 'diabetes', 'cancer_history', 
                     'kidney_disease', 'copd']
X['severe_comorbidity_count'] = X[severe_conditions].sum(axis=1)

X = X.drop(columns=['smoker_num'], errors='ignore')

print(f"   → Features após engineering: {X.shape[1]}")

# ============================================================================
# 4. PREPARAR DADOS
# ============================================================================
print("\n[4/6] Preparando dados...")

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

X_train_orig, X_test_orig, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Random Forest não precisa de normalização, mas precisa de encoding
preprocess = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("numeric", "passthrough", num_cols)  # Sem StandardScaler!
    ]
)

preprocess.fit(X_train_orig)
X_train = preprocess.transform(X_train_orig)
X_test = preprocess.transform(X_test_orig)

print(f"   → Treino: {X_train.shape[0]:,} | Teste: {X_test.shape[0]:,}")
print(f"   → Features após encoding: {X_train.shape[1]}")

# ============================================================================
# 5. TREINAR RANDOM FOREST
# ============================================================================
print("\n[5/6] Treinando Random Forest...")
print("   (Isto pode demorar alguns minutos...)")

start_time = time.time()

# Configuração do Random Forest
# n_estimators: Número de árvores (mais = melhor, mas mais lento)
# max_depth: Profundidade máxima (None = sem limite, pode causar overfitting)
# min_samples_split: Mínimo de amostras para dividir um nó
# n_jobs: -1 usa todos os cores do CPU
model = RandomForestRegressor(
    n_estimators=200,       # 200 árvores
    max_depth=20,           # Limitar profundidade para evitar overfitting
    min_samples_split=10,   # Mínimo 10 amostras para split
    min_samples_leaf=5,     # Mínimo 5 amostras por folha
    n_jobs=-1,              # Usar todos os cores
    random_state=42,
    verbose=1
)

model.fit(X_train, y_train)

train_time = time.time() - start_time
print(f"\n   → Tempo de treino: {train_time:.1f} segundos")

# ============================================================================
# 6. AVALIAÇÃO
# ============================================================================
print("\n[6/6] Avaliando modelo...")

preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)
medae = median_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)
wmape = (np.sum(np.abs(y_test - preds)) / np.sum(y_test)) * 100

print(f"\n{'='*65}")
print(f" RESULTADOS - RANDOM FOREST")
print(f"{'='*65}")
print(f"\n--- Métricas Absolutas ---")
print(f"{'MAE':<20} | ${mae:,.2f}")
print(f"{'RMSE':<20} | ${rmse:,.2f}")
print(f"{'MedAE':<20} | ${medae:,.2f}")
print(f"\n--- Métricas Relativas ---")
print(f"{'WMAPE':<20} | {wmape:.2f}%")
print(f"{'R² Score':<20} | {r2:.4f} ({r2*100:.1f}%)")
print(f"{'='*65}")

# ============================================================================
# 7. FEATURE IMPORTANCE
# ============================================================================
print("\n" + "=" * 70)
print(" TOP 20 FEATURES MAIS IMPORTANTES")
print("=" * 70)

# Obter nomes das features após encoding
ohe = preprocess.named_transformers_["categorical"]
cat_feature_names = list(ohe.get_feature_names_out(cat_cols))
all_feature_names = cat_feature_names + num_cols

# Criar DataFrame de importância
importance_df = pd.DataFrame({
    'feature': all_feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(importance_df.head(20).to_string(index=False))

# ============================================================================
# 8. GUARDAR RESULTADOS
# ============================================================================
print("\n" + "=" * 70)
print(" GUARDANDO MODELO")
print("=" * 70)

joblib.dump(model, "experiments/01_random_forest_model.pkl")
joblib.dump(preprocess, "experiments/01_random_forest_preprocess.pkl")
importance_df.to_csv("experiments/01_random_forest_importance.csv", index=False)

# Guardar métricas para comparação
metrics = {
    'model': 'Random Forest',
    'r2': r2,
    'mae': mae,
    'rmse': rmse,
    'wmape': wmape,
    'train_time': train_time
}
joblib.dump(metrics, "experiments/01_random_forest_metrics.pkl")

print("   → Modelo guardado: 01_random_forest_model.pkl")
print("   → Métricas guardadas: 01_random_forest_metrics.pkl")

print(f"\n{'='*65}")
print(f" EXPERIMENTO 01 CONCLUÍDO")
print(f" R² = {r2:.4f} | MAE = ${mae:,.2f} | WMAPE = {wmape:.2f}%")
print(f"{'='*65}")
