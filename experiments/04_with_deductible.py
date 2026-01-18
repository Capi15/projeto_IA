"""
================================================================================
 EXPERIMENTO 04: TESTE COM DEDUCTIBLE
================================================================================
 Descrição: Testa se incluir 'deductible' (franquia) melhora as previsões.
            Deductible NÃO é necessariamente data leakage porque:
            - É definido ANTES do paciente usar o seguro
            - É uma característica do plano, não do comportamento

 Questão de Investigação:
   A franquia escolhida pelo paciente revela informação sobre os custos?
   - Pacientes com deductible alto podem ser mais saudáveis (não esperam usar)
   - Pacientes com deductible baixo podem antecipar custos altos

 Método: Random Forest com e sem deductible para comparação justa.
================================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
import joblib
import warnings
import time
warnings.filterwarnings('ignore')

print("=" * 70)
print(" EXPERIMENTO 04: RANDOM FOREST COM DEDUCTIBLE")
print(" Investigar se 'deductible' melhora previsões sem ser leakage")
print("=" * 70)

# ============================================================================
# 1. CARREGAR DADOS
# ============================================================================
print("\n[1/6] Carregando dados...")
df = pd.read_excel("ECF_2.xlsx")
print(f"   → Dataset: {df.shape[0]:,} linhas × {df.shape[1]} colunas")

# ============================================================================
# 2. CONFIGURAR TARGET E REMOVER LEAKAGE (MAS MANTER DEDUCTIBLE!)
# ============================================================================
print("\n[2/6] Removendo data leakage (mantendo deductible)...")

target = "annual_medical_cost"

# NOTA: Removemos as mesmas colunas de leakage, MAS mantemos 'deductible'
cols_to_drop = [
    target, "person_id",
    "risk_score", "is_high_risk",
    "annual_premium", "monthly_premium", "copay",
    "claims_count", "avg_claim_amount", "total_claims_paid",
    "policy_changes_last_2yrs", "provider_quality",
    # NÃO removemos deductible!
]

X = df.drop(columns=cols_to_drop, errors='ignore')
y = df[target].astype(float)

# Verificar se deductible existe
if 'deductible' in X.columns:
    print(f"   → ✓ Deductible INCLUÍDO: valores de ${X['deductible'].min():,.0f} a ${X['deductible'].max():,.0f}")
    print(f"   → Média: ${X['deductible'].mean():,.0f}")
else:
    print("   → ⚠️ Coluna 'deductible' não encontrada!")

# ============================================================================
# 3. FEATURE ENGINEERING (igual aos outros)
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

# Feature extra: interação entre deductible e condições
if 'deductible' in X.columns:
    X['deductible_risk'] = X['deductible'] * X['risk_proxy']
    X['deductible_chronic'] = X['deductible'] * X['chronic_count']

X = X.drop(columns=['smoker_num'], errors='ignore')

print(f"   → Features: {X.shape[1]}")

# ============================================================================
# 4. PREPARAR DADOS
# ============================================================================
print("\n[4/6] Preparando dados...")

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

X_train_orig, X_test_orig, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

preprocess = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("numeric", "passthrough", num_cols)
    ]
)

preprocess.fit(X_train_orig)
X_train = preprocess.transform(X_train_orig)
X_test = preprocess.transform(X_test_orig)

print(f"   → Treino: {X_train.shape[0]:,} | Teste: {X_test.shape[0]:,}")

# ============================================================================
# 5. TREINAR RANDOM FOREST
# ============================================================================
print("\n[5/6] Treinando Random Forest com deductible...")

start_time = time.time()

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=4,
    n_jobs=-1,
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
print(f" RESULTADOS - RANDOM FOREST COM DEDUCTIBLE")
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
print(" TOP 20 FEATURES (verificar posição de deductible)")
print("=" * 70)

ohe = preprocess.named_transformers_["categorical"]
cat_feature_names = list(ohe.get_feature_names_out(cat_cols))
all_feature_names = cat_feature_names + num_cols

importance_df = pd.DataFrame({
    'feature': all_feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(importance_df.head(20).to_string(index=False))

# Verificar posição de deductible
deductible_importance = importance_df[importance_df['feature'].str.contains('deductible', case=False)]
print(f"\n--- Importância do Deductible ---")
print(deductible_importance.to_string(index=False))

# ============================================================================
# 8. GUARDAR RESULTADOS
# ============================================================================
print("\n" + "=" * 70)
print(" GUARDANDO MODELO")
print("=" * 70)

joblib.dump(model, "experiments/04_rf_with_deductible_model.pkl")
joblib.dump(preprocess, "experiments/04_rf_with_deductible_preprocess.pkl")
importance_df.to_csv("experiments/04_rf_with_deductible_importance.csv", index=False)

metrics = {
    'model': 'Random Forest + Deductible',
    'r2': r2,
    'mae': mae,
    'rmse': rmse,
    'wmape': wmape,
    'train_time': train_time,
    'includes_deductible': True
}
joblib.dump(metrics, "experiments/04_rf_with_deductible_metrics.pkl")

print("   → Modelo guardado: 04_rf_with_deductible_model.pkl")

print(f"\n{'='*65}")
print(f" EXPERIMENTO 04 CONCLUÍDO")
print(f" R² = {r2:.4f} | MAE = ${mae:,.2f} | WMAPE = {wmape:.2f}%")
print(f"{'='*65}")
print(f"\n NOTA: Compare este R² com o Experimento 01 (sem deductible)")
print(f"       Se for MUITO maior, deductible pode ser data leakage suave.")
