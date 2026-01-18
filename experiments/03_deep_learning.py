"""
================================================================================
 EXPERIMENTO 03: DEEP LEARNING (REDE NEURAL)
================================================================================
 Descrição: Rede Neural profunda com múltiplas camadas densas.
            Arquitetura: 512 → 256 → 128 → 64 → 1

 Vantagens:
   - Pode capturar relações extremamente complexas
   - Flexível e escalável
   - Bom para datasets muito grandes

 Desvantagens:
   - Requer mais dados para treinar bem
   - Mais difícil de interpretar (black box)
   - Geralmente pior que Gradient Boosting em dados tabulares

 Hipótese: Provavelmente vai ter performance similar ou inferior ao
           Gradient Boosting devido à natureza tabular dos dados.
================================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import warnings
import time
warnings.filterwarnings('ignore')

print("=" * 70)
print(" EXPERIMENTO 03: DEEP LEARNING (NEURAL NETWORK)")
print(" Dataset: 100k linhas | Sem Data Leakage")
print("=" * 70)

# ============================================================================
# 1. CARREGAR DADOS
# ============================================================================
print("\n[1/7] Carregando dados...")
df = pd.read_excel("ECF_2.xlsx")
print(f"   → Dataset: {df.shape[0]:,} linhas × {df.shape[1]} colunas")

# ============================================================================
# 2. CONFIGURAR TARGET E REMOVER LEAKAGE
# ============================================================================
print("\n[2/7] Removendo data leakage...")

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

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================
print("\n[3/7] Feature Engineering...")

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

print(f"   → Features: {X.shape[1]}")

# ============================================================================
# 4. PREPARAR DADOS
# ============================================================================
print("\n[4/7] Preparando dados...")

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

X_train_orig, X_test_orig, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Para Neural Networks, precisamos de normalização
preprocess = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("numeric", StandardScaler(), num_cols)  # StandardScaler é crucial para NN!
    ]
)

preprocess.fit(X_train_orig)
X_train = preprocess.transform(X_train_orig)
X_test = preprocess.transform(X_test_orig)

# Também normalizar o target para treino mais estável
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test_np = y_test.values

print(f"   → Treino: {X_train.shape[0]:,} | Teste: {X_test.shape[0]:,}")
print(f"   → Dimensão input: {X_train.shape[1]}")

# ============================================================================
# 5. CONSTRUIR REDE NEURAL
# ============================================================================
print("\n[5/7] Construindo Rede Neural...")

input_dim = X_train.shape[1]

model = Sequential([
    # Camada 1: 512 neurónios
    Dense(512, activation='relu', input_dim=input_dim,
          kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),
    
    # Camada 2: 256 neurónios
    Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),
    
    # Camada 3: 128 neurónios
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.2),
    
    # Camada 4: 64 neurónios
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.2),
    
    # Output: 1 valor (custo)
    Dense(1, activation='linear')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print(f"   → Parâmetros: {model.count_params():,}")

# ============================================================================
# 6. TREINAR MODELO
# ============================================================================
print("\n[6/7] Treinando modelo...")

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
]

start_time = time.time()

history = model.fit(
    X_train, y_train_scaled,
    validation_split=0.15,
    epochs=100,
    batch_size=256,
    callbacks=callbacks,
    verbose=1
)

train_time = time.time() - start_time
print(f"\n   → Tempo de treino: {train_time:.1f} segundos")
print(f"   → Epochs completadas: {len(history.history['loss'])}")

# ============================================================================
# 7. AVALIAÇÃO
# ============================================================================
print("\n[7/7] Avaliando modelo...")

preds_scaled = model.predict(X_test, verbose=0).ravel()
preds = y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()

mae = mean_absolute_error(y_test_np, preds)
mse = mean_squared_error(y_test_np, preds)
rmse = np.sqrt(mse)
medae = median_absolute_error(y_test_np, preds)
r2 = r2_score(y_test_np, preds)
wmape = (np.sum(np.abs(y_test_np - preds)) / np.sum(y_test_np)) * 100

print(f"\n{'='*65}")
print(f" RESULTADOS - DEEP LEARNING")
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
# 8. GUARDAR RESULTADOS
# ============================================================================
print("\n" + "=" * 70)
print(" GUARDANDO MODELO")
print("=" * 70)

model.save("experiments/03_deep_learning_model.keras")
joblib.dump(preprocess, "experiments/03_deep_learning_preprocess.pkl")
joblib.dump(y_scaler, "experiments/03_deep_learning_y_scaler.pkl")

metrics = {
    'model': 'Deep Learning',
    'r2': r2,
    'mae': mae,
    'rmse': rmse,
    'wmape': wmape,
    'train_time': train_time,
    'epochs': len(history.history['loss'])
}
joblib.dump(metrics, "experiments/03_deep_learning_metrics.pkl")

print("   → Modelo guardado: 03_deep_learning_model.keras")

print(f"\n{'='*65}")
print(f" EXPERIMENTO 03 CONCLUÍDO")
print(f" R² = {r2:.4f} | MAE = ${mae:,.2f} | WMAPE = {wmape:.2f}%")
print(f"{'='*65}")
