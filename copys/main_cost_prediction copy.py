"""
================================================================================
 MODELO DE PREVISÃO DE CUSTOS MÉDICOS ANUAIS
================================================================================
 Autor: [Tino]
 Data: Janeiro 2026
 Objetivo: Prever o custo médico anual (annual_medical_cost) de um paciente
           com base nas suas características demográficas, clínicas e histórico.

 IMPORTANTE: Este modelo foi desenhado para funcionar em cenários REAIS,
 onde queremos prever custos para NOVOS clientes antes de lhes atribuir um preço.
 Por isso, removemos todas as variáveis que só existiriam DEPOIS de conhecer o custo.
================================================================================
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print(" MODELO DE PREVISÃO DE CUSTOS MÉDICOS ANUAIS")
print(" Versão: Sem Data Leakage (Produção)")
print("=" * 70)

# ============================================================================
# 1. CARREGAR DADOS
# ============================================================================
print("\n[1/7] Carregando dados...")

df = pd.read_excel("ECF_2.xlsx")
print(f"   → Dataset carregado: {df.shape[0]:,} linhas × {df.shape[1]} colunas")

# ============================================================================
# 2. DEFINIR TARGET E REMOVER DATA LEAKAGE
# ============================================================================
# 
# O QUE É DATA LEAKAGE?
# ---------------------
# Data Leakage acontece quando usamos informação no treino que NÃO estaria
# disponível no momento da previsão real. Por exemplo:
#   - Usar 'total_claims_paid' para prever 'annual_medical_cost' é leakage
#     porque os claims só são pagos DEPOIS de o custo acontecer!
#
# CENÁRIO REAL DE NEGÓCIO:
# ------------------------
# Uma seguradora recebe um novo cliente. Precisa de definir o preço do seguro
# ANTES de o cliente usar qualquer serviço médico. Nesse momento, a seguradora
# NÃO tem acesso a:
#   - Quantos claims o cliente vai fazer (claims_count)
#   - Quanto vai custar em média cada claim (avg_claim_amount)
#   - O risco calculado (risk_score) - porque este é derivado do custo!
#   - O prémio (annual_premium) - porque é o que queremos calcular!
#
# Por isso, REMOVEMOS todas essas variáveis do treino.
# ============================================================================

print("\n[2/7] Configurando target e removendo data leakage...")

target = "annual_medical_cost"

# ==========================================================================
# COLUNAS A REMOVER (DATA LEAKAGE)
# ==========================================================================
# Cada coluna removida tem uma justificação específica:
#
# person_id: Identificador único, não tem valor preditivo
# risk_score: É CALCULADO pela seguradora com base no custo esperado (circular!)
# is_high_risk: Derivado direto do risk_score
# annual_premium: É o PREÇO que a seguradora cobra - calculado COM BASE no custo
# monthly_premium: Simplesmente annual_premium / 12
# copay: Valor que o cliente paga do bolso - definido com base no risco
# claims_count: Número de claims feitos - só existe DEPOIS de usar o seguro
# avg_claim_amount: Valor médio por claim - só existe DEPOIS dos claims
# total_claims_paid: Total pago em claims - É PRATICAMENTE O TARGET!
# policy_changes_last_2yrs: Pode indicar ajustes por problemas de custo
# provider_quality: Pode ser ajustado ao perfil de risco do cliente
# ==========================================================================

cols_to_drop = [
    target,                      # O próprio target (annual_medical_cost)
    "person_id",                 # ID sem valor preditivo
    
    # --- DATA LEAKAGE: Variáveis derivadas do custo ---
    "risk_score",                # Calculado COM BASE no custo esperado
    "is_high_risk",              # Binário derivado do risk_score
    
    # --- DATA LEAKAGE: Variáveis de preço (output da seguradora) ---
    "annual_premium",            # Preço anual - é o que queremos CALCULAR
    "monthly_premium",           # annual_premium / 12
    "copay",                     # Definido com base no risco
    
    # --- DATA LEAKAGE: Variáveis de claims (só existem após uso) ---
    "claims_count",              # Quantos claims o cliente fez
    "avg_claim_amount",          # Média por claim
    "total_claims_paid",         # Soma dos claims ≈ custo real!
    
    # --- VARIÁVEIS AMBÍGUAS (removidas por precaução) ---
    "policy_changes_last_2yrs",  # Pode refletir ajustes de risco
    "provider_quality",          # Pode ser atribuído com base no risco
]

# Separar X (features) e y (target)
X = df.drop(columns=cols_to_drop, errors='ignore')
y = df[target].astype(float)

print(f"   → Target: {target}")
print(f"   → Colunas removidas (leakage): {len(cols_to_drop)}")
print(f"   → Features restantes: {X.shape[1]}")

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================
#
# PORQUÊ CRIAR NOVAS FEATURES?
# ----------------------------
# As redes neuronais conseguem aprender relações complexas, MAS ajudamos o
# modelo se criarmos features que capturam conhecimento do domínio médico.
#
# Por exemplo: um fumador de 60 anos com diabetes é muito mais caro do que
# a soma individual desses fatores. A feature 'age_chronic_interaction'
# captura esta interação.
# ============================================================================

print("\n[3/7] Criando features derivadas (Feature Engineering)...")

# --------------------------------------------------------------------------
# 3.1 RISK PROXY (Substituto do risk_score sem leakage)
# --------------------------------------------------------------------------
# Como removemos o risk_score (era leakage), criamos um proxy baseado em
# fatores de risco conhecidos na literatura médica:
#   - Doenças crónicas (30%): Maior driver de custos
#   - Idade (25%): Custos aumentam exponencialmente com idade
#   - BMI (20%): Obesidade correlacionada com múltiplas doenças
#   - Tabagismo (15%): Fator de risco cardiovascular e oncológico
#   - Medicamentos (10%): Proxy para complexidade de saúde
# --------------------------------------------------------------------------

# Converter 'smoker' para numérico se necessário
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

print("   → risk_proxy: Proxy de risco baseado em fatores clínicos")

# --------------------------------------------------------------------------
# 3.2 UTILIZATION SCORE (Índice de Utilização de Serviços)
# --------------------------------------------------------------------------
# Captura a intensidade de uso de serviços de saúde no passado.
# Usa log1p (log(1+x)) para comprimir valores extremos.
# Hospitalizações pesam mais (0.6) porque são mais caras.
# --------------------------------------------------------------------------

X['utilization_score'] = (
    0.6 * np.log1p(X['hospitalizations_last_3yrs']) +
    0.4 * np.log1p(X['visits_last_year'])
)
print("   → utilization_score: Histórico de utilização de serviços")

# --------------------------------------------------------------------------
# 3.3 PROCEDURE INTENSITY (Intensidade de Procedimentos)
# --------------------------------------------------------------------------
# Soma ponderada de procedimentos médicos realizados.
# Cirurgias têm peso 3x porque são muito mais caras que exames.
# --------------------------------------------------------------------------

X['procedure_intensity'] = (
    X['proc_surgery_count'] * 3 +      # Cirurgias = alto custo
    X['proc_imaging_count'] * 1.5 +    # Imagiologia = custo moderado
    X['proc_lab_count'] +               # Lab = custo baixo
    X['proc_consult_count'] +           # Consultas = custo baixo
    X['proc_physio_count']              # Fisioterapia = custo baixo
)
print("   → procedure_intensity: Intensidade de procedimentos médicos")

# --------------------------------------------------------------------------
# 3.4 HOSPITAL SEVERITY (Severidade de Hospitalizações)
# --------------------------------------------------------------------------
# Média de dias por hospitalização. Hospitalizações longas indicam
# casos mais graves e, portanto, mais caros.
# --------------------------------------------------------------------------

X['hospital_severity'] = (
    X['days_hospitalized_last_3yrs'] / 
    X['hospitalizations_last_3yrs'].replace(0, 1)
)
print("   → hospital_severity: Dias médios por hospitalização")

# --------------------------------------------------------------------------
# 3.5 INTERACTION FEATURES (Interações entre variáveis)
# --------------------------------------------------------------------------
# Capturam efeitos combinados que são maiores que a soma das partes.
# Ex: Idade avançada + muitas doenças crónicas = custo exponencial
# --------------------------------------------------------------------------

X['age_chronic_interaction'] = X['age'] * X['chronic_count']
X['bmi_chronic_interaction'] = X['bmi'] * X['chronic_count']
X['age_bmi_interaction'] = (X['age'] / 100) * (X['bmi'] / 50)

print("   → age_chronic_interaction: Interação idade × doenças")
print("   → bmi_chronic_interaction: Interação BMI × doenças")
print("   → age_bmi_interaction: Interação idade × BMI")

# --------------------------------------------------------------------------
# 3.6 COMORBIDITY INDEX (Índice de Comorbilidades)
# --------------------------------------------------------------------------
# Conta doenças "graves" (cardiovascular, diabetes, cancro, renal).
# Estas são as que mais impactam custos a longo prazo.
# --------------------------------------------------------------------------

severe_conditions = ['cardiovascular_disease', 'diabetes', 'cancer_history', 
                     'kidney_disease', 'copd']
X['severe_comorbidity_count'] = X[severe_conditions].sum(axis=1)
print("   → severe_comorbidity_count: Número de comorbilidades graves")

# Remover coluna temporária
X = X.drop(columns=['smoker_num'], errors='ignore')

print(f"\n   → Total de features após engineering: {X.shape[1]}")

# ============================================================================
# 4. IDENTIFICAR TIPOS DE COLUNAS
# ============================================================================

print("\n[4/7] Identificando tipos de colunas...")

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

print(f"   → Colunas categóricas ({len(cat_cols)}): {cat_cols}")
print(f"   → Colunas numéricas ({len(num_cols)}): {len(num_cols)} colunas")

# ============================================================================
# 5. DIVISÃO DOS DADOS (TRAIN/TEST SPLIT)
# ============================================================================
#
# PORQUÊ DIVIDIR ANTES DO PRÉ-PROCESSAMENTO?
# ------------------------------------------
# Se calcularmos a média/desvio padrão (StandardScaler) usando TODOS os dados,
# a informação do conjunto de TESTE "vaza" para o TREINO. Isto é outra forma
# de Data Leakage!
#
# Regra de ouro: O modelo não pode "ver" absolutamente NADA do teste durante
# o treino, incluindo estatísticas como média e desvio padrão.
# ============================================================================

print("\n[5/7] Dividindo dados em treino/teste...")

X_train_orig, X_test_orig, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 80% treino, 20% teste
    random_state=42     # Semente para reprodutibilidade
)

print(f"   → Treino: {X_train_orig.shape[0]:,} amostras")
print(f"   → Teste: {X_test_orig.shape[0]:,} amostras")

# ============================================================================
# 6. PRÉ-PROCESSAMENTO
# ============================================================================
#
# ONEHOTENCODER (Categóricas):
# ----------------------------
# Transforma categorias em colunas binárias.
# Ex: smoker = ["Never", "Current", "Former"] → 3 colunas (0 ou 1)
#
# STANDARDSCALER (Numéricas):
# ---------------------------
# Coloca todas as variáveis na mesma escala: média=0, desvio=1
# MOTIVO: Redes neuronais funcionam muito mal se misturarmos:
#   - Idade: 18-80
#   - Income: 10,000-500,000
# A escala diferente faz com que a rede dê mais importância a valores grandes.
# ============================================================================

print("\n[6/7] Pré-processando dados...")

preprocess = ColumnTransformer(
    transformers=[
        # OneHotEncoder: handle_unknown='ignore' previne erros com categorias novas
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        # StandardScaler: normaliza para média=0, desvio=1
        ("numeric", StandardScaler(), num_cols)
    ]
)

# CRÍTICO: .fit() apenas no TREINO!
preprocess.fit(X_train_orig)

# Aplicar transformação em ambos
X_train = preprocess.transform(X_train_orig)
X_test = preprocess.transform(X_test_orig)

# Converter para float32 (TensorFlow prefere, usa menos memória)
X_train = np.array(X_train, dtype="float32")
X_test = np.array(X_test, dtype="float32")

print(f"   → Shape após encoding: {X_train.shape[1]} features")
print(f"   → Treino: {X_train.shape}")
print(f"   → Teste: {X_test.shape}")

# ============================================================================
# 7. ARQUITETURA DA REDE NEURONAL
# ============================================================================
#
# DECISÕES DE ARQUITETURA:
# ------------------------
# 1. CAMADAS EM FUNIL (512→256→128→64→1):
#    Começa larga para capturar muitos padrões, depois comprime para extrair
#    apenas os mais importantes. É como um funil de informação.
#
# 2. BATCH NORMALIZATION:
#    Normaliza as ativações entre camadas. Estabiliza o treino e permite
#    usar learning rates mais altos. Acelera a convergência.
#
# 3. DROPOUT (0.3, 0.25, 0.2):
#    Desliga neurónios aleatoriamente durante o treino. Previne overfitting
#    ao forçar a rede a não depender de neurónios específicos.
#    Dropout decresce nas camadas mais profundas (menos neurónios para desligar).
#
# 4. L2 REGULARIZATION:
#    Penaliza pesos muito grandes. Força o modelo a usar pesos pequenos,
#    o que resulta em modelos mais simples e generalizáveis.
#
# 5. ACTIVATION 'relu':
#    Rectified Linear Unit. Standard da indústria para camadas ocultas.
#    Resolve o problema de "vanishing gradients" de funções como sigmoid.
#
# 6. OUTPUT LINEAR:
#    Como é regressão (prever valor contínuo), a última camada não tem
#    ativação (ou tem 'linear'). Não queremos limitar o output.
# ============================================================================

print("\n[7/7] Construindo modelo de rede neuronal...")

model = Sequential([
    # Camada 1: Entrada + primeira oculta
    Dense(512, activation='relu', input_shape=(X_train.shape[1],),
          kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),
    
    # Camada 2
    Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.25),
    
    # Camada 3
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.2),
    
    # Camada 4
    Dense(64, activation='relu'),
    Dropout(0.1),
    
    # Camada de saída: 1 neurónio (regressão)
    Dense(1, activation='linear')
])

# --------------------------------------------------------------------------
# COMPILAÇÃO DO MODELO
# --------------------------------------------------------------------------
# Optimizer: Adam com learning rate 0.001 (default, funciona bem na maioria)
# Loss: MSE (Mean Squared Error) - penaliza erros grandes quadraticamente
# Metrics: MAE (Mean Absolute Error) - mais interpretável para humanos
# --------------------------------------------------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print("\n" + "=" * 70)
print(" ARQUITETURA DO MODELO")
print("=" * 70)
model.summary()

# ============================================================================
# 8. CALLBACKS PARA TREINO
# ============================================================================
#
# EARLY STOPPING:
# ---------------
# Monitoriza o val_loss (erro na validação). Se não melhorar durante 25
# épocas (patience), para o treino. Evita overfitting e poupa tempo.
# restore_best_weights=True: No fim, usa os pesos da melhor época.
#
# REDUCE LR ON PLATEAU:
# ---------------------
# Se o val_loss estagnar durante 10 épocas, reduz o learning rate para 50%.
# Permite "ajuste fino" quando o modelo está perto do ótimo.
# ============================================================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=25,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,          # Reduz LR para metade
    patience=10,         # Espera 10 épocas antes de reduzir
    min_lr=0.00001,      # Não reduz abaixo disto
    verbose=1
)

# ============================================================================
# 9. TREINO DO MODELO
# ============================================================================

print("\n" + "=" * 70)
print(" INICIANDO TREINO")
print("=" * 70)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,    # 20% do treino para validação
    epochs=200,              # Máximo de épocas (early stopping vai parar antes)
    batch_size=64,           # Atualiza pesos a cada 64 amostras
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ============================================================================
# 10. AVALIAÇÃO DO MODELO
# ============================================================================

print("\n" + "=" * 70)
print(" AVALIAÇÃO DO MODELO")
print("=" * 70)

# Fazer previsões no conjunto de teste
preds = model.predict(X_test).flatten()

# --------------------------------------------------------------------------
# MÉTRICAS NÃO NORMALIZADAS (valores absolutos)
# --------------------------------------------------------------------------
mae = mean_absolute_error(y_test, preds)
mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)
medae = median_absolute_error(y_test, preds)

# --------------------------------------------------------------------------
# MÉTRICAS NORMALIZADAS (percentuais - comparáveis entre datasets)
# --------------------------------------------------------------------------
y_max = y_test.max()
y_min = y_test.min()
y_range = y_max - y_min
y_mean = y_test.mean()

# NRMSE: Erro como % do range dos dados
nrmse = rmse / y_range

# CV-RMSE: Erro como % da média (Coeficiente de Variação)
cv_rmse = rmse / y_mean

# R² Score: Quanto da variância o modelo explica (1 = perfeito)
r2 = r2_score(y_test, preds)

# WMAPE: Weighted Mean Absolute Percentage Error
# Mais robusto que MAPE (não explode com zeros)
wmape = (np.sum(np.abs(y_test - preds)) / np.sum(y_test)) * 100

# --------------------------------------------------------------------------
# EXIBIR RESULTADOS
# --------------------------------------------------------------------------
print(f"\n{'='*65}")
print(f" MÉTRICAS DE PERFORMANCE (Conjunto de Teste)")
print(f"{'='*65}")
print(f"\n--- Métricas Absolutas (em $) ---")
print(f"{'MAE (Erro Médio)':<25} | ${mae:,.2f}")
print(f"{'RMSE (Erro Quadrático)':<25} | ${rmse:,.2f}")
print(f"{'MedAE (Erro Mediano)':<25} | ${medae:,.2f}")

print(f"\n--- Métricas Relativas (%) ---")
print(f"{'WMAPE':<25} | {wmape:.2f}%")
print(f"{'CV-RMSE':<25} | {cv_rmse*100:.2f}%")
print(f"{'NRMSE':<25} | {nrmse*100:.2f}%")

print(f"\n--- Qualidade do Modelo ---")
print(f"{'R² Score':<25} | {r2:.4f}")
print(f"{'Interpretação':<25} | O modelo explica {r2*100:.1f}% da variância")
print(f"{'='*65}")

# --------------------------------------------------------------------------
# VALIDAÇÃO: Verificar se há Data Leakage
# --------------------------------------------------------------------------
print(f"\n⚠️  VERIFICAÇÃO DE DATA LEAKAGE:")
if r2 > 0.90:
    print(f"   ❌ R² = {r2:.4f} é SUSPEITOSAMENTE ALTO!")
    print(f"      Pode haver data leakage. Verificar features.")
elif r2 > 0.70:
    print(f"   ✅ R² = {r2:.4f} está EXCELENTE para este problema.")
elif r2 > 0.50:
    print(f"   ✅ R² = {r2:.4f} está BOM. Custos médicos são difíceis de prever.")
else:
    print(f"   ⚠️  R² = {r2:.4f} está baixo. Considerar mais feature engineering.")

# ============================================================================
# 11. GRÁFICOS DE ANÁLISE
# ============================================================================

print("\n\nGerando gráficos de análise...")

# Calcular resíduos (erros)
residuals = y_test - preds

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# --------------------------------------------------------------------------
# Gráfico 1: Real vs Previsto
# --------------------------------------------------------------------------
# Se o modelo fosse perfeito, todos os pontos estariam na linha vermelha.
# --------------------------------------------------------------------------
ax1 = axes[0, 0]
ax1.scatter(y_test, preds, alpha=0.3, color='royalblue', s=10)
ax1.plot([y_min, y_max], [y_min, y_max], 'r--', lw=2, label='Previsão Perfeita')
ax1.set_xlabel("Custo Real ($)")
ax1.set_ylabel("Custo Previsto ($)")
ax1.set_title("Real vs Previsto")
ax1.legend()
ax1.grid(True, alpha=0.3)

# --------------------------------------------------------------------------
# Gráfico 2: Distribuição dos Resíduos
# --------------------------------------------------------------------------
# Se centrado no 0 = modelo não tem bias (viés) sistemático
# Se enviesado para um lado = modelo subestima ou sobrestima
# --------------------------------------------------------------------------
ax2 = axes[0, 1]
sns.histplot(residuals, kde=True, color='purple', bins=50, ax=ax2)
ax2.axvline(0, color='r', linestyle='--', label='Zero (Ideal)')
ax2.axvline(residuals.mean(), color='orange', linestyle='-', label=f'Média: ${residuals.mean():.0f}')
ax2.set_xlabel("Erro (Resíduo) em $")
ax2.set_title("Distribuição dos Erros")
ax2.legend()
ax2.grid(True, alpha=0.3)

# --------------------------------------------------------------------------
# Gráfico 3: Curvas de Treino
# --------------------------------------------------------------------------
# Se val_loss sobe enquanto loss desce = Overfitting
# Se ambas descem juntas = Treino saudável
# --------------------------------------------------------------------------
ax3 = axes[1, 0]
ax3.plot(history.history["loss"], label="Treino")
ax3.plot(history.history["val_loss"], label="Validação")
ax3.set_xlabel("Épocas")
ax3.set_ylabel("Loss (MSE)")
ax3.set_title("Curvas de Aprendizagem")
ax3.legend()
ax3.grid(True)

# --------------------------------------------------------------------------
# Gráfico 4: Resíduos vs Valores Previstos
# --------------------------------------------------------------------------
# Deve ser uma "nuvem" aleatória. Se houver padrão (ex: funil),
# o modelo tem problemas de heteroscedasticidade.
# --------------------------------------------------------------------------
ax4 = axes[1, 1]
ax4.scatter(preds, residuals, alpha=0.3, color='green', s=10)
ax4.axhline(0, color='r', linestyle='--')
ax4.set_xlabel("Valor Previsto ($)")
ax4.set_ylabel("Resíduo ($)")
ax4.set_title("Resíduos vs Previstos (Verificar Heterocedasticidade)")
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("model_analysis.png", dpi=150, bbox_inches='tight')
plt.show()

print("   → Gráficos salvos em 'model_analysis.png'")

# ============================================================================
# 12. ANÁLISE POR GRUPO (FAIRNESS / BIAS)
# ============================================================================
#
# PORQUÊ ANALISAR POR GRUPO?
# --------------------------
# Na saúde, é crítico verificar se o modelo funciona igualmente bem para
# todos os grupos demográficos. Um modelo que funciona bem para homens mas
# mal para mulheres é injusto e potencialmente ilegal.
# ============================================================================

print("\n" + "=" * 70)
print(" ANÁLISE DE FAIRNESS (Viés por Grupo)")
print("=" * 70)

analysis_df = X_test_orig.copy()
analysis_df['Real'] = y_test.values
analysis_df['Predicted'] = preds
analysis_df['Abs_Error'] = np.abs(analysis_df['Real'] - analysis_df['Predicted'])
analysis_df['Pct_Error'] = (analysis_df['Abs_Error'] / analysis_df['Real']) * 100

for col in cat_cols[:5]:  # Analisar top 5 categóricas
    if col in analysis_df.columns:
        print(f"\n--- {col.upper()} ---")
        group_metrics = analysis_df.groupby(col).agg(
            N=('Real', 'count'),
            Custo_Medio_Real=('Real', 'mean'),
            MAE=('Abs_Error', 'mean'),
            Erro_Pct_Medio=('Pct_Error', 'mean')
        ).round(2)
        print(group_metrics.to_string())

# ============================================================================
# 13. TABELA DE PREVISÕES EXEMPLO
# ============================================================================

print("\n" + "=" * 70)
print(" EXEMPLOS DE PREVISÕES")
print("=" * 70)

results_df = pd.DataFrame({
    "Custo_Real": y_test.values,
    "Custo_Previsto": preds,
    "Erro_Absoluto": np.abs(y_test.values - preds),
    "Erro_Percentual": (np.abs(y_test.values - preds) / y_test.values * 100).round(1)
})

print("\nPrimeiras 20 previsões:")
print(results_df.head(20).to_string())

# ============================================================================
# 14. GUARDAR MODELO E ARTEFACTOS
# ============================================================================

print("\n" + "=" * 70)
print(" GUARDANDO MODELO")
print("=" * 70)

# Guardar modelo Keras
model.save("cost_model_final.keras")
print("   → Modelo guardado: cost_model_final.keras")

# Guardar preprocessador (ESSENCIAL para usar o modelo em produção!)
joblib.dump(preprocess, "preprocess_cost_final.pkl")
print("   → Preprocessador guardado: preprocess_cost_final.pkl")

# Guardar lista de features usadas
feature_info = {
    'categorical_columns': cat_cols,
    'numeric_columns': num_cols,
    'total_features': X_train.shape[1],
    'target': target,
    'metrics': {
        'r2': float(r2),
        'mae': float(mae),
        'rmse': float(rmse),
        'wmape': float(wmape)
    }
}
joblib.dump(feature_info, "feature_info.pkl")
print("   → Info de features guardada: feature_info.pkl")

print("\n" + "=" * 70)
print(" TREINO CONCLUÍDO COM SUCESSO!")
print("=" * 70)
print(f"\n📊 Resumo Final:")
print(f"   • R² Score: {r2:.4f}")
print(f"   • MAE: ${mae:,.2f}")
print(f"   • WMAPE: {wmape:.2f}%")
print(f"\n💾 Ficheiros gerados:")
print(f"   • cost_model_final.keras (modelo)")
print(f"   • preprocess_cost_final.pkl (preprocessador)")
print(f"   • feature_info.pkl (metadados)")
print(f"   • model_analysis.png (gráficos)")
