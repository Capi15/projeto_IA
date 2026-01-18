# 🏥 Modelo de Previsão de Custos Médicos Anuais

## 📋 Índice
1. [Introdução](#introdução)
2. [O Problema de Negócio](#o-problema-de-negócio)
3. [Dataset](#dataset)
4. [Metodologia](#metodologia)
5. [Data Leakage - O Problema Crítico](#data-leakage---o-problema-crítico)
6. [Feature Engineering](#feature-engineering)
7. [Arquitetura do Modelo](#arquitetura-do-modelo)
8. [Resultados Esperados](#resultados-esperados)
9. [Como Usar](#como-usar)
10. [Conclusões](#conclusões)

---

## Introdução

Este projeto desenvolve um modelo de **Machine Learning** para prever o **custo médico anual** de um paciente com base nas suas características demográficas, clínicas e histórico de utilização de serviços de saúde.

O modelo utiliza uma **Rede Neuronal Profunda** (Deep Neural Network) implementada em TensorFlow/Keras e foi desenhado especificamente para funcionar em **cenários reais de produção**, evitando um problema comum chamado **Data Leakage**.

---

## O Problema de Negócio

### Cenário Real

Uma **seguradora de saúde** recebe um novo cliente que quer contratar um seguro médico. A seguradora precisa de:

1. **Avaliar o risco** do cliente (probabilidade de custos elevados)
2. **Definir um preço justo** para o prémio mensal
3. **Garantir rentabilidade** sem cobrar preços injustos

### O Desafio

No momento em que o cliente se apresenta, a seguradora **NÃO TEM** acesso a:
- Quantos claims o cliente vai fazer no futuro
- Qual será o custo total dos seus tratamentos
- Se ele vai ser hospitalizado ou não

A seguradora só tem acesso a:
- ✅ Dados demográficos (idade, sexo, região)
- ✅ Histórico médico (doenças crónicas, hospitalizações passadas)
- ✅ Estilo de vida (tabagismo, IMC, álcool)
- ✅ Características do plano escolhido

### Objetivo do Modelo

> **Prever o `annual_medical_cost` de um paciente usando APENAS informação disponível no momento da subscrição do seguro.**

---

## Dataset

### Fonte
O dataset contém **100.000 registos** de pacientes com **54 variáveis**.

### Categorias de Variáveis

| Categoria | Variáveis | Descrição |
|-----------|-----------|-----------|
| **Demográficas** | `age`, `sex`, `region`, `urban_rural`, `income`, `education` | Características socioeconómicas |
| **Lifestyle** | `bmi`, `smoker`, `alcohol_freq` | Fatores de estilo de vida |
| **Clínicas** | `hypertension`, `diabetes`, `copd`, `chronic_count`, etc. | Condições de saúde |
| **Utilização** | `visits_last_year`, `hospitalizations_last_3yrs` | Histórico de uso |
| **Procedimentos** | `proc_surgery_count`, `proc_imaging_count`, etc. | Procedimentos realizados |
| **Plano** | `plan_type`, `network_tier`, `deductible` | Características do seguro |
| **Custos** | `annual_medical_cost`, `total_claims_paid`, etc. | Variáveis de custo |

---

## Metodologia

### Pipeline de Desenvolvimento

```
┌─────────────────┐
│ 1. Carregar     │
│    Dados        │
└────────┬────────┘
         ▼
┌─────────────────┐
│ 2. Remover      │
│    Data Leakage │  ← Passo CRÍTICO!
└────────┬────────┘
         ▼
┌─────────────────┐
│ 3. Feature      │
│    Engineering  │
└────────┬────────┘
         ▼
┌─────────────────┐
│ 4. Train/Test   │
│    Split        │  ← ANTES do pré-processamento!
└────────┬────────┘
         ▼
┌─────────────────┐
│ 5. Pré-         │
│    processamento│
└────────┬────────┘
         ▼
┌─────────────────┐
│ 6. Treinar      │
│    Modelo       │
└────────┬────────┘
         ▼
┌─────────────────┐
│ 7. Avaliar &    │
│    Validar      │
└─────────────────┘
```

---

## Data Leakage - O Problema Crítico

### O que é Data Leakage?

**Data Leakage** (fuga de informação) ocorre quando o modelo de treino tem acesso a informação que **não estaria disponível no momento da previsão real**.

### Exemplo Prático

Imagina que treinas um modelo para prever se um email é spam, mas usas como feature "se o email está na pasta Spam". O modelo teria 100% de accuracy, mas seria **completamente inútil** porque no momento real não sabes ainda onde o email vai parar!

### No Nosso Contexto

| Variável | Problema | Porquê é Leakage? |
|----------|----------|-------------------|
| `risk_score` | ❌ Remover | É **calculado pela seguradora** com base no custo esperado! |
| `annual_premium` | ❌ Remover | É o **preço que a seguradora cobra** - derivado do risco |
| `total_claims_paid` | ❌ Remover | É **praticamente o target** - soma dos custos pagos |
| `claims_count` | ❌ Remover | Só existe **depois** de o cliente usar o seguro |
| `is_high_risk` | ❌ Remover | Derivado direto do `risk_score` |

### Colunas Removidas

```python
cols_to_drop = [
    "person_id",              # ID sem valor preditivo
    "risk_score",             # Calculado COM BASE no custo esperado
    "is_high_risk",           # Derivado do risk_score
    "annual_premium",         # Preço cobrado (output da seguradora)
    "monthly_premium",        # annual_premium / 12
    "copay",                  # Definido com base no risco
    "claims_count",           # Só existe após uso do seguro
    "avg_claim_amount",       # Derivado dos claims
    "total_claims_paid",      # Soma dos claims ≈ target
    "policy_changes_last_2yrs", # Pode refletir ajustes de risco
    "provider_quality",       # Pode ser atribuído ao risco
]
```

### Como Detetar Leakage?

> **Regra de ouro:** Se o R² do modelo for > 0.90, provavelmente há leakage!

Prever custos médicos é **intrinsecamente difícil** porque depende de:
- Acidentes imprevisíveis
- Diagnósticos inesperados
- Variações individuais na resposta a tratamentos

Um R² de 0.60-0.75 é **excelente** para este problema. Se o modelo atingir R² > 0.95, é sinal de que está a "fazer batota" com informação do futuro.

---

## Feature Engineering

### Porquê Criar Novas Features?

Embora redes neuronais consigam aprender relações complexas, ajudamos o modelo ao **codificar conhecimento do domínio médico** em features derivadas.

### Features Criadas

#### 1. Risk Proxy (Substituto do Risk Score)

Como removemos `risk_score` (era leakage), criamos um proxy baseado em **fatores de risco conhecidos na literatura médica**:

```python
risk_proxy = (
    0.30 × (chronic_count / max_chronic) +  # Doenças crónicas
    0.25 × (age / 100) +                     # Idade
    0.20 × (bmi / 50) +                      # IMC
    0.15 × smoker_status +                   # Tabagismo
    0.10 × (medication_count / max_meds)     # Medicamentos
)
```

**Justificação dos pesos:**
- **30% Doenças Crónicas**: Principal driver de custos médicos
- **25% Idade**: Custos aumentam exponencialmente após os 50 anos
- **20% BMI**: Obesidade está associada a múltiplas condições
- **15% Tabagismo**: Fator de risco cardiovascular e oncológico
- **10% Medicamentos**: Proxy para complexidade de saúde

#### 2. Utilization Score

Captura a **intensidade de uso de serviços de saúde**:

```python
utilization_score = 0.6 × log(1 + hospitalizations) + 0.4 × log(1 + visits)
```

- Hospitalizações pesam mais (0.6) porque são **mais caras**
- Usa `log1p` para comprimir valores extremos

#### 3. Procedure Intensity

Soma ponderada de **procedimentos médicos**:

```python
procedure_intensity = (
    proc_surgery × 3 +      # Cirurgias = custo muito alto
    proc_imaging × 1.5 +    # Imagiologia = custo moderado
    proc_lab +              # Laboratório
    proc_consult +          # Consultas
    proc_physio             # Fisioterapia
)
```

#### 4. Hospital Severity

Média de **dias por hospitalização** (indicador de gravidade):

```python
hospital_severity = days_hospitalized / hospitalizations
```

#### 5. Interaction Features

Capturam efeitos **não-lineares** que são maiores que a soma das partes:

```python
age_chronic_interaction = age × chronic_count
bmi_chronic_interaction = bmi × chronic_count
age_bmi_interaction = (age/100) × (bmi/50)
```

**Exemplo**: Um fumador de 60 anos com diabetes custa **muito mais** do que:
- Custo(fumador) + Custo(60 anos) + Custo(diabético)

#### 6. Severe Comorbidity Count

Conta **doenças graves** especificamente:

```python
severe_conditions = ['cardiovascular_disease', 'diabetes', 
                     'cancer_history', 'kidney_disease', 'copd']
severe_comorbidity_count = sum(severe_conditions)
```

---

## Arquitetura do Modelo

### Rede Neuronal Profunda

```
┌─────────────────────────────────────────────────┐
│              Input Layer                        │
│         (N features após encoding)              │
└─────────────────────┬───────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────┐
│    Dense(512) + BatchNorm + Dropout(0.3)        │
│    Activation: ReLU, L2 Regularization          │
└─────────────────────┬───────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────┐
│    Dense(256) + BatchNorm + Dropout(0.25)       │
│    Activation: ReLU, L2 Regularization          │
└─────────────────────┬───────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────┐
│    Dense(128) + BatchNorm + Dropout(0.2)        │
│    Activation: ReLU, L2 Regularization          │
└─────────────────────┬───────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────┐
│         Dense(64) + Dropout(0.1)                │
│              Activation: ReLU                   │
└─────────────────────┬───────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────┐
│            Dense(1) - Output                    │
│           Activation: Linear                    │
└─────────────────────────────────────────────────┘
```

### Justificação das Escolhas

| Componente | Escolha | Justificação |
|------------|---------|--------------|
| **Arquitetura** | Funil (512→256→128→64→1) | Comprime informação gradualmente, extraindo padrões essenciais |
| **BatchNormalization** | Após cada Dense | Estabiliza treino, permite LR mais alto, acelera convergência |
| **Dropout** | 0.3→0.25→0.2→0.1 | Previne overfitting; decresce nas camadas profundas |
| **L2 Regularization** | λ=0.001 | Penaliza pesos grandes, força modelos mais simples |
| **Activation** | ReLU (ocultas), Linear (output) | ReLU resolve vanishing gradients; Linear para regressão |
| **Optimizer** | Adam | Standard da indústria, adapta LR automaticamente |
| **Loss** | MSE | Penaliza erros grandes quadraticamente |

### Callbacks

| Callback | Configuração | Propósito |
|----------|--------------|-----------|
| **EarlyStopping** | patience=25 | Para treino se val_loss não melhorar |
| **ReduceLROnPlateau** | patience=10, factor=0.5 | Reduz LR quando estagnar |

---

## Resultados Esperados

### Métricas Típicas (SEM Data Leakage)

| Métrica | Valor Esperado | Interpretação |
|---------|---------------|---------------|
| **R² Score** | 0.55 - 0.75 | O modelo explica 55-75% da variância |
| **MAE** | $2,000 - $4,000 | Erro médio em dólares |
| **WMAPE** | 15% - 30% | Erro percentual ponderado |
| **CV-RMSE** | 20% - 40% | Coeficiente de variação |

### Comparação: Com vs Sem Leakage

| Cenário | R² | MAE | WMAPE |
|---------|-----|-----|-------|
| **COM Leakage** ❌ | 0.95+ | $200 | 2% |
| **SEM Leakage** ✅ | 0.60-0.75 | $2,500 | 20% |

> ⚠️ **O modelo COM leakage parece melhor, mas é INÚTIL na prática!**

### Porque R² de 0.65 é BOM?

Prever custos médicos é **intrinsecamente difícil** porque:

1. **Eventos imprevisíveis**: Acidentes, diagnósticos súbitos
2. **Variabilidade individual**: Mesma doença, custos diferentes
3. **Fatores não observados**: Genética, compliance com tratamentos
4. **Aleatoriedade**: Alguns pacientes simplesmente têm "azar"

Na literatura académica, modelos de custo médico com R² > 0.50 são considerados **muito bons**.

---

## Como Usar

### Treinar o Modelo

```bash
cd /home/tino/Code/school/3_Semestre/AI/projeto_IA
python main_cost_prediction.py
```

### Ficheiros Gerados

| Ficheiro | Descrição |
|----------|-----------|
| `cost_model_final.keras` | Modelo treinado |
| `preprocess_cost_final.pkl` | Pipeline de pré-processamento |
| `feature_info.pkl` | Metadados das features |
| `model_analysis.png` | Gráficos de análise |

### Usar para Previsões Novas

```python
import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Carregar artefactos
model = load_model("cost_model_final.keras")
preprocess = joblib.load("preprocess_cost_final.pkl")
feature_info = joblib.load("feature_info.pkl")

# Novo paciente (exemplo)
novo_paciente = pd.DataFrame({
    'age': [45],
    'sex': ['Male'],
    'bmi': [28.5],
    'smoker': ['Former'],
    'chronic_count': [2],
    # ... outras features
})

# Aplicar mesmo feature engineering do treino
# ... (criar risk_proxy, utilization_score, etc.)

# Transformar e prever
X_novo = preprocess.transform(novo_paciente)
custo_previsto = model.predict(X_novo)
print(f"Custo anual previsto: ${custo_previsto[0][0]:,.2f}")
```

---

## Conclusões

### O Que Aprendemos

1. **Data Leakage é o inimigo nº1** em ML para negócios
   - Métricas incríveis no treino, modelo inútil em produção

2. **Feature Engineering baseado em domínio** melhora resultados
   - Conhecimento médico codificado em features derivadas

3. **R² "baixo" pode ser excelente** dependendo do problema
   - Custos médicos são intrinsecamente difíceis de prever

4. **Simplicidade > Complexidade**
   - Modelo bem desenhado supera modelo complexo com leakage

### Trabalho Futuro

- [ ] Experimentar Gradient Boosting (XGBoost, LightGBM)
- [ ] Adicionar mais interações de features
- [ ] Testar ensemble de modelos
- [ ] Implementar SHAP values para interpretabilidade

---

## Referências

1. Dataset: Kaggle - Medical Insurance Cost Prediction
2. TensorFlow/Keras Documentation
3. Scikit-learn User Guide
4. "Predictive Modeling of Health Care Costs" - Society of Actuaries

---

**Autor**: Tino  
**Data**: Janeiro 2026  
**Versão**: 1.0 (Sem Data Leakage)
