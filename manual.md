# Risk Score Prediction Model

Sistema de **Machine Learning para predição de Risk Score contínuo**, desenvolvido com foco em boas práticas, robustez, interpretabilidade e preparação para uso em produção.

---

## 📌 Visão Geral

Este projeto implementa um modelo de **regressão baseado em Redes Neuronais** para prever um *risk score* a partir de dados demográficos e socioeconómicos.  
O sistema evita problemas comuns como **data leakage**, **overfitting** e **bias algorítmico**, sendo adequado para contextos académicos e profissionais.

---

## 🛠️ Tecnologias Utilizadas

- **Python 3.x**
- **Pandas / NumPy** – Manipulação de dados
- **Scikit-learn** – Pré-processamento e métricas
- **TensorFlow / Keras** – Rede Neuronal
- **Matplotlib / Seaborn** – Visualização
- **Joblib** – Persistência de modelos


---

## 📊 Dados

- **Fonte:** `ECF_2.xlsx`
- **Variável alvo:** `risk_score` (contínua)

### ❌ Remoção de Data Leakage

Foram removidas variáveis que representam consequências do risco, tais como:

- `monthly_premium`
- `annual_premium`
- `total_claims_paid`
- `is_high_risk`

> Estas variáveis não estariam disponíveis para novos clientes e causariam fuga de informação.

---

## 🔀 Divisão dos Dados

- **80% Treino**
- **20% Teste**

⚠️ A divisão é feita **antes do pré-processamento** para evitar data leakage.

---

## ⚙️ Pré-processamento

Realizado com `ColumnTransformer`:

### Variáveis Categóricas
- `OneHotEncoder(handle_unknown='ignore')`

### Variáveis Numéricas
- `StandardScaler`

### Conversão de Tipo
- Dados convertidos para `float32` (melhor desempenho e menor uso de memória)

---

## 🧠 Arquitetura da Rede Neuronal

Arquitetura em **funil**, típica para regressão:

- Dense (256, ReLU)
- Dropout (0.3)
- Dense (128, ReLU)
- Dropout (0.2)
- Dense (64, ReLU)
- Dense (1, Linear)

### Justificação
- **ReLU:** rápida convergência
- **Dropout:** reduz overfitting
- **Saída linear:** adequada para regressão contínua

---

## 🧪 Compilação

- **Otimizador:** Adam (lr = 0.001)
- **Loss:** Mean Squared Error (MSE)
- **Métrica:** Mean Absolute Error (MAE)

---

## 🚀 Treino

- Máx. **200 épocas**
- **Batch size:** 32
- **Validação:** 20% do treino
- **Early Stopping:** patience = 20

> O modelo restaura automaticamente os melhores pesos.

---

## 📈 Avaliação

### Métricas Absolutas
- MAE
- RMSE
- MedAE

### Métricas Relativas
- R² Score
- CV-RMSE
- NRMSE
- WMAPE (robusta a zeros)

---

## 📉 Visualizações

- Real vs Previsto
- Distribuição dos resíduos
- Curvas de treino (training vs validation loss)

Estas análises ajudam a identificar **overfitting** e **enviesamentos**.

---

## ⚖️ Fairness & Bias Analysis

O desempenho é avaliado por grupos categóricos (ex.: género, região).

🎯 Objetivo: garantir que o modelo funciona de forma equitativa entre diferentes grupos.

---

## 📋 Resultados Individuais

Tabela final com:
- Valor real
- Valor previsto
- Erro absoluto

Permite inspeção qualitativa dos resultados.

---

## 💾 Persistência

São guardados:
- `risk_score_model.keras` – Modelo treinado
- `preprocess.pkl` – Pipeline de pré-processamento

⚠️ **Sem o preprocessador, o modelo não pode ser reutilizado corretamente.**

---

## ✅ Conclusão

Este projeto segue boas práticas de Machine Learning, sendo:

- Robusto
- Escalável
- Interpretável
- Preparado para produção

Adequado para **projetos académicos, dissertações ou aplicações reais**.

---

## 👤 Autors

Alexandre Carvalho  19255

Bruno Ribeiro       30767

Tiago Oliveira      a16622
