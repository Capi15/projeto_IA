"""
================================================================================
 COMPARAR TODOS OS EXPERIMENTOS
================================================================================
 Este script carrega os resultados de todos os experimentos e gera uma
 tabela comparativa para o relatório.

 Requisitos: Executar primeiro os experimentos 01, 02, 03, 04
================================================================================
"""

import os
import joblib
import pandas as pd

print("=" * 70)
print(" COMPARAÇÃO DE TODOS OS MODELOS")
print("=" * 70)

# Diretório dos experimentos
exp_dir = "experiments"

# Lista de ficheiros de métricas
metric_files = [
    ("01_random_forest_metrics.pkl", "Random Forest"),
    ("02_gradient_boosting_metrics.pkl", "Gradient Boosting"),
    ("03_deep_learning_metrics.pkl", "Deep Learning"),
    ("04_rf_with_deductible_metrics.pkl", "RF + Deductible"),
]

results = []

print("\n[1/3] Carregando resultados dos experimentos...\n")

for filename, expected_name in metric_files:
    filepath = os.path.join(exp_dir, filename)
    if os.path.exists(filepath):
        metrics = joblib.load(filepath)
        results.append({
            'Modelo': metrics.get('model', expected_name),
            'R²': metrics.get('r2', 0),
            'MAE ($)': metrics.get('mae', 0),
            'RMSE ($)': metrics.get('rmse', 0),
            'WMAPE (%)': metrics.get('wmape', 0),
            'Tempo (s)': metrics.get('train_time', 0),
        })
        print(f"   ✓ Carregado: {filename}")
    else:
        print(f"   ✗ Não encontrado: {filename}")

if not results:
    print("\n   ⚠️  Nenhum resultado encontrado!")
    print("   Execute primeiro os experimentos:")
    print("      python experiments/01_random_forest.py")
    print("      python experiments/02_gradient_boosting.py")
    print("      python experiments/03_deep_learning.py")
    print("      python experiments/04_with_deductible.py")
    exit()

# ============================================================================
# CRIAR TABELA COMPARATIVA
# ============================================================================
print("\n[2/3] Criando tabela comparativa...\n")

df = pd.DataFrame(results)

# Ordenar por R² (melhor primeiro)
df = df.sort_values('R²', ascending=False).reset_index(drop=True)

# Adicionar ranking
df.insert(0, 'Rank', range(1, len(df) + 1))

# ============================================================================
# MOSTRAR RESULTADOS
# ============================================================================
print("=" * 80)
print(" TABELA COMPARATIVA DOS MODELOS")
print("=" * 80)

# Formatação bonita
print(f"\n{'Rank':<5} | {'Modelo':<25} | {'R²':>8} | {'MAE ($)':>12} | {'WMAPE (%)':>10}")
print("-" * 80)

for _, row in df.iterrows():
    print(f"{row['Rank']:<5} | {row['Modelo']:<25} | {row['R²']:>8.4f} | "
          f"${row['MAE ($)']:>10,.2f} | {row['WMAPE (%)']:>10.2f}%")

print("-" * 80)

# ============================================================================
# ANÁLISE DOS RESULTADOS
# ============================================================================
print("\n" + "=" * 80)
print(" ANÁLISE DOS RESULTADOS")
print("=" * 80)

best = df.iloc[0]
worst = df.iloc[-1]

print(f"\n🏆 MELHOR MODELO: {best['Modelo']}")
print(f"   R² = {best['R²']:.4f} | MAE = ${best['MAE ($)']:,.2f} | WMAPE = {best['WMAPE (%)']:.2f}%")

print(f"\n📉 PIOR MODELO: {worst['Modelo']}")
print(f"   R² = {worst['R²']:.4f} | MAE = ${worst['MAE ($)']:,.2f} | WMAPE = {worst['WMAPE (%)']:.2f}%")

# Verificar se incluir deductible ajudou
if len(df) >= 4:
    rf_normal = df[df['Modelo'] == 'Random Forest']['R²'].values
    rf_deduct = df[df['Modelo'] == 'RF + Deductible']['R²'].values
    
    if len(rf_normal) > 0 and len(rf_deduct) > 0:
        diff = rf_deduct[0] - rf_normal[0]
        print(f"\n📊 IMPACTO DO DEDUCTIBLE:")
        if diff > 0.05:
            print(f"   ⚠️  Incluir deductible aumentou R² em {diff:.4f}")
            print(f"   Isto pode indicar data leakage suave (deductible correlacionado com custo)")
        elif diff > 0:
            print(f"   Incluir deductible aumentou R² ligeiramente ({diff:.4f})")
            print(f"   Provavelmente não é data leakage significativo")
        else:
            print(f"   Incluir deductible não ajudou (diferença: {diff:.4f})")

# ============================================================================
# GUARDAR RESULTADOS
# ============================================================================
print("\n[3/3] Guardando resultados...\n")

df.to_csv("experiments/00_comparison_results.csv", index=False)
print("   → Tabela guardada em: experiments/00_comparison_results.csv")

# Criar resumo para o relatório
summary = f"""
================================================================================
 RESUMO PARA O RELATÓRIO
================================================================================

OBJETIVO:
Prever o custo anual médico (annual_medical_cost) de pacientes usando
diferentes abordagens de Machine Learning.

DATA LEAKAGE:
Foram removidas variáveis que constituem data leakage, como annual_premium,
total_claims_paid, risk_score, etc., pois estas não estariam disponíveis
no momento da previsão para novos pacientes.

MODELOS TESTADOS:
"""

for _, row in df.iterrows():
    summary += f"\n{row['Rank']}. {row['Modelo']}: R² = {row['R²']:.4f}, MAE = ${row['MAE ($)']:,.2f}"

summary += f"""

MELHOR MODELO: {best['Modelo']}
- R² Score: {best['R²']:.4f} ({best['R²']*100:.1f}% da variância explicada)
- MAE: ${best['MAE ($)']:,.2f} (erro médio absoluto)
- WMAPE: {best['WMAPE (%)']:.2f}% (erro percentual)

CONCLUSÃO:
Os valores de R² relativamente baixos (< 0.3) são ESPERADOS para este problema,
pois a maior parte das variáveis com alta correlação com o custo foram removidas
por serem data leakage. Um R² superior a 0.9 indicaria que o modelo está a usar
informação que não estaria disponível em produção.

================================================================================
"""

with open("experiments/00_summary_for_report.txt", "w") as f:
    f.write(summary)

print("   → Resumo guardado em: experiments/00_summary_for_report.txt")
print(summary)
