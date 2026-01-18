import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
try:
    from sklearn.ensemble import HistGradientBoostingRegressor
except Exception:
    HistGradientBoostingRegressor = None
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import joblib

# ===========================
# Configuracao basica
# ===========================
SEED = 42
np.random.seed(SEED)

RISK_VERSION = "v1"
OUTPUT_DIR = os.path.join("risk_models", RISK_VERSION)
os.makedirs(OUTPUT_DIR, exist_ok=True)

RUN_COST_MODEL = True
COST_TARGET_COLUMN = "annual_medical_cost"
CONFIDENCE_TOLERANCE = 0.20
USE_ONLY_SAFE_FEATURES = True
USE_LOG_TARGET = True
COST_MODEL_TYPE = "hist_gb"  # options: "hist_gb", "nn"

RESULTS_FILE = "risk_classes_experiments.csv"
VERSIONED_RESULTS_FILE = os.path.join(OUTPUT_DIR, "risk_classes_experiments.csv")
METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata.json")
GLOBAL_PREPROCESS_FILE = os.path.join(OUTPUT_DIR, "preprocess_global.pkl")
CLINICAL_PREPROCESS_FILE = os.path.join(OUTPUT_DIR, "preprocess_clinical.pkl")
COST_PREPROCESS_FILE = os.path.join(OUTPUT_DIR, "preprocess_cost.pkl")
SOCIO_PREPROCESS_FILE = os.path.join(OUTPUT_DIR, "preprocess_socio.pkl")
COST_MODEL_FILE = os.path.join(
    OUTPUT_DIR,
    f"cost_model_{COST_MODEL_TYPE}" + (".keras" if COST_MODEL_TYPE == "nn" else ".pkl")
)
COST_MODEL_PREPROCESS_FILE = os.path.join(OUTPUT_DIR, "preprocess_cost_model.pkl")
COST_MODEL_DROPPED_FILE = os.path.join(OUTPUT_DIR, "cost_model_dropped_columns.json")

# ===========================
# 1) Carregar dados
# ===========================
data_frame = pd.read_excel("ECF_2.xlsx")
target_column = "risk_score"

# ===========================
# Grupos de colunas (podes ajustar se necessario)
# ===========================
clinical_columns = [
    "age", "sex", "bmi", "smoker", "alcohol_freq",
    "visits_last_year", "hospitalizations_last_3yrs",
    "days_hospitalized_last_3yrs", "medication_count",
    "systolic_bp", "diastolic_bp", "ldl", "hba1c",
    "chronic_count", "hypertension", "diabetes", "asthma",
    "copd", "cardiovascular_disease", "cancer_history",
    "kidney_disease", "liver_disease", "arthritis", "mental_health",
    "proc_imaging_count", "proc_surgery_count", "proc_physio_count",
    "proc_consult_count", "proc_lab_count", "is_high_risk",
    "had_major_procedure"
]

cost_columns = [
    "plan_type", "network_tier", "deductible", "copay",
    "policy_term_years", "policy_changes_last_2yrs", "provider_quality",
    "annual_medical_cost", "annual_premium", "monthly_premium",
    "claims_count", "avg_claim_amount", "total_claims_paid"
]

socio_columns = [
    "region", "urban_rural", "income", "education", "marital_status",
    "employment_status", "household_size", "dependents"
]

# ===========================
# 2) Remover colunas de leakage
# ===========================
leakage_columns = [
    "is_high_risk",
    "monthly_premium",
    "annual_premium",
    "avg_claim_amount",
    "total_claims_paid"
]

columns_to_drop = [target_column, "person_id"] + leakage_columns
dropped_columns = [c for c in columns_to_drop if c in data_frame.columns]

features_df = data_frame.drop(columns=columns_to_drop, errors="ignore")

categorical_columns = features_df.select_dtypes(include=["object"]).columns
numeric_columns = features_df.select_dtypes(exclude=["object"]).columns

print("Categorical columns:", list(categorical_columns))
print("Numeric columns:", list(numeric_columns))

# ===========================
# 3) Split antes do preprocessamento
# ===========================
train_index, _ = train_test_split(
    features_df.index, test_size=0.2, random_state=SEED
)
train_df = features_df.loc[train_index]

# ===========================
# Funcoes de apoio
# ===========================
def available_columns(columns, reference_columns):
    return [c for c in columns if c in reference_columns]

def build_preprocessor(train_subset):
    cat_cols = train_subset.select_dtypes(include=["object"]).columns
    num_cols = train_subset.select_dtypes(exclude=["object"]).columns
    transformers = []
    if len(cat_cols) > 0:
        transformers.append(
            ("categorical", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        )
    if len(num_cols) > 0:
        transformers.append(("numeric", StandardScaler(), num_cols))
    if not transformers:
        return None
    preprocessor = ColumnTransformer(transformers=transformers)
    preprocessor.fit(train_subset)
    return preprocessor

def compute_scores(preprocessor, train_subset, full_subset):
    train_processed = preprocessor.transform(train_subset)
    full_processed = preprocessor.transform(full_subset)
    train_processed = np.asarray(train_processed, dtype="float32")
    full_processed = np.asarray(full_processed, dtype="float32")
    train_score = np.array(train_processed.sum(axis=1)).ravel()
    full_score = np.array(full_processed.sum(axis=1)).ravel()
    return train_score, full_score

def quantile_thresholds(score_values):
    q1, q2 = np.quantile(score_values, [1 / 3, 2 / 3])
    if q1 == q2:
        q2 = q1 + 1e-6
    return float(q1), float(q2)

def apply_classes(score_values, thresholds, labels=("low", "medium", "high")):
    q1, q2 = thresholds
    return pd.cut(
        score_values,
        bins=[-np.inf, q1, q2, np.inf],
        labels=labels,
        include_lowest=True
    )

def add_composite_risk(method_name, columns, preprocessor_path=None):
    available = available_columns(columns, features_df.columns)
    if not available:
        risk_results[method_name] = "unknown"
        return None, []
    train_subset = train_df[available]
    full_subset = features_df[available]
    preprocessor = build_preprocessor(train_subset)
    if preprocessor is None:
        risk_results[method_name] = "unknown"
        return None, available
    train_score, full_score = compute_scores(
        preprocessor, train_subset, full_subset
    )
    thresholds = quantile_thresholds(train_score)
    risk_results[method_name] = apply_classes(full_score, thresholds)
    if preprocessor_path:
        joblib.dump(preprocessor, preprocessor_path)
    return thresholds, available

# ===========================
# 4) Preprocessamento global
# ===========================
global_preprocessor = build_preprocessor(train_df)
if global_preprocessor is None:
    raise ValueError("Nao existem colunas suficientes para calcular o risco.")

global_train_processed = global_preprocessor.transform(train_df)
global_processed = global_preprocessor.transform(features_df)
global_train_processed = np.asarray(global_train_processed, dtype="float32")
global_processed = np.asarray(global_processed, dtype="float32")

joblib.dump(global_preprocessor, GLOBAL_PREPROCESS_FILE)

# ===========================
# 5) Calculo de risco em classes
# ===========================
risk_results = pd.DataFrame(index=features_df.index)
if "person_id" in data_frame.columns:
    risk_results["person_id"] = data_frame["person_id"]

metadata = {
    "seed": SEED,
    "columns": {
        "dropped": dropped_columns,
        "categorical": list(categorical_columns),
        "numeric": list(numeric_columns)
    },
    "thresholds": {},
    "kmeans_cluster_map": {}
}

# 5.1) Soma padronizada global
global_train_score = np.array(global_train_processed.sum(axis=1)).ravel()
global_score = np.array(global_processed.sum(axis=1)).ravel()
global_thresholds = quantile_thresholds(global_train_score)
risk_results["risk_class_global_sum"] = apply_classes(
    global_score, global_thresholds
)
metadata["thresholds"]["global_sum"] = list(global_thresholds)

# 5.2) PCA no primeiro componente
pca_model = PCA(n_components=1, random_state=SEED)
pca_model.fit(global_train_processed)
pca_train_score = pca_model.transform(global_train_processed).ravel()
pca_score = pca_model.transform(global_processed).ravel()
pca_thresholds = quantile_thresholds(pca_train_score)
risk_results["risk_class_pca1"] = apply_classes(pca_score, pca_thresholds)
metadata["thresholds"]["pca1"] = list(pca_thresholds)

# 5.3) KMeans (3 clusters) ordenados por risco medio
kmeans_model = KMeans(n_clusters=3, random_state=SEED, n_init=10)
kmeans_model.fit(global_train_processed)
cluster_labels = kmeans_model.predict(global_processed)
cluster_centers = kmeans_model.cluster_centers_
cluster_risk = cluster_centers.mean(axis=1)
order = np.argsort(cluster_risk)
cluster_map = {int(order[0]): "low", int(order[1]): "medium", int(order[2]): "high"}
risk_results["risk_class_kmeans"] = (
    pd.Series(cluster_labels, index=features_df.index).map(cluster_map).values
)
metadata["kmeans_cluster_map"] = {str(k): v for k, v in cluster_map.items()}

# 5.4) Score de utilizacao clinica
utilization_columns = available_columns([
    "visits_last_year", "hospitalizations_last_3yrs",
    "days_hospitalized_last_3yrs", "medication_count",
    "proc_imaging_count", "proc_surgery_count",
    "proc_physio_count", "proc_consult_count", "proc_lab_count"
], features_df.columns)

util_thresholds, util_used = add_composite_risk(
    "risk_class_utilization", utilization_columns
)
metadata["thresholds"]["utilization"] = (
    list(util_thresholds) if util_thresholds else None
)
metadata["columns"]["utilization_used"] = util_used

# 5.5) Score clinico
clinical_thresholds, clinical_used = add_composite_risk(
    "risk_class_clinical", clinical_columns, CLINICAL_PREPROCESS_FILE
)
metadata["thresholds"]["clinical"] = (
    list(clinical_thresholds) if clinical_thresholds else None
)
metadata["columns"]["clinical_used"] = clinical_used

# 5.6) Score de custo
cost_thresholds, cost_used = add_composite_risk(
    "risk_class_cost", cost_columns, COST_PREPROCESS_FILE
)
metadata["thresholds"]["cost"] = (
    list(cost_thresholds) if cost_thresholds else None
)
metadata["columns"]["cost_used"] = cost_used

# 5.7) Score socioeconomico
socio_thresholds, socio_used = add_composite_risk(
    "risk_class_socio", socio_columns, SOCIO_PREPROCESS_FILE
)
metadata["thresholds"]["socio"] = (
    list(socio_thresholds) if socio_thresholds else None
)
metadata["columns"]["socio_used"] = socio_used

# ===========================
# 6) Guardar resultados
# ===========================
risk_results.to_csv(RESULTS_FILE, index=False)
risk_results.to_csv(VERSIONED_RESULTS_FILE, index=False)

with open(METADATA_FILE, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=True, indent=2)

print("\n==============================")
print(" RISK CLASS EXPERIMENTS (NO risk_score)")
print("==============================\n")
print(risk_results.head(10))
print(f"\nFicheiro criado: {RESULTS_FILE}")
print(f"Versao guardada em: {VERSIONED_RESULTS_FILE}")

# ===========================
# 7) Modelo supervisionado para custo (opcional)
# ===========================
if RUN_COST_MODEL:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        plotting_available = True
    except Exception as exc:
        print(f"\nAviso: sem graficos (matplotlib/seaborn): {exc}")
        plotting_available = False
        plt = None
        sns = None

    if COST_TARGET_COLUMN not in data_frame.columns:
        print(
            f"\nAviso: coluna {COST_TARGET_COLUMN} nao encontrada, a parte "
            "de custo foi ignorada."
        )
    else:
        # Colunas removidas para evitar leakage ao prever custo anual
        cost_drop_columns = [
            COST_TARGET_COLUMN,
            "risk_score",
            "person_id",
            "is_high_risk",
            "monthly_premium",
            "annual_premium",
            "avg_claim_amount",
            "claims_count",
            "total_claims_paid",
            "copay",
            "plan_type",
            "network_tier",
            "deductible",
            "policy_term_years",
            "policy_changes_last_2yrs",
            "provider_quality",
        ]

        if USE_ONLY_SAFE_FEATURES:
            safe_columns = [
                c for c in (clinical_columns + socio_columns)
                if c in data_frame.columns and c not in cost_drop_columns
            ]
            if not safe_columns:
                raise ValueError(
                    "Nao existem colunas seguras suficientes para o modelo."
                )
            cost_features = data_frame[safe_columns].copy()
            cost_dropped = [
                c for c in data_frame.columns
                if c not in safe_columns + [COST_TARGET_COLUMN]
            ]
        else:
            cost_features = data_frame.drop(
                columns=cost_drop_columns, errors="ignore"
            )
            cost_dropped = [
                c for c in cost_drop_columns if c in data_frame.columns
            ]

        cost_target = data_frame[COST_TARGET_COLUMN].astype(float)

        cost_cat_columns = cost_features.select_dtypes(
            include=["object"]
        ).columns
        cost_num_columns = cost_features.select_dtypes(
            exclude=["object"]
        ).columns

        print("\nCost model categorical columns:", list(cost_cat_columns))
        print("Cost model numeric columns:", list(cost_num_columns))

        X_train_orig, X_test_orig, y_train_raw, y_test_raw = train_test_split(
            cost_features, cost_target, test_size=0.2, random_state=SEED
        )

        if USE_LOG_TARGET and (y_train_raw < 0).any():
            print("\nAviso: valores negativos no target, log1p ignorado.")
            use_log_target = False
        else:
            use_log_target = USE_LOG_TARGET

        if use_log_target:
            y_train = np.log1p(y_train_raw)
            y_test = np.log1p(y_test_raw)
        else:
            y_train = y_train_raw.copy()
            y_test = y_test_raw.copy()

        cost_preprocessor = ColumnTransformer(
            transformers=[
                ("categorical",
                 Pipeline([
                     ("imputer",
                      SimpleImputer(strategy="most_frequent")),
                     ("onehot",
                      OneHotEncoder(handle_unknown="ignore")),
                 ]),
                 cost_cat_columns),
                ("numeric",
                 Pipeline([
                     ("imputer", SimpleImputer(strategy="median")),
                     ("scaler", StandardScaler()),
                 ]),
                 cost_num_columns),
            ]
        )

        cost_preprocessor.fit(X_train_orig)
        X_train = cost_preprocessor.transform(X_train_orig)
        X_test = cost_preprocessor.transform(X_test_orig)
        if hasattr(X_train, "toarray"):
            X_train = X_train.toarray()
            X_test = X_test.toarray()
        X_train = np.asarray(X_train, dtype="float32")
        X_test = np.asarray(X_test, dtype="float32")

        print("Cost train shape:", X_train.shape)
        print("Cost test shape:", X_test.shape)

        model_available = True
        history = None

        if COST_MODEL_TYPE == "nn":
            try:
                import tensorflow as tf
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import Dense, Dropout
                from tensorflow.keras.callbacks import EarlyStopping
            except Exception as exc:
                print(
                    f"\nAviso: TensorFlow indisponivel para o modelo NN: {exc}"
                )
                model_available = False
            else:
                tf.random.set_seed(SEED)
                model = Sequential([
                    Dense(256, activation="relu",
                          input_shape=(X_train.shape[1],)),
                    Dropout(0.3),
                    Dense(128, activation="relu"),
                    Dropout(0.2),
                    Dense(64, activation="relu"),
                    Dense(1, activation="linear"),
                ])

                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss="mse",
                    metrics=["mae"],
                )

                early_stop = EarlyStopping(
                    monitor="val_loss",
                    patience=20,
                    restore_best_weights=True,
                )

                history = model.fit(
                    X_train,
                    y_train,
                    validation_split=0.2,
                    epochs=200,
                    batch_size=32,
                    callbacks=[early_stop],
                    verbose=1,
                )

                preds_raw = model.predict(X_test).flatten()
        elif COST_MODEL_TYPE == "hist_gb":
            if HistGradientBoostingRegressor is None:
                print("\nAviso: HistGradientBoostingRegressor indisponivel.")
                model_available = False
            else:
                model = HistGradientBoostingRegressor(
                    loss="squared_error",
                    learning_rate=0.05,
                    max_depth=6,
                    max_leaf_nodes=31,
                    min_samples_leaf=20,
                    l2_regularization=0.1,
                    max_iter=500,
                    random_state=SEED,
                )
                model.fit(X_train, y_train)
                preds_raw = model.predict(X_test)
        else:
            print(f"\nAviso: tipo de modelo desconhecido: {COST_MODEL_TYPE}")
            model_available = False

        if model_available:
            if use_log_target:
                preds = np.expm1(preds_raw)
                y_test_eval = y_test_raw.values
            else:
                preds = preds_raw
                y_test_eval = y_test_raw.values

            preds = np.clip(preds, 0, None)

            mae = mean_absolute_error(y_test_eval, preds)
            mse = mean_squared_error(y_test_eval, preds)
            rmse = np.sqrt(mse)
            medae = median_absolute_error(y_test_eval, preds)

            y_max = y_test_eval.max()
            y_min = y_test_eval.min()
            y_range = y_max - y_min
            y_mean = y_test_eval.mean()

            nrmse_range = rmse / y_range if y_range != 0 else np.nan
            cv_rmse = rmse / y_mean if y_mean != 0 else np.nan
            r2 = r2_score(y_test_eval, preds)

            sum_abs_error = np.sum(np.abs(y_test_eval - preds))
            sum_actual = np.sum(y_test_eval)
            wmape = (sum_abs_error / sum_actual) * 100 if sum_actual != 0 else np.nan
            abs_error = np.abs(y_test_eval - preds)
            abs_real = np.abs(y_test_eval)
            valid_mask = abs_real > 0
            if valid_mask.any():
                relative_error = abs_error[valid_mask] / abs_real[valid_mask]
                confidence_percent = (
                    (relative_error <= CONFIDENCE_TOLERANCE).mean() * 100
                )
            else:
                confidence_percent = np.nan

            print(f"\n{'Metrica':<25} | {'Valor':<12} | {'Interpretacao'}")
            print("-" * 70)
            print(
                f"{'MAE (Erro Medio Abs)':<25} | {mae:>10.2f} | "
                "Erro medio no custo anual"
            )
            print(
                f"{'RMSE (Erro Quadratico)':<25} | {rmse:>10.2f} | "
                "Penaliza erros grandes"
            )
            print(
                f"{'MedAE (Erro Mediano)':<25} | {medae:>10.2f} | "
                "Erro tipico sem outliers"
            )
            print(
                f"{'CV-RMSE (% da Media)':<25} | {cv_rmse * 100:>9.2f}% | "
                "Erro relativo a media"
            )
            print(
                f"{'R2 Score':<25} | {r2:>10.4f} | "
                "Variancia explicada"
            )
            print("-" * 70)
            print(
                f"{'NRMSE (% do Range)':<25} | {nrmse_range * 100:>9.2f}% | "
                "Erro relativo ao range"
            )
            print(
                f"{'WMAPE (Erro Total)':<25} | {wmape:>9.2f}% | "
                "Erro relativo a soma total"
            )
            print("-" * 70)
            print(
                f"Confianca do modelo (±{int(CONFIDENCE_TOLERANCE * 100)}%): "
                f"{confidence_percent:.2f}%"
            )
            print(
                "Interpretacao: percentagem de previsoes cuja diferenca "
                "relativa ao valor real fica dentro da tolerancia."
            )

            residuals = y_test_eval - preds

            if plotting_available:
                plt.figure(figsize=(14, 5))

                plt.subplot(1, 2, 1)
                plt.scatter(y_test_eval, preds, alpha=0.5, color="royalblue")
                plt.plot([y_min, y_max], [y_min, y_max], "r--", lw=2)
                plt.xlabel("Valor Real (Annual Medical Cost)")
                plt.ylabel("Valor Previsto")
                plt.title("Real vs Previsto (Ideal = Linha Vermelha)")
                plt.grid(True, alpha=0.3)

                plt.subplot(1, 2, 2)
                sns.histplot(residuals, kde=True, color="purple", bins=30)
                plt.axvline(0, color="r", linestyle="--")
                plt.xlabel("Erro (Residuo)")
                plt.title("Distribuicao dos Erros (Ideal = Centrado no 0)")
                plt.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.show()

            print("\n===============================================")
            print(" PERFORMANCE POR GRUPO (CATEGORICAL ANALYSIS)")
            print("===============================================\n")

            analysis_df = X_test_orig.copy()
            analysis_df["Real"] = y_test_eval
            analysis_df["Predicted"] = preds
            analysis_df["Abs_Error"] = np.abs(
                analysis_df["Real"] - analysis_df["Predicted"]
            )

            for col in cost_cat_columns:
                if col in analysis_df.columns:
                    print(f"--- Analise por: {col.upper()} ---")
                    group_metrics = analysis_df.groupby(col).agg(
                        Count=("Real", "count"),
                        MAE=("Abs_Error", "mean"),
                        Mean_Real=("Real", "mean"),
                    ).sort_values(by="MAE", ascending=False)
                    print(group_metrics)
                    print("-" * 50)
                    print("\n")

            if plotting_available and history is not None:
                plt.figure(figsize=(10, 5))
                plt.plot(history.history["loss"], label="Training Loss")
                plt.plot(history.history["val_loss"], label="Validation Loss")
                plt.xlabel("Epochs")
                plt.ylabel("Loss (MSE)")
                plt.title("Training vs Validation Loss")
                plt.legend()
                plt.grid()
                plt.show()

            results_df = pd.DataFrame({
                "Real": y_test_eval,
                "Predicted": preds,
                "Absolute_Error": np.abs(y_test_eval - preds),
            })
            print("\nPrimeiras 20 previsoes:\n")
            print(results_df.head(20))

            if COST_MODEL_TYPE == "nn":
                model.save(COST_MODEL_FILE)
            else:
                joblib.dump(model, COST_MODEL_FILE)
            joblib.dump(cost_preprocessor, COST_MODEL_PREPROCESS_FILE)

            with open(COST_MODEL_DROPPED_FILE, "w", encoding="utf-8") as f:
                json.dump(cost_dropped, f, ensure_ascii=True)

            print("\nModelo de custo guardado.")
