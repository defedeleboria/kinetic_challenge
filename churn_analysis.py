"""
KINETIC CHALLENGE

# Run internal tests (with and without plots):
python churn_analysis.py --run-tests

# Run the full pipeline with data + RF + SHAP + plots:
python churn_analysis.py --data-dir ./data --plots --shap --model rf
"""

# ──────────────────────────────
# Imports
# ──────────────────────────────

from __future__ import annotations

import argparse
import importlib
import json
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore", category=FutureWarning)

# ──────────────────────────────
# CLI helpers
# ──────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Churn prediction workflow")
    p.add_argument("--data-dir", type=Path, default=Path("./data"), help="CSV directory")
    p.add_argument(
        "--outputs", type=Path, default=Path("./outputs"), help="Artefact folder"
    )
    p.add_argument(
        "--plots",
        action="store_true",
        help="Generate Matplotlib/Seaborn plots (if libs present)",
    )
    p.add_argument(
        "--shap",
        action="store_true",
        help="Generate SHAP summary plot (requires shap + matplotlib)",
    )
    p.add_argument(
        "--model",
        type=str,
        choices=["rf", "logreg", "xgb"],
        default="rf",
        help="Model to train: rf (RandomForest), logreg (LogReg), xgb (XGBoost)",
    )
    p.add_argument(
        "--self-test",
        action="store_true",
        help="Run pipeline only on synthetic data (ignores --data-dir)",
    )
    p.add_argument(
        "--run-tests",
        action="store_true",
        help="Run two smoke tests (with & without plots) and exit",
    )
    return p.parse_args()

# ──────────────────────────────
# Lazy imports
# ──────────────────────────────

def _lazy_import(libname: str):
    try:
        return importlib.import_module(libname)
    except ModuleNotFoundError:
        return None


def _lazy_import_plots():
    plt = _lazy_import("matplotlib.pyplot")
    sns = _lazy_import("seaborn")
    if plt is None or sns is None:
        warnings.warn(
            "matplotlib/seaborn not available → plots serán omitidos", RuntimeWarning
        )
    return plt, sns

# ──────────────────────────────
# Data loading helpers
# ──────────────────────────────

def _require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Archivo requerido no encontrado: {path}\n"
            "Coloca users.csv, usage_logs.csv y churn_labels.csv en --data-dir."
        )

def load_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    users_fp = data_dir / "users.csv"
    usage_fp = data_dir / "usage_logs.csv"
    churn_fp = data_dir / "churn_labels.csv"
    for fp in (users_fp, usage_fp, churn_fp):
        _require_file(fp)
    users = pd.read_csv(users_fp, parse_dates=["signup_date"])
    usage = pd.read_csv(usage_fp, parse_dates=["date"])
    churn = pd.read_csv(churn_fp)
    return users.merge(churn, on="user_id", how="left"), usage

# ──────────────────────────────
# Feature engineering
# ──────────────────────────────

def aggregate_usage(usage: pd.DataFrame) -> pd.DataFrame:
    agg = (
        usage.groupby("user_id")
        .agg(
            total_actions=("actions_performed", "sum"),
            total_time_spent=("time_spent_minutes", "sum"),
            total_documents_created=("documents_created", "sum"),
            total_logins=("logins", "sum"),
            days_active=("date", "nunique"),
        )
        .reset_index()
    )
    usage_sorted = usage.sort_values(["user_id", "date"])
    usage_sorted["roll_logins_7d"] = usage_sorted.groupby("user_id")["logins"].transform(
        lambda s: s.rolling(7, 1).mean()
    )
    roll_feat = (
        usage_sorted.groupby("user_id")["roll_logins_7d"]
        .mean()
        .reset_index()
        .rename(columns={"roll_logins_7d": "avg_roll_logins_7d"})
    )
    return agg.merge(roll_feat, on="user_id", how="left")

def engineer_features(users: pd.DataFrame, usage: pd.DataFrame) -> pd.DataFrame:
    data = users.merge(aggregate_usage(usage), on="user_id", how="left")
    activity_cols = [
        "total_actions",
        "total_time_spent",
        "total_documents_created",
        "total_logins",
        "days_active",
        "avg_roll_logins_7d",
    ]
    data[activity_cols] = data[activity_cols].fillna(0)
    data["days_since_signup"] = (usage["date"].max() - data["signup_date"]).dt.days
    return data

# ──────────────────────────────
# EDA
# ──────────────────────────────

def run_eda(users: pd.DataFrame, usage: pd.DataFrame, out_dir: Path, make_plots: bool):
    if not make_plots:
        return
    plt, sns = _lazy_import_plots()
    if plt is None or sns is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    merged = engineer_features(users, usage)

    # Histograma de tiempo total
    sns.histplot(
        data=merged, x="total_time_spent", hue="churned", bins=30, kde=True, stat="density"
    )
    plt.title("Distribución de tiempo total vs churn")
    plt.tight_layout()
    plt.savefig(out_dir / "eda_time_spent.png")
    plt.close()

    # Tendencia diaria de logins
    daily = usage.groupby("date")["logins"].mean()
    plt.figure(figsize=(8, 4))
    daily.plot()
    plt.title("Promedio de logins diarios")
    plt.xlabel("Fecha")
    plt.ylabel("Logins promedio")
    plt.tight_layout()
    plt.savefig(out_dir / "eda_daily_logins.png")
    plt.close()

# ──────────────────────────────
# Model building
# ──────────────────────────────

NUMERIC_FEATURES = [
    "total_actions",
    "total_time_spent",
    "total_documents_created",
    "total_logins",
    "days_active",
    "days_since_signup",
    "avg_roll_logins_7d",
]
CATEGORICAL_FEATURES = ["plan_type", "country"]

def build_model(name: str):
    if name == "logreg":
        return LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs")
    if name == "xgb":
        xgb = _lazy_import("xgboost")
        if xgb is not None:
            return xgb.XGBClassifier(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.9,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
            )
        warnings.warn("xgboost no instalado → usando RandomForest", RuntimeWarning)
    # default RandomForest
    return RandomForestClassifier(
        n_estimators=500, random_state=42, n_jobs=-1, class_weight="balanced"
    )

def build_pipeline(model_name: str) -> Pipeline:
    num_proc = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_proc = Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    preproc = ColumnTransformer(
        [("num", num_proc, NUMERIC_FEATURES), ("cat", cat_proc, CATEGORICAL_FEATURES)]
    )
    model = build_model(model_name)
    return Pipeline([("preprocessor", preproc), ("model", model)])

# ──────────────────────────────
# Temporal split
# ──────────────────────────────

def temporal_split(data: pd.DataFrame, cutoff: float = 0.8):
    split_date = data["signup_date"].quantile(cutoff)
    mask = data["signup_date"] < split_date
    X_train = data.loc[mask, NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = data.loc[mask, "churned"].astype(int)
    X_test = data.loc[~mask, NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_test = data.loc[~mask, "churned"].astype(int)
    return X_train, X_test, y_train, y_test

# ──────────────────────────────
# Evaluation & artefacts
# ──────────────────────────────

def _ensure_outputs(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def _save_metrics(metrics: Dict, out: Path):
    _ensure_outputs(out)
    with open(out / "metrics.json", "w", encoding="utf8") as f:
        json.dump(metrics, f, indent=2)

def evaluate(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    out: Path,
    plots: bool,
    shap_flag: bool,
):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    auc = roc_auc_score(y_test, y_prob)
    metrics: Dict[str, float] = {
        "precision": round(float(prec), 3),
        "recall":    round(float(rec), 3),
        "f1":        round(float(f1), 3),
        "roc_auc":   round(float(roc_auc_score(y_test, y_prob)), 3),
    }
    print("\n--- Métricas test ---")
    print(json.dumps(metrics, indent=2))

    _save_metrics(metrics, out)
    _ensure_outputs(out)
    joblib.dump(model, out / "model.joblib")

    plt, sns = _lazy_import_plots()
    if plots and plt is not None and sns is not None:
        # ROC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={auc:.2f}")
        plt.plot([0, 1], [0, 1], "--")
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve"); plt.legend()
        plt.tight_layout(); plt.savefig(out / "roc_curve.png"); plt.close()

        # Feature importance (si aplica)
        if hasattr(model.named_steps["model"], "feature_importances_"):
            ohe = model.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
            feat_names = NUMERIC_FEATURES + list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
            imp = model.named_steps["model"].feature_importances_
            sns.barplot(
                x="importance",
                y="feature",
                data=pd.DataFrame({"feature": feat_names, "importance": imp}).nlargest(15, "importance"),
                orient="h",
            )
            plt.title("Top 15 Feature Importances")
            plt.tight_layout(); plt.savefig(out / "feature_importance.png"); plt.close()

    # SHAP interpretabilidad
    if shap_flag:
        shap = _lazy_import("shap")
        if shap is None:
            warnings.warn("shap no instalado → se omite SHAP summary", RuntimeWarning)
        elif plt is None:
            warnings.warn("matplotlib requerido para SHAP plots", RuntimeWarning)
        else:
            explainer = shap.TreeExplainer(model.named_steps["model"])
            shap_values = explainer.shap_values(model.named_steps["preprocessor"].transform(X_test))
            shap.summary_plot(shap_values, feature_names=NUMERIC_FEATURES + list(
                model.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(
                    CATEGORICAL_FEATURES
                )
            ), show=False)
            plt.tight_layout(); plt.savefig(out / "shap_summary.png"); plt.close()

    return metrics

# ──────────────────────────────
# Synthetic dataset helper
# ──────────────────────────────

def _make_synthetic_dataset(tmpdir: Path):
    rng = np.random.default_rng(42)
    num_users = 20
    users = pd.DataFrame(
        {
            "user_id": np.arange(1, num_users + 1),
            "signup_date": pd.Timestamp("2025-01-01") + pd.to_timedelta(
                rng.integers(0, 30, num_users), unit="D"
            ),
            "plan_type": rng.choice(["free", "basic", "premium"], num_users),
            "country": rng.choice(["US", "ES", "UK", "AR"], num_users),
        }
    )
    users["churned"] = rng.choice([0, 1], num_users, p=[0.7, 0.3])

    records = []
    for uid in users["user_id"]:
        for day in range(30):
            records.append(
                {
                    "user_id": uid,
                    "date": pd.Timestamp("2025-02-01") + pd.Timedelta(days=day),
                    "actions_performed": rng.integers(0, 20),
                    "time_spent_minutes": rng.integers(0, 120),
                    "documents_created": rng.integers(0, 5),
                    "logins": rng.integers(0, 4),
                }
            )
    usage_logs = pd.DataFrame(records)

    users[["user_id", "signup_date", "plan_type", "country"]].to_csv(
        tmpdir / "users.csv", index=False
    )
    usage_logs.to_csv(tmpdir / "usage_logs.csv", index=False)
    users[["user_id", "churned"]].to_csv(tmpdir / "churn_labels.csv", index=False)
    return tmpdir

# ──────────────────────────────
# Test suite
# ──────────────────────────────

def _run_smoke_test(make_plots: bool):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _make_synthetic_dataset(tmp_path)
        users, usage = load_data(tmp_path)

        data = engineer_features(users, usage)
        X_train, X_test, y_train, y_test = temporal_split(data)
        model = build_pipeline("rf")
        model.fit(X_train, y_train)
        metrics = evaluate(
            model, X_test, y_test, tmp_path / "out", plots=make_plots, shap_flag=False
        )
        assert metrics["roc_auc"] > 0.5, "AUC should be better than random"
        print(
            f"Smoke test passed (plots={'on' if make_plots else 'off'}) — AUC {metrics['roc_auc']}"
        )

# ──────────────────────────────
# Main entry
# ──────────────────────────────

def main():
    args = parse_args()

    if args.run_tests:
        _run_smoke_test(False)
        _run_smoke_test(True)
        print("✔️  All internal smoke tests passed")
        sys.exit(0)

    # Data loading
    if args.self_test:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = _make_synthetic_dataset(Path(tmp))
            users, usage = load_data(data_dir)
    else:
        users, usage = load_data(args.data_dir)

    # Exploratory analysis
    run_eda(users, usage, args.outputs / "eda", args.plots)

    # Feature engineering & split
    data = engineer_features(users, usage)
    X_train, X_test, y_train, y_test = temporal_split(data)

    # Model
    pipeline = build_pipeline(args.model)
    pipeline.fit(X_train, y_train)

    # Evaluation & artefacts
    evaluate(pipeline, X_test, y_test, args.outputs, args.plots, args.shap)

    print("ANALYSIS FINISH!")


if __name__ == "__main__":
    main()
