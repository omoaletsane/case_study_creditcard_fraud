#!/usr/bin/env python3
"""
Credit Card Fraud Detection — end-to-end pipeline (modular, testable)

Features:
  - Load dataset from CREDITCARD_CSV env var or ./creditcard.csv
  - EDA: histograms (Amount, Time), class imbalance bar, V1–V28 correlation heatmap
  - Stratified 80/20 split
  - Pipelines: StandardScaler → SMOTE → Classifier (LR / RF)
  - Hyperparameter tuning: LR via GridSearchCV, RF via RandomizedSearchCV (5-fold, scoring=F1)
  - Threshold optimization per model (maximize F1 via CV on train)
  - Final test evaluation at fixed 0.50 and best-F1 thresholds
  - Plots: PR curves, confusion matrices, comparison bars
  - Tables: CSV exports under ./outputs

Refactor:
  - Exposes reusable functions for unit/integration tests
  - Safe matplotlib backend (Agg) for CI
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import Dict, Tuple

# Safe, headless plotting for CI and servers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

from scipy.stats import randint

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_predict,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    auc,
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# ----------------------- Config & Utilities -----------------------

RANDOM_STATE = 42
OUT_DIR = "outputs"
V_COLS = [f"V{i}" for i in range(1, 29)]

warnings.filterwarnings("ignore")


def ensure_outputs_dir(path: str = OUT_DIR) -> None:
    """Create outputs directory if it doesn't exist."""
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def savefig(path: str, fig=None, tight: bool = True) -> None:
    """Save current or provided matplotlib figure."""
    if fig is None:
        fig = plt.gcf()
    if tight:
        plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"[saved] {path}")


def table_to_csv(df: pd.DataFrame, filename: str, out_dir: str = OUT_DIR) -> str:
    path = os.path.join(out_dir, filename)
    df.to_csv(path, index=False)
    print(f"[saved] {path}")
    return path


def get_csv_path(default: str = "creditcard.csv") -> str:
    return os.environ.get("CREDITCARD_CSV", default)


# ----------------------- Data I/O -----------------------

def load_data(path: str | None = None) -> pd.DataFrame:
    """Load dataset; expects Time, Amount, V1..V28, Class."""
    if path is None:
        path = get_csv_path()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Set env var CREDITCARD_CSV or place creditcard.csv in the working directory."
        )
    df = pd.read_csv(path)
    # Defensive checks
    expected = {"Time", "Amount", "Class"}
    if not expected.issubset(df.columns):
        missing = expected - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")
    return df


# ----------------------- EDA Plots -----------------------

def plot_class_imbalance(df: pd.DataFrame, out_dir: str = OUT_DIR) -> pd.DataFrame:
    counts = (
        df["Class"]
        .map({0: "Legit", 1: "Fraud"})
        .value_counts()
        .rename_axis("Class")
        .reset_index(name="Count")
    )
    counts["Percent"] = (counts["Count"] / counts["Count"].sum() * 100.0).round(4)
    table_to_csv(counts, "class_distribution.csv", out_dir)

    plt.figure(figsize=(6, 4))
    ax = sns.barplot(data=counts, x="Class", y="Count", hue="Class", dodge=False)
    for i, row in counts.iterrows():
        ax.text(i, row["Count"] * 1.01, f"{row['Count']:,}", ha="center")
    plt.title("Class Imbalance: Legit vs Fraud")
    plt.ylabel("Number of Transactions")

    # Remove legend only if one was created
    leg = plt.gca().get_legend()
    if leg is not None:
        leg.remove()

    savefig(os.path.join(out_dir, "class_imbalance_bar.png"))
    plt.close()
    return counts


def plot_histograms(df: pd.DataFrame, out_dir: str = OUT_DIR) -> None:
    for col, bins, fname in [("Amount", 60, "hist_amount.png"), ("Time", 60, "hist_time.png")]:
        plt.figure(figsize=(7, 4))
        sns.histplot(df[col], bins=bins, kde=False)
        plt.title(f"Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        savefig(os.path.join(out_dir, fname))
        plt.close()


def plot_vcorr_heatmap(df: pd.DataFrame, out_dir: str = OUT_DIR) -> None:
    missing = [c for c in V_COLS if c not in df.columns]
    if missing:
        print(f"[warn] Missing PCA columns; skipping heatmap. Missing: {missing}")
        return
    corr = df[V_COLS].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True, cbar_kws={"shrink": 0.7})
    plt.title("Correlation Heatmap: V1–V28")
    savefig(os.path.join(out_dir, "heatmap_v1_v28.png"))
    plt.close()


# ----------------------- Split & Pipelines -----------------------

def stratified_split(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, ...]:
    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )


def build_pipelines() -> Tuple[ImbPipeline, ImbPipeline]:
    scaler = StandardScaler(with_mean=True, with_std=True)
    smote = SMOTE(random_state=RANDOM_STATE, sampling_strategy="auto", k_neighbors=5)

    pipe_lr = ImbPipeline(
        steps=[
            ("scaler", scaler),
            ("smote", smote),
            ("clf", LogisticRegression(max_iter=2000, solver="liblinear", random_state=RANDOM_STATE)),
        ]
    )

    pipe_rf = ImbPipeline(
        steps=[
            ("scaler", scaler),
            ("smote", smote),
            ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
        ]
    )
    return pipe_lr, pipe_rf


# ----------------------- Hyperparameter Tuning -----------------------

def tune_models(
    pipe_lr: ImbPipeline,
    pipe_rf: ImbPipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter_rf: int = 25,
) -> Tuple[GridSearchCV, RandomizedSearchCV]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Logistic Regression: Grid
    param_grid_lr = {
        "clf__penalty": ["l1", "l2"],
        "clf__C": [0.01, 0.1, 1.0, 10.0, 100.0],
    }
    search_lr = GridSearchCV(
        estimator=pipe_lr,
        param_grid=param_grid_lr,
        scoring="f1",
        n_jobs=-1,
        cv=cv,
        verbose=1,
    )

    # Random Forest: Randomized
    param_dist_rf = {
        "clf__n_estimators": randint(300, 900),
        "clf__max_depth": [None, 5, 10, 20, 30],
        "clf__min_samples_split": randint(2, 11),
        "clf__min_samples_leaf": randint(1, 5),
        "clf__max_features": ["sqrt", "log2", 0.3, 0.5, 0.8],
    }
    search_rf = RandomizedSearchCV(
        estimator=pipe_rf,
        param_distributions=param_dist_rf,
        n_iter=n_iter_rf,
        scoring="f1",
        n_jobs=-1,
        cv=cv,
        random_state=RANDOM_STATE,
        verbose=1,
    )

    print("Tuning Logistic Regression …")
    search_lr.fit(X_train, y_train)
    print("Tuning Random Forest …")
    search_rf.fit(X_train, y_train)

    return search_lr, search_rf


# ----------------------- Threshold Optimization -----------------------

def find_best_threshold_cv(
    estimator, X_train: pd.DataFrame, y_train: pd.Series
) -> Tuple[float, float]:
    """
    Cross-validated predicted probabilities on TRAIN to pick the threshold that maximizes F1.
    Returns (best_threshold, pr_auc_cv).
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    proba = cross_val_predict(
        estimator, X_train, y_train, cv=cv, n_jobs=-1, method="predict_proba"
    )[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_train, proba)
    f1s = 2 * (precision * recall) / (precision + recall + 1e-12)
    best_idx = np.nanargmax(f1s)
    # thresholds is length-1 of precision/recall; guard if best_idx is out of range
    best_th = 0.5 if best_idx >= len(thresholds) else float(thresholds[best_idx])
    pr_auc_val = auc(recall, precision)
    return best_th, pr_auc_val


# ----------------------- Evaluation & Plots -----------------------

def eval_at_threshold(y_true: np.ndarray, proba: np.ndarray, th: float) -> Tuple[float, float, float, np.ndarray]:
    y_pred = (proba >= th).astype(int)
    P = precision_score(y_true, y_pred, zero_division=0)
    R = recall_score(y_true, y_pred, zero_division=0)
    F1 = f1_score(y_true, y_pred, zero_division=0)
    CM = confusion_matrix(y_true, y_pred)
    return P, R, F1, CM


def plot_pr_curves(
    y_test: np.ndarray,
    proba_lr: np.ndarray,
    proba_rf: np.ndarray,
    out_dir: str = OUT_DIR,
) -> None:
    prec_lr, rec_lr, _ = precision_recall_curve(y_test, proba_lr)
    prec_rf, rec_rf, _ = precision_recall_curve(y_test, proba_rf)

    plt.figure(figsize=(7, 6))
    plt.plot(rec_lr, prec_lr, label=f"LR (PR-AUC={auc(rec_lr, prec_lr):.4f})")
    plt.plot(rec_rf, rec_rf * 0 + prec_rf, alpha=0)  # noop to keep axes consistent
    plt.plot(rec_rf, prec_rf, label=f"RF (PR-AUC={auc(rec_rf, prec_rf):.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curves (Test)")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc="lower left")
    savefig(os.path.join(out_dir, "pr_curves_test.png"))
    plt.close()


def plot_conf_mx(cm: np.ndarray, title: str, outpath: str) -> None:
    plt.figure(figsize=(5.8, 4.8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        xticklabels=["Pred Legit", "Pred Fraud"],
        yticklabels=["Actual Legit", "Actual Fraud"],
    )
    plt.title(title)
    savefig(outpath)
    plt.close()


def comparison_bar_chart(
    metrics_050: pd.DataFrame, metrics_best: pd.DataFrame, out_dir: str = OUT_DIR
) -> None:
    comp_tbl = pd.concat(
        [metrics_050.assign(Setting="Threshold 0.50"), metrics_best.assign(Setting="Best F1 Threshold")],
        ignore_index=True,
    )
    table_to_csv(comp_tbl, "comparison_metrics_table.csv", out_dir)

    plot_df = comp_tbl.melt(
        id_vars=["Model", "Threshold", "Setting"],
        value_vars=["Precision", "Recall", "F1"],
        var_name="Metric",
        value_name="Score",
    )
    plt.figure(figsize=(8.5, 5.5))
    sns.barplot(data=plot_df, x="Metric", y="Score", hue="Model")
    plt.ylim(0, 1.05)
    plt.title("Model Comparison: Precision, Recall, F1 (by setting)")
    plt.xlabel("")
    plt.ylabel("Score")
    savefig(os.path.join(out_dir, "compare_precision_recall_f1.png"))
    plt.close()


# ----------------------- Main workflow -----------------------

def main() -> Dict[str, object]:
    """
    Runs the full pipeline and returns a dict of key artifacts for tests.
    Returned keys:
      - 'best_table' (pd.DataFrame)
      - 'metrics_050' (pd.DataFrame)
      - 'metrics_best' (pd.DataFrame)
      - 'lr_best_th' (float)
      - 'rf_best_th' (float)
    """
    ensure_outputs_dir(OUT_DIR)

    print("\n=== 1) Loading data ===")
    df = load_data()
    print(df.head())

    print("\n=== 2) EDA figures ===")
    plot_class_imbalance(df, OUT_DIR)
    plot_histograms(df, OUT_DIR)
    plot_vcorr_heatmap(df, OUT_DIR)

    print("\n=== 3) Train/Test Split (80/20, stratified) ===")
    X_train, X_test, y_train, y_test = stratified_split(df)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    print("\n=== 4) Pipelines (StandardScaler → SMOTE → Classifier) ===")
    pipe_lr, pipe_rf = build_pipelines()

    print("\n=== 5) Hyperparameter Optimization (5-fold, scoring=F1) ===")
    search_lr, search_rf = tune_models(pipe_lr, pipe_rf, X_train, y_train, n_iter_rf=25)

    best_rows = [
        {"Model": "Logistic Regression", "Best Params": search_lr.best_params_, "CV Best F1": round(search_lr.best_score_, 6)},
        {"Model": "Random Forest", "Best Params": search_rf.best_params_, "CV Best F1": round(search_rf.best_score_, 6)},
    ]
    best_table = pd.DataFrame(best_rows)
    table_to_csv(best_table, "best_params_and_cv_scores.csv", OUT_DIR)
    print(best_table)

    print("\n=== 6) Threshold Optimization on TRAIN (CV) ===")
    lr_best_th, lr_pr_auc_cv = find_best_threshold_cv(search_lr.best_estimator_, X_train, y_train)
    rf_best_th, rf_pr_auc_cv = find_best_threshold_cv(search_rf.best_estimator_, X_train, y_train)
    print(f"LR best threshold = {lr_best_th:.3f} | PR-AUC (CV) = {lr_pr_auc_cv:.6f}")
    print(f"RF best threshold = {rf_best_th:.3f} | PR-AUC (CV) = {rf_pr_auc_cv:.6f}")

    print("\n=== 7) Final Fit on TRAIN and Evaluation on TEST ===")
    best_lr_est = search_lr.best_estimator_.fit(X_train, y_train)
    best_rf_est = search_rf.best_estimator_.fit(X_train, y_train)

    proba_lr_test = best_lr_est.predict_proba(X_test)[:, 1]
    proba_rf_test = best_rf_est.predict_proba(X_test)[:, 1]

    # Metrics @ 0.50 threshold
    rows_050 = []
    for name, proba in [("Logistic Regression", proba_lr_test), ("Random Forest", proba_rf_test)]:
        P, R, F1, _ = eval_at_threshold(y_test.values, proba, 0.50)
        rows_050.append({"Model": name, "Threshold": 0.50, "Precision": P, "Recall": R, "F1": F1})
    metrics_050 = pd.DataFrame(rows_050)
    table_to_csv(metrics_050, "metrics_test_threshold_050.csv", OUT_DIR)
    print(metrics_050)

    # Metrics @ best-F1 thresholds
    rows_best = []
    for name, proba, th in [
        ("Logistic Regression", proba_lr_test, lr_best_th),
        ("Random Forest", proba_rf_test, rf_best_th),
    ]:
        P, R, F1, _ = eval_at_threshold(y_test.values, proba, th)
        rows_best.append({"Model": name, "Threshold": round(th, 4), "Precision": P, "Recall": R, "F1": F1})
    metrics_best = pd.DataFrame(rows_best)
    table_to_csv(metrics_best, "metrics_test_bestF1threshold.csv", OUT_DIR)
    print(metrics_best)

    print("\n=== 8) PR Curves (Test) ===")
    plot_pr_curves(y_test.values, proba_lr_test, proba_rf_test, OUT_DIR)

    print("\n=== 9) Confusion Matrices ===")
    # Fixed 0.50
    for name, proba in [("LR", proba_lr_test), ("RF", proba_rf_test)]:
        _, _, _, cm = eval_at_threshold(y_test.values, proba, 0.50)
        plot_conf_mx(cm, f"Confusion Matrix — {name} (threshold=0.50)", os.path.join(OUT_DIR, f"cm_{name.lower()}_050.png"))
    # Best-F1 thresholds
    for name, proba, th in [("LR", proba_lr_test, lr_best_th), ("RF", proba_rf_test, rf_best_th)]:
        _, _, _, cm = eval_at_threshold(y_test.values, proba, th)
        plot_conf_mx(cm, f"Confusion Matrix — {name} (best F1 th={th:.2f})", os.path.join(OUT_DIR, f"cm_{name.lower()}_bestF1.png"))

    print("\n=== 10) Comparison Summary Bar Chart ===")
    comparison_bar_chart(metrics_050, metrics_best, OUT_DIR)

    print("\nDONE ✅ All figures saved to ./outputs and tables exported as CSV.\n")

    # Return artifacts for tests
    return {
        "best_table": best_table,
        "metrics_050": metrics_050,
        "metrics_best": metrics_best,
        "lr_best_th": lr_best_th,
        "rf_best_th": rf_best_th,
    }


if __name__ == "__main__":
    # Entrypoint for CLI execution
    _ = main()