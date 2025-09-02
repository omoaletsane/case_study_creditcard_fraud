#!/usr/bin/env python3
"""
End‑to‑end package for the case study:
Detecting fraudulent credit card transactions (Pozzolo et al., 2015)

What this script does, automatically:
 1) Loads creditcard.csv (expects it in the same folder, configurable via DATA_PATH)
 2) EDA figures: histograms (Amount, Time), class imbalance bar, correlation heatmap (V1–V28)
 3) Leakage‑safe 80/20 split, pipelines with scaling + SMOTE
 4) Two classifiers: Logistic Regression (GridSearchCV) and Random Forest (RandomizedSearchCV)
 5) Cross‑validated tuning (scoring = F1), best params reported
 6) Threshold optimization per model (best F1 on TRAIN via CV predictions)
 7) Test evaluation at 0.50 and at each model’s best‑F1 threshold
 8) PR curves, confusion matrices, and a comparison bar chart
 9) Saves all tables (CSV) and figures (PNG) under ./outputs

Usage (terminal):
    python credit_card_fraud_end_to_end.py

Requirements are auto‑installed if missing: numpy, pandas, matplotlib, seaborn,
scikit‑learn, imbalanced‑learn, scipy.

Author: (Your Name)
"""

import os
import sys
import warnings
import subprocess

# --- Auto-install missing packages -------------------------------------------------------------
REQUIRED = [
    "numpy", "pandas", "matplotlib", "seaborn",
    "scikit-learn", "imbalanced-learn", "scipy"
]
for pkg in REQUIRED:
    try:
        __import__(pkg if pkg != "scikit-learn" else "sklearn")
    except ImportError:
        print(f"Installing missing package: {pkg} …")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# --- Imports (now safe) ------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import randint, uniform

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_val_predict
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    precision_recall_curve, auc
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")  # keep output clean (convergence, etc.)

# --- Config ------------------------------------------------------------------------------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_PATH = os.environ.get("CREDITCARD_CSV", "creditcard.csv")
OUT_DIR   = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# --- Helpers -----------------------------------------------------------------------------------

def savefig(path, fig=None, tight=True):
    if fig is None:
        fig = plt.gcf()
    if tight:
        plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"[saved] {path}")


def table_to_csv(df: pd.DataFrame, fname: str) -> str:
    path = os.path.join(OUT_DIR, fname)
    df.to_csv(path, index=False)
    print(f"[saved] {path}")
    return path


def print_header(txt: str):
    print("\n" + "=" * 100)
    print(txt)
    print("=" * 100)

# --- Load data ---------------------------------------------------------------------------------
print_header("1) Loading data")
if not os.path.exists(DATA_PATH):
    sys.exit(f"ERROR: Could not find {DATA_PATH}. Place creditcard.csv in this folder or set CREDITCARD_CSV env var.")

df = pd.read_csv(DATA_PATH)
# Expect columns: Time, V1..V28, Amount, Class (0 legit, 1 fraud)
print(df.head())
print(df.describe().T.head())

# --- EDA: class imbalance, histograms, correlation heatmap -------------------------------------
print_header("2) Exploratory Data Analysis (figures)")

# Class imbalance table
class_counts = df['Class'].value_counts().rename(index={0: 'Legit', 1: 'Fraud'}).rename_axis('Class').reset_index(name='Count')
class_counts['Percent'] = (class_counts['Count'] / class_counts['Count'].sum() * 100).round(4)
print(class_counts)
_ = table_to_csv(class_counts, 'class_distribution.csv')

# Class imbalance bar chart
plt.figure(figsize=(6,4))
sns.barplot(data=class_counts, x='Class', y='Count', hue='Class', dodge=False)
for index, row in class_counts.iterrows():
    plt.text(index, row['Count']*1.01, f"{row['Count']:,}", ha='center')
plt.title('Class Imbalance: Legit vs Fraud')
plt.ylabel('Number of Transactions')
plt.legend().remove()
savefig(os.path.join(OUT_DIR, 'class_imbalance_bar.png'))
plt.close()

# Histograms: Amount & Time
for col, bins, fname in [("Amount", 60, 'hist_amount.png'), ("Time", 60, 'hist_time.png')]:
    plt.figure(figsize=(7,4))
    sns.histplot(df[col], bins=bins, kde=False)
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel('Count')
    savefig(os.path.join(OUT_DIR, fname))
    plt.close()

# Correlation heatmap for V1..V28
v_cols = [f"V{i}" for i in range(1,29)]
missing_v = [c for c in v_cols if c not in df.columns]
if not missing_v:
    corr = df[v_cols].corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, cmap='coolwarm', center=0, square=True, cbar_kws={'shrink':0.7})
    plt.title('Correlation Heatmap: V1–V28')
    savefig(os.path.join(OUT_DIR, 'heatmap_v1_v28.png'))
    plt.close()
else:
    print(f"[warn] Missing expected PCA columns (V1..V28). Skipping heatmap. Missing: {missing_v}")

# --- Split data --------------------------------------------------------------------------------
print_header("3) Train/Test Split (stratified 80/20)")
X = df.drop(columns=['Class'])
y = df['Class'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)
print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

# --- Pipelines: scaling + SMOTE + classifier ----------------------------------------------------
print_header("4) Building pipelines (StandardScaler → SMOTE → Classifier)")

# We'll scale ALL features for SMOTE distance fairness; trees are robust to scaling anyway
scaler = StandardScaler(with_mean=True, with_std=True)
smote  = SMOTE(random_state=RANDOM_STATE, sampling_strategy='auto', k_neighbors=5)

# Logistic Regression pipeline
pipe_lr = ImbPipeline(steps=[
    ('scaler', scaler),
    ('smote',  smote),
    ('clf',    LogisticRegression(max_iter=2000, solver='liblinear', n_jobs=None, random_state=RANDOM_STATE))
])

# Random Forest pipeline
pipe_rf = ImbPipeline(steps=[
    ('scaler', scaler),
    ('smote',  smote),
    ('clf',    RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))
])

# --- Hyperparameter search setups --------------------------------------------------------------
print_header("5) Hyperparameter Optimization (5-fold CV, scoring=F1)")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# LR: Grid Search
param_grid_lr = {
    'clf__penalty': ['l1', 'l2'],
    'clf__C': [0.01, 0.1, 1.0, 10.0, 100.0]
}
search_lr = GridSearchCV(
    estimator=pipe_lr,
    param_grid=param_grid_lr,
    scoring='f1',
    n_jobs=-1,
    cv=cv,
    verbose=1
)

# RF: Randomized Search
param_dist_rf = {
    'clf__n_estimators': randint(300, 900),
    'clf__max_depth': [None, 5, 10, 20, 30],
    'clf__min_samples_split': randint(2, 11),
    'clf__min_samples_leaf': randint(1, 5),
    'clf__max_features': ['sqrt', 'log2', 0.3, 0.5, 0.8]
}
search_rf = RandomizedSearchCV(
    estimator=pipe_rf,
    param_distributions=param_dist_rf,
    n_iter=25,
    scoring='f1',
    n_jobs=-1,
    cv=cv,
    random_state=RANDOM_STATE,
    verbose=1
)

# Fit searches
print("Tuning Logistic Regression …")
search_lr.fit(X_train, y_train)
print("Tuning Random Forest …")
search_rf.fit(X_train, y_train)

# Best params & CV F1
best_rows = []
for name, search in [("Logistic Regression", search_lr), ("Random Forest", search_rf)]:
    best_rows.append({
        'Model': name,
        'Best Params': search.best_params_,
        'CV Best F1': round(search.best_score_, 6)
    })

best_table = pd.DataFrame(best_rows)
print(best_table)
_ = table_to_csv(best_table, 'best_params_and_cv_scores.csv')

# --- Threshold optimization on TRAIN via CV predictions ----------------------------------------
print_header("6) Threshold Optimization (maximize F1 on TRAIN via CV-predict)")

# Returns best threshold and CV-PR AUC
def find_best_threshold(estimator, X, y, cv):
    # Cross‑validated predicted probabilities for positive class
    proba = cross_val_predict(estimator, X, y, cv=cv, method='predict_proba', n_jobs=-1)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y, proba)
    f1s = 2 * (precision * recall) / (precision + recall + 1e-12)
    best_idx = np.nanargmax(f1s)
    best_th = 0.5 if best_idx >= len(thresholds) else thresholds[best_idx]
    pr_auc = auc(recall, precision)
    return best_th, pr_auc, (precision, recall)

best_lr_est = search_lr.best_estimator_
best_rf_est = search_rf.best_estimator_

print("Finding best threshold for LR …")
lr_th, lr_pr_auc_cv, (lr_prec_cv, lr_rec_cv) = find_best_threshold(best_lr_est, X_train, y_train, cv)
print(f"LR best threshold (CV) = {lr_th:.3f} | PR-AUC (CV) = {lr_pr_auc_cv:.6f}")

print("Finding best threshold for RF …")
rf_th, rf_pr_auc_cv, (rf_prec_cv, rf_rec_cv) = find_best_threshold(best_rf_est, X_train, y_train, cv)
print(f"RF best threshold (CV) = {rf_th:.3f} | PR-AUC (CV) = {rf_pr_auc_cv:.6f}")

# --- Final fit on TRAIN and evaluation on TEST -------------------------------------------------
print_header("7) Final Fit on TRAIN and Evaluation on TEST")

best_lr_est.fit(X_train, y_train)
best_rf_est.fit(X_train, y_train)

# Predicted probabilities on TEST
proba_lr_test = best_lr_est.predict_proba(X_test)[:, 1]
proba_rf_test = best_rf_est.predict_proba(X_test)[:, 1]

# Helper to evaluate at a specific threshold

def evaluate_at_threshold(y_true, proba, th):
    y_pred = (proba >= th).astype(int)
    return (
        precision_score(y_true, y_pred, zero_division=0),
        recall_score(y_true, y_pred, zero_division=0),
        f1_score(y_true, y_pred, zero_division=0),
        confusion_matrix(y_true, y_pred)
    )

# Tables: metrics at 0.50 threshold
rows_050 = []
for name, proba in [("Logistic Regression", proba_lr_test), ("Random Forest", proba_rf_test)]:
    P, R, F1, CM = evaluate_at_threshold(y_test, proba, 0.50)
    rows_050.append({'Model': name, 'Threshold': 0.50, 'Precision': P, 'Recall': R, 'F1': F1})
metrics_050 = pd.DataFrame(rows_050)
print(metrics_050)
_ = table_to_csv(metrics_050, 'metrics_test_threshold_050.csv')

# Tables: metrics at each model's BEST‑F1 threshold (from TRAIN CV)
rows_best = []
for name, proba, th in [
    ("Logistic Regression", proba_lr_test, lr_th),
    ("Random Forest",      proba_rf_test, rf_th)
]:
    P, R, F1, CM = evaluate_at_threshold(y_test, proba, th)
    rows_best.append({'Model': name, 'Threshold': round(th, 4), 'Precision': P, 'Recall': R, 'F1': F1})
metrics_best = pd.DataFrame(rows_best)
print(metrics_best)
_ = table_to_csv(metrics_best, 'metrics_test_bestF1threshold.csv')

# --- PR curves on TEST -------------------------------------------------------------------------
print_header("8) PR Curves on TEST (both models)")

prec_lr, rec_lr, _ = precision_recall_curve(y_test, proba_lr_test)
prec_rf, rec_rf, _ = precision_recall_curve(y_test, proba_rf_test)

plt.figure(figsize=(7,6))
plt.plot(rec_lr, prec_lr, label=f"LR (PR-AUC={auc(rec_lr, prec_lr):.4f})")
plt.plot(rec_rf, prec_rf, label=f"RF (PR-AUC={auc(rec_rf, prec_rf):.4f})")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision–Recall Curves (Test)')
plt.xlim(0,1); plt.ylim(0,1)
plt.legend(loc='lower left')
savefig(os.path.join(OUT_DIR, 'pr_curves_test.png'))
plt.close()

# --- Confusion matrices (heatmaps) --------------------------------------------------------------
print_header("9) Confusion Matrices (heatmaps)")

# At fixed 0.50
for name, proba in [("LR", proba_lr_test), ("RF", proba_rf_test)]:
    _, _, _, cm = evaluate_at_threshold(y_test, proba, 0.50)
    plt.figure(figsize=(5.8,4.8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Pred Legit','Pred Fraud'],
                yticklabels=['Actual Legit','Actual Fraud'])
    plt.title(f'Confusion Matrix — {name} (threshold=0.50)')
    savefig(os.path.join(OUT_DIR, f'cm_{name.lower()}_050.png'))
    plt.close()

# At best-F1 thresholds
for name, proba, th in [("LR", proba_lr_test, lr_th), ("RF", proba_rf_test, rf_th)]:
    _, _, _, cm = evaluate_at_threshold(y_test, proba, th)
    plt.figure(figsize=(5.8,4.8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Pred Legit','Pred Fraud'],
                yticklabels=['Actual Legit','Actual Fraud'])
    plt.title(f'Confusion Matrix — {name} (best F1 th={th:.2f})')
    savefig(os.path.join(OUT_DIR, f'cm_{name.lower()}_bestF1.png'))
    plt.close()

# --- Comparison bar chart (Precision, Recall, F1) ----------------------------------------------
print_header("10) Comparison Summary Bar Chart")

comp_tbl = pd.concat([
    metrics_050.assign(Setting='Threshold 0.50'),
    metrics_best.assign(Setting='Best F1 Threshold')
], ignore_index=True)

# Long form for plotting
plot_df = comp_tbl.melt(id_vars=['Model','Threshold','Setting'], value_vars=['Precision','Recall','F1'],
                        var_name='Metric', value_name='Score')

plt.figure(figsize=(8.5,5.5))
sns.barplot(data=plot_df, x='Metric', y='Score', hue='Model')
plt.ylim(0,1.05)
plt.title('Model Comparison: Precision, Recall, F1 (by setting)')
plt.xlabel(''); plt.ylabel('Score')
savefig(os.path.join(OUT_DIR, 'compare_precision_recall_f1.png'))
plt.close()

# Also save table versions
_ = table_to_csv(comp_tbl, 'comparison_metrics_table.csv')

print_header("DONE ✅ All figures saved to ./outputs and tables exported as CSV.")
print("Suggested tables to include in the report:")
print(" - class_distribution.csv")
print(" - best_params_and_cv_scores.csv")
print(" - metrics_test_threshold_050.csv")
print(" - metrics_test_bestF1threshold.csv")
print(" - comparison_metrics_table.csv")
