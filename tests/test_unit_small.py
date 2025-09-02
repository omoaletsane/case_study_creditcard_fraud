import numpy as np
import pandas as pd

from credit_card_fraud_end_to_end import (
    stratified_split,
    build_pipelines,
    eval_at_threshold,
)

def make_tiny_df(n_legit=200, n_fraud=10, n_features=6, seed=0):
    """Create a small, synthetic fraud dataset for unit tests."""
    rng = np.random.default_rng(seed)
    X_legit = rng.normal(0, 1, size=(n_legit, n_features))
    X_fraud = rng.normal(1.0, 1, size=(n_fraud, n_features))
    X = np.vstack([X_legit, X_fraud])
    y = np.array([0]*n_legit + [1]*n_fraud)
    cols = [f"V{i}" for i in range(1, n_features+1)]
    df = pd.DataFrame(X, columns=cols)
    df["Time"] = rng.integers(0, 10000, size=len(df))
    df["Amount"] = np.abs(rng.normal(50, 30, size=len(df)))
    df["Class"] = y
    return df

def test_tiny_data_and_split():
    df = make_tiny_df()
    assert "Class" in df.columns
    assert df["Class"].nunique() == 2

    X_train, X_test, y_train, y_test = stratified_split(df, test_size=0.2)
    assert len(X_train) + len(X_test) == len(df)
    assert y_train.mean() > 0
    assert y_test.mean() > 0

def test_build_pipelines():
    pipe_lr, pipe_rf = build_pipelines()
    assert "clf" in dict(pipe_lr.named_steps)
    assert "clf" in dict(pipe_rf.named_steps)

def test_eval_at_threshold():
    y_true = np.array([0, 0, 1, 1])
    proba  = np.array([0.1, 0.2, 0.8, 0.7])
    P, R, F1, CM = eval_at_threshold(y_true, proba, th=0.5)
    assert P >= 0.99 and R >= 0.99 and F1 >= 0.99
    assert CM.shape == (2, 2)
