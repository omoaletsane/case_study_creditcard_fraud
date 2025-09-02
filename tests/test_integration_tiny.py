import os
import pandas as pd

def make_tiny_df(n_legit=300, n_fraud=12, n_features=6, seed=123):
    """
    Create a small dataset where the minority has >= 6 samples so SMOTE(k=5) can run.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    X_legit = rng.normal(0, 1, size=(n_legit, n_features))
    X_fraud = rng.normal(1.2, 1, size=(n_fraud, n_features))
    X = np.vstack([X_legit, X_fraud])
    y = np.array([0]*n_legit + [1]*n_fraud)
    cols = [f"V{i}" for i in range(1, n_features+1)]
    df = pd.DataFrame(X, columns=cols)
    df["Time"] = rng.integers(0, 50000, size=len(df))
    df["Amount"] = np.abs(rng.normal(60, 25, size=len(df)))
    df["Class"] = y
    return df

def test_end_to_end_on_tiny(tmp_path, monkeypatch):
    # 1) Write a tiny CSV in a temporary directory
    df = make_tiny_df()
    csv_path = tmp_path / "creditcard.csv"
    df.to_csv(csv_path, index=False)

    # 2) Point the script to this CSV and run inside tmp_path so outputs/ is isolated
    monkeypatch.setenv("CREDITCARD_CSV", str(csv_path))
    monkeypatch.chdir(tmp_path)

    # 3) Import module AFTER env is set and CWD changed
    import credit_card_fraud_end_to_end as mod

    # 4) Speed up tuning in tests: monkeypatch tune_models to use fewer RF candidates
    orig_tune = mod.tune_models
    def fast_tune(pipe_lr, pipe_rf, X_train, y_train, n_iter_rf=5):
        return orig_tune(pipe_lr, pipe_rf, X_train, y_train, n_iter_rf=5)
    monkeypatch.setattr(mod, "tune_models", fast_tune)

    # 5) Run the full pipeline
    artifacts = mod.main()

    # 6) Assert key artifacts/tables exist
    out_dir = tmp_path / "outputs"
    assert out_dir.is_dir()

    for fname in [
        "class_distribution.csv",
        "best_params_and_cv_scores.csv",
        "metrics_test_threshold_050.csv",
        "metrics_test_bestF1threshold.csv",
        "comparison_metrics_table.csv",
    ]:
        assert (out_dir / fname).exists(), f"Missing expected output table: {f}"

    # 7) Returned artifacts sanity
    assert "best_table" in artifacts
    assert "metrics_050" in artifacts
    assert "metrics_best" in artifacts
