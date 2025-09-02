# Basic import/smoke tests: ensures the module can be imported without executing the pipeline.
import importlib

def test_import_module():
    mod = importlib.import_module("credit_card_fraud_end_to_end")
    assert hasattr(mod, "build_pipelines")
    assert hasattr(mod, "load_data")
    assert hasattr(mod, "main")
