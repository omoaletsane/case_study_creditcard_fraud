import os, sys
# Add the repo root (one directory up from tests/) to PYTHONPATH so tests can `import credit_card_fraud_end_to_end`
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
