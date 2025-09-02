# Credit Card Fraud Detection — Case Study

## Background
Fraudulent credit card transactions occur much less frequently than legitimate ones, yet they cause significant harm to both institutions and individuals. Machine learning offers automated approaches for fraud detection, especially in highly imbalanced datasets (Hajek et al., 2022; Velarde et al., 2024).

This case study uses the **Credit Card Fraud Detection dataset** (Pozzolo et al., 2015), available at [Machine Learning Group – ULB](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). The dataset contains anonymized transaction features (`V1–V28` from PCA), `Time`, `Amount`, and the target `Class` (0 = legitimate, 1 = fraud).

---

## Objectives
- Explore the dataset: distributions, imbalance, correlations.  
- Implement two classifiers using **pipelines** to prevent overfitting:  
- Logistic Regression (baseline, interpretable).  
- Random Forest (non-linear, robust).  
- Optimize classifiers using **Grid Search** (Logistic Regression) and **Randomized Search** (Random Forest).  
- Compare classifiers in terms of **Precision, Recall, and F1-score**.  
- Reflect on model behavior, preprocessing effects, and optimization strategies.

---

## Methodology

### Preprocessing
- Features standardized with `StandardScaler`.  
- Severe class imbalance handled using **SMOTE** oversampling.  
- Pipelines ensure scaling and resampling are applied only within training folds (no leakage).

### Modeling
- **Logistic Regression** with hyperparameter tuning for penalty type (`l1`, `l2`) and regularization strength `C`.  
- **Random Forest** with randomized search over number of trees, depth, feature subsets, and split rules.  
- Evaluation via **5-fold stratified cross-validation** and an **80/20 train-test split**.

### Evaluation
- Metrics: **Precision, Recall, F1** (preferred over Accuracy).  
- Visuals:  
  - Class imbalance bar chart  
  - Histograms of Amount and Time  
  - Correlation heatmap (V1–V28)  
  - Precision–Recall curves  
  - Confusion matrices (heatmaps)  
  - Metric comparison bar chart  

All outputs (figures and CSV tables) are saved in the `outputs/` folder.

---

## How to Run

### Prerequisites
- Python 3.9+  
- Packages: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `imbalanced-learn`, `scipy`.

### Steps
1. **Download the dataset** from [Kaggle / ULB Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).  
2. Place the downloaded file (`creditcard.csv`) in the same folder as `credit_card_fraud_end_to_end.py`.  
3. Open a terminal / Anaconda prompt in that folder.  
4. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate   # Windows
   source .venv/bin/activate   # macOS/Linux
   ```
5. Run the script:
   ```bash
   python credit_card_fraud_end_to_end.py
   ```
6. Check the `outputs/` folder for results (figures and CSV tables).

---

## References
- Hajek, P., Abedin, M. Z., & Sivarajah, U. (2022). Fraud detection in mobile payment systems using an XGBoost-based framework. *Information Systems Frontiers*, 1–19.  
- Machine Learning Group – ULB (n.d.). Credit Card Fraud Detection. https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  
- Pozzolo, A. D., Caelen, O., Johnson, R. A., & Bontempi, G. (2015). *Calibrating Probability with Undersampling for Unbalanced Classification*. 2015 IEEE Symposium Series on Computational Intelligence, 159–166. https://doi.org/10.1109/SSCI.2015.33  
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.  
- Velarde, G., Weichert, M., Deshmunkh, A., Deshmane, S., Sudhir, A., Sharma, K., & Joshi, V. (2024). Tree boosting methods for balanced and imbalanced classification and their robustness over time in risk assessment. *Intelligent Systems with Applications*, 200354. https://doi.org/10.1016/j.iswa.2024.200354  
