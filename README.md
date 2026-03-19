
# 🚀 Customer Churn Prediction: High-Precision ML Pipeline

![Customer Churn](https://img.shields.io/badge/Model-Ensemble-blueviolet)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.91-green)
![Python](https://img.shields.io/badge/Python-3.12-blue)

## 📌 Project Overview
This project aims to predict customer churn with high precision using a dataset of nearly **600,000 customers**. By identifying "at-risk" customers early, businesses can implement retention strategies to reduce revenue loss.

**Key Achievement:** Achieved a holdout **ROC-AUC score of 0.9090** using a strategic ensemble of tree-based models.

---

## 🛠️ The Workflow

### 1. Data Audit & Cleaning 🧹
* **Target Mapping:** Converted categorical 'Churn' labels into binary (0/1) format.
* **Type Casting:** Handled `TotalCharges` as a float to ensure numerical consistency.
* **Integrity Check:** Verified zero missing values across all features.

### 2. Advanced Feature Engineering 🏗️
To move beyond raw data, four key behavioral features were engineered:
* **Promote:** Calculated the gap between `TotalCharges` and expected monthly spend to detect discounts/overcharges.
* **Remain_Contract:** Estimated time remaining on One-year and Two-year commitments.
* **Total_Services:** An engagement score summing all add-on services (TechSupport, OnlineBackup, etc.).
* **Avg_Cost:** Normalized monthly charges per active service to measure "perceived value."

### 3. Preprocessing & Encoding 🪄
* **Fast Label Encoding:** Implemented an optimized encoding strategy to sync Train and Test categories without data leakage.
* **Class Balancing (SMOTE):** Addressed the **3.4:1 imbalance** by oversampling the minority class in the training set, growing the training pool to ~920k rows for better pattern recognition.

### 4. Model Ensemble Strategy 🤖
We utilized a **5-Fold Cross-Validation** strategy to ensure stability and prevent overfitting. The final prediction is a weighted ensemble of:
1. **Random Forest:** The primary driver for stability and generalization.
2. **XGBoost:** Optimized with the `hist` tree method for high-speed gradient boosting.
3. **LightGBM:** A leaf-wise growth model for capturing complex, non-linear relationships.

---

## 📊 Performance Metrics
| Model | CV ROC-AUC | Holdout ROC-AUC |
| :--- | :--- | :--- |
| Random Forest | 0.9062 | **0.9090** |
| XGBoost | 0.9034 | 0.9045 |
| **Master Ensemble** | **0.9580** | **0.9050** |

---

## 🚀 How to Run
1. Ensure `requirements.txt` dependencies are installed: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `imblearn`.
2. Place `train.csv` and `test.csv` in the input directory.
3. Run the notebook cells sequentially to generate `submission_ensemble_master.csv`.

---

## 💡 Future Improvements
* **Hyperparameter Tuning:** Implement `Optuna` or `GridSearchCV` for fine-tuning LightGBM.
* **Feature Selection:** Use SHAP values to remove low-impact features and reduce noise.
* **Deep Learning:** Experiment with TabNet for potential gains in structured data performance.
