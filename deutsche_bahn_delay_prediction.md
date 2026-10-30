# 🚆 Deutsche Bahn Delay Prediction: Data Cleaning & Machine Learning

## 📌 Executive Summary & Project Mandate
* **Goal**: Predict Deutsche Bahn train delays (`has_delay`: binary 0 = On-Time, 1 = Delayed) from pre-departure operational and schedule attributes.
* **Dataset Scope**: Cleaned dataset of **21,438 trips** (reduced from an initial 77,000 rows following the removal of outright cancellations and non-DB categories such as S-Bahn and buses).
* **Class Balance**: 
  * Test Split: 4,288 samples (**2,881 Delayed**, **1,407 On-Time**).

---

## ⚙️ Tech Stack & Project Structure
* **Language**: Python 3.x
* **Libraries**: Pandas, NumPy, Scikit-Learn (Modeling & Preprocessing), XGBoost, Matplotlib & Seaborn (Visualizations).
* **Main Notebook**: `DB_ML_analysis_manual_encoding_2.ipynb`

---

## 🛠️ Feature Engineering & Methodology Highlights
* **Temporal Parsing**: Merged `date` and `scheduled_time` into a unified datetime object, safely extracting `year`, `month`, `day`, `day_of_week`, `scheduled_hour`, `scheduled_minute`, and an `is_weekend` indicator.
* **Categorical Consolidation**:
  * `train_category`: Filtered strictly for DB long-distance and regional models (`ICE`, `RE`, `IC`, `RB`, `IRE`, `FEX`, `EC`).
  * `origin_grouped`: Top 12 busiest main stations (`Hbf`) + `"Other"`.
  * `destination_grouped`: Cleaned route strings into Top 20 destinations + `"Other"`.
* **Domain Interaction Feature**:
  * `is_ice_major_hub`: High-risk bottleneck flag for ICE trains originating or terminating at major congestion hubs (*München, Berlin Südkreuz, Koblenz, Frankfurt(Main), Köln*).
* **Preprocessing Architecture**:
  * One-hot encoded categorical columns using `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
  * Standardized numerical features with `MinMaxScaler()` fitted strictly on `X_train` to prevent data leakage.

---

## 📊 Model Benchmark Results (Default Baselines)
Evaluated on the 20% stratified test set (4,288 samples):
* **Random Forest**: **76.33% Accuracy**, 78.46% Precision, 89.27% Recall (Strongest overall baseline balance).
* **KNN ($k=5$)**: 73.69% Accuracy, 78.36% Precision, 84.07% Recall.
* **XGBoost (untuned)**: 73.51% Accuracy, 74.79% Precision, 91.36% Recall.
* **Gradient Boosting**: 72.04% Accuracy, 73.03% Precision, **92.57% Recall**.
* **AdaBoost**: 71.29% Accuracy, 73.01% Precision, 90.87% Recall.

---

## 🏆 Hyperparameter Tuning & Champion Model (Random Forest)
Both Random Forest and XGBoost underwent rigorous 5-fold cross-validation (`GridSearchCV`). **Random Forest** emerged as the final champion model, demonstrating the highest test accuracy and an optimal balance between precision and recall.

* **Best Parameters**: `max_depth: 20`, `max_features: 'sqrt'`, `min_samples_leaf: 1`, `min_samples_split: 5`, `n_estimators: 100`.
* **CV Training Accuracy**: 75.63%.
* **Test Set Accuracy**: **77.05%**.
* **Test Set Precision**: 78.63%.
* **Test Set Recall**: 90.42%.
* **ROC-AUC Score**: **0.8173**.

*(Note: Tuned XGBoost performed closely as a runner-up with a 76.17% Test Accuracy and 0.8106 ROC-AUC).*

---

## 🔍 Error & Diagnostic Analysis
1. **Confusion Matrix Analysis (Tuned Random Forest)**:
   * **True Positives (Delayed caught)**: **2,517**.
   * **True Negatives (On-Time caught)**: **749**.
   * **False Positives (False alarms)**: **658**.
   * **False Negatives (Missed delays)**: **364**.
2. **Feature Importance Insights**:
   * **Random Forest**: Emphasizes platform track bottlenecks, scheduled departure minute, and high-frequency long-distance connections as the top operational drivers for delay.
3. **Probability Distribution & ROC-AUC**:
   * **ROC-AUC Score**: **0.8173**.

---

## 🚀 Future Work & Model Expansion
To further improve predictive performance and contextual understanding, future development steps will focus on integrating:
* **Geospatial Metrics**: Examining physical route distance and transit distances between hubs.
* **Weather Conditions**: Merging historical weather data (precipitation, temperature) along the transit routes.
* **Delay Classifications**: Analyzing specific documented delay reasons (e.g., technical faults vs. track congestion).
* **Macro Seasonality**: Evaluating broader seasonal events, national holidays, and major track construction periods.