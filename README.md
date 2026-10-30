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
* **AdaBoost**: 71.29% Accuracy, 73.01
