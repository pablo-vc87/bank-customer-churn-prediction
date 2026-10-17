# Bank Customer Churn Prediction

> Versión en español: [LEEME.md](LEEME.md)

A binary classification project that predicts **which customers of Beta Bank are likely to leave the bank**, so the bank can try to retain them before they go.

The project compares **Logistic Regression** and **Random Forest** under different class-imbalance strategies (decision thresholds, class weights, and oversampling) and tunes the Random Forest hyperparameters. The selected model, **Random Forest (120 trees, max depth 13) trained with oversampling of the minority class**, achieved an **F1-score of 0.6076 on validation** and **0.6108 F1 with 0.8655 ROC-AUC on the independent test set**, above the requirement of **F1 ≥ 0.59**.

---

# Business Context

Beta Bank is losing customers month after month, slowly but steadily. The bank found that **keeping an existing customer is cheaper than attracting a new one**, so it wants to identify the customers who are about to leave and offer them retention benefits in time. A reliable model helps to:

* Focus retention offers on the customers with the highest risk.
* Avoid spending on customers who would stay anyway.
* Measure how many of the customers who leave can be detected in advance.

The model must predict one target:

| Target | Description |
| --- | --- |
| `Exited` | 1 if the customer left the bank, 0 if the customer stayed |

---

# Churn Modeling Concepts

| Term | Meaning |
| --- | --- |
| Churn | A customer who leaves the bank |
| Class imbalance | Only about 20% of customers left, so a model can look accurate by predicting "stays" for everyone |
| Decision threshold | The probability above which the model predicts churn (0.5 by default) |
| `class_weight='balanced'` | Gives more weight to the minority class during training |
| Oversampling | Repeating minority-class rows in the training set to balance the classes |
| Stratified split | Keeping the same churn proportion in train, validation, and test |

## What Each Error Means for the Bank

| Outcome | Meaning | Impact |
| --- | --- | --- |
| True Positive | The customer leaves and the model detects it | The bank can act to retain them |
| True Negative | The customer stays and the model predicts it | No resources are spent |
| False Positive | The model predicts churn, but the customer stays | A retention offer is spent unnecessarily |
| False Negative | The model predicts the customer stays, but they leave | **The customer is lost without any attempt to retain them** |

False negatives are the most expensive error, so besides F1 the project pays special attention to **Recall** on the churn class.

---

# Evaluation Metric

The main metric is the **F1-score** of the churn class, the harmonic mean of Precision and Recall:

$$
F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

F1 is only high when the model **finds most of the customers who leave** (Recall) and **is right when it flags them** (Precision). Accuracy is not useful here: a model that predicts "stays" for everyone would reach about 80% Accuracy and detect no churn at all.

**ROC-AUC** was recorded as a secondary metric, because it measures how well the model ranks customers by risk regardless of the threshold.

The project requirement was **F1 ≥ 0.59** on the test set.

---

# Dataset

| File | Rows | Columns | Description |
| --- | ---: | ---: | --- |
| `data/Churn.csv` | 10,000 | 14 | One row per customer: profile, account information, and whether they left |

| Column | Description |
| --- | --- |
| `RowNumber`, `CustomerId`, `Surname` | Row index, unique customer ID, and last name |
| `CreditScore` | Credit score |
| `Geography` | Country of residence (France, Germany, or Spain) |
| `Gender` | Gender |
| `Age` | Age |
| `Tenure` | Years the customer has been with the bank (0–10) |
| `Balance` | Account balance |
| `NumOfProducts` | Number of bank products used |
| `HasCrCard` | 1 if the customer has a credit card |
| `IsActiveMember` | 1 if the customer is active |
| `EstimatedSalary` | Estimated salary |
| `Exited` | Target: 1 if the customer left |

* **2,037 customers (20.37%)** left the bank and **7,963 (79.63%)** stayed.
* Customers are from France (5,014), Germany (2,509), and Spain (2,477).
* `Tenure` has **909 missing values**; it is the only column with missing data.

**Data source.** The dataset was provided by **TripleTen** as part of its Data Science Bootcamp and is included in this repository for educational and reproducibility purposes.

---

# Project Workflow

```text
Customer data (10,000 rows)
            │
            ▼
1. Data preparation
   ├── Missing Tenure → 0
   ├── Gender → binary, Geography → One-Hot Encoding
   └── Drop identifiers (RowNumber, CustomerId, Surname)
            │
            ▼
2. Stratified split: 70% train / 15% validation / 15% test
            │
            ▼
3. Logistic Regression
   ├── Baseline
   ├── Manual decision thresholds
   ├── class_weight='balanced'
   └── Oversampling (repeat=3)
            │
            ▼
4. Random Forest
   ├── Hyperparameter search (n_estimators, max_depth)
   ├── Fine search around the best values
   ├── Oversampling (repeat=3)
   └── class_weight='balanced'
            │
            ▼
5. Select the best model on validation and evaluate it once on test
```

---

# 1. Data Preparation

* **Missing values:** the 909 missing values in `Tenure` were replaced with **0**.
* **Encoding:**
  * `Gender` was mapped to binary (`Male` = 1, `Female` = 0).
  * `Geography` was one-hot encoded with `drop_first=True`, creating `Geography_Germany` and `Geography_Spain` (France is the reference).
* **Identifiers removed:**
  * `RowNumber` and `CustomerId` have almost no correlation with `Exited` (−0.017 and −0.006).
  * `Surname` has 2,932 different values, most of them shared by very few customers, and 1,181 surnames appear in more than one country. It carries no useful churn signal and mostly reflects the customer's country, which is already in `Geography`.

The final feature set has **11 features**:

```text
CreditScore, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard,
IsActiveMember, EstimatedSalary, Geography_Germany, Geography_Spain
```

---

# 2. Data Analysis

## Class Imbalance

| Class | Customers | Share |
| --- | ---: | ---: |
| Stayed (0) | 7,963 | 79.63% |
| Left (1) | 2,037 | 20.37% |

With only one churner for every four customers who stay, an unbalanced model tends to predict "stays" almost always. This is why several balancing strategies were compared.

## Data Split

The data were split with `stratify` so every set keeps the 20.4% churn rate:

```text
100% of the data (10,000 customers)
│
├── 70% TRAIN ─────────→ 7,004 customers (1,426 churners) → model training
├── 15% VALIDATION ────→ 1,500 customers   (306 churners) → model selection
└── 15% TEST ──────────→ 1,496 customers   (305 churners) → final evaluation (used only once)
```

A 70/15/15 split was chosen to maximize the training data while keeping about 300 churners in each evaluation set.

## Oversampling

For oversampling, the churn rows of the training set were repeated **3 times** (5,578 // 1,426 = 3), going from 1,426 to **4,278 churners** vs 5,578 non-churners (43.4% churn). Validation and test were never modified.

---

# 3. Modeling

## Logistic Regression

| Strategy | F1 | Recall | Precision | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| Baseline (threshold 0.5) | 0.1070 | 0.06 | 0.39 | 0.6740 |
| Baseline, best manual threshold (0.20) | 0.4085 | 0.66 | 0.30 | 0.6740 |
| `class_weight='balanced'` | **0.4513** | 0.67 | 0.34 | **0.7237** |
| Oversampling (repeat=3) | 0.4370 | 0.53 | 0.37 | 0.7036 |

*Validation results.*

* The baseline detected only **6% of the customers who left**.
* Lowering the threshold raised Recall (0.91 at a 0.10 threshold) but hurt Precision, so F1 never went above 0.41.
* Balancing the classes helped, but **no Logistic Regression reached the F1 ≥ 0.59 requirement**. The features were not scaled, and the oversampled model showed a convergence warning, so the linear model was probably not used at its best.

## Random Forest: Hyperparameter Search

Every combination of `n_estimators` and `max_depth` was trained and evaluated on validation:

| Search | `n_estimators` | `max_depth` | Best combination | Best F1 |
| --- | --- | --- | --- | ---: |
| 1 | 50–200, step 25 | 3–15 | 100 trees, depth 13 | 0.5912 |
| 2 (fine) | 80–120, step 10 | 11–16 | **120 trees, depth 13** | **0.5941** |

Shallow trees (depth 3–4) had F1 below 0.36; F1 rose steadily up to a depth of about 13.

## Random Forest: Imbalance Strategies

All three models use 120 trees and a maximum depth of 13:

| Strategy | F1 | Recall | Precision | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| No balancing | 0.5941 | 0.46 | **0.83** | 0.8375 |
| `class_weight='balanced'` | 0.5939 | 0.51 | 0.72 | 0.8388 |
| **Oversampling (repeat=3)** | **0.6076** | **0.57** | 0.65 | **0.8420** |

*Validation results.*

* Without balancing, the Random Forest already met the F1 requirement, but it **missed more than half of the customers who left** (Recall 0.46).
* **Oversampling gave the best F1 and the highest Recall**, detecting 57% of churners instead of 46%, at the cost of lower Precision.
* Class weights barely changed F1 and improved Recall only slightly.

## Final Result

The Random Forest with oversampling was evaluated **once** on the test set:

```text
Model:          RandomForestClassifier(n_estimators=120, max_depth=13) + oversampling (repeat=3)
Validation F1:  0.6076
Test F1:        0.6108
Test ROC-AUC:   0.8655
Test Recall:    0.61
Test Precision: 0.61
Test Accuracy:  0.84
Requirement:    F1 ≥ 0.59   ✓
```

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Stayed (0) | 0.90 | 0.90 | 0.90 | 1,191 |
| Left (1) | 0.61 | 0.61 | 0.61 | 305 |

The test results are consistent with validation (F1 0.6108 vs 0.6076), so the model generalizes well to new customers.

---

# Conclusions

1. **The class imbalance was the main challenge.** Without balancing, Logistic Regression detected only 6% of the customers who left.

2. **Logistic Regression did not reach the requirement** with any strategy (best F1 = 0.45 with class weights).

3. **Random Forest clearly outperformed Logistic Regression** (ROC-AUC about 0.84 vs 0.72), which suggests non-linear relationships between the features and churn.

4. **Oversampling was the best balancing strategy for Random Forest**: it improved both F1 and Recall without changing the hyperparameters.

5. **The final model meets the requirement on unseen data**: F1 = 0.6108 and ROC-AUC = 0.8655 on the test set.

## Business Interpretation

The model detects about **61% of the customers who are going to leave**, and about 61% of the customers it flags actually leave. The bank can use the predicted probability to rank customers by risk and focus retention offers, such as cashback or gift cards, on the highest-risk group. About 39% of churners would still be missed with the current threshold, so the model should support retention decisions, not replace them.

## Next Steps

* Scale the numeric features (`StandardScaler`) before training Logistic Regression and compare it again.
* Tune the decision threshold **on validation** according to the cost of a retention offer versus the cost of losing a customer.
* Use stratified cross-validation for hyperparameter search instead of a single validation set.
* Analyze feature importance to explain which customer characteristics drive churn.
* Try gradient boosting models (for example, LightGBM).

---

# Project Structure

```text
bank-customer-churn-prediction/
│
├── data/
│   └── Churn.csv
│
├── notebooks/
│   └── customer_churn_prediction.ipynb
│
├── src/
│   └── funciones_personales.py
│
├── .gitignore
├── environment.yml
├── requirements.txt
├── LEEME.md
└── README.md
```

`src/funciones_personales.py` contains the helper functions used in the notebook: `roc_plot_enhanced` (classification report, F1, ROC-AUC, ROC and precision-recall curves, and a threshold table), `upsample` (oversampling), and `forest_classif_comb_f1` (Random Forest hyperparameter search).

---

# Installation

Clone the repository:

```bash
git clone https://github.com/pablo-vc87/bank-customer-churn-prediction.git

cd bank-customer-churn-prediction
```

## Option 1: Conda

```bash
conda env create -f environment.yml
conda activate bank_customer_churn
jupyter notebook
```

## Option 2: pip

Create and activate a virtual environment, then:

```bash
pip install -r requirements.txt
jupyter notebook
```

Open `notebooks/customer_churn_prediction.ipynb` and run all cells.

---

# Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* scikit-learn (Logistic Regression, Random Forest)
* Jupyter Notebook

---

# Skills Demonstrated

* Binary classification with class imbalance
* Decision threshold analysis
* Class weights and oversampling
* Hyperparameter tuning for Random Forest
* Train / validation / test methodology with stratified splits
* F1, Precision, Recall, and ROC-AUC evaluation
* ROC and precision-recall curves
* Translating classification errors into business impact

---

## Author

**Pablo Andrés Vázquez Calva**

Data Scientist

GitHub: https://github.com/pablo-vc87

LinkedIn: https://www.linkedin.com/in/pablo-avc/
