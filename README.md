# Diabetes Classification Using Machine Learning

> Dataset: Diabetes Patient Records (10,000 observations)  
> Models: KNN · Random Forest · Support Vector Machine (SVM)  
> Evaluation: With and Without Cross-Validation

---

## Project Overview

This project builds and evaluates multiple machine learning classification models to predict whether a patient is **Diabetic (1)** or **Non-Diabetic (0)** using clinical and physiological features. Each model is trained both with and without cross-validation and compared across accuracy, classification report, and confusion matrix metrics.

**Best Model: Random Forest — 93.8% Accuracy | Lowest False Negatives (75)**

---

## Dataset

- **Records:** 10,000 patient observations
- **Features:** Glucose level, blood pressure, BMI, insulin level, age, diabetes pedigree function
- **Target:** `Diabetic` (0 = Non-Diabetic, 1 = Diabetic)
- **Class balance:** ~67% Non-Diabetic / ~33% Diabetic (moderate imbalance)
- **Missing values:** None

---

## Model Results — Without Cross-Validation

| Model | Accuracy | Diabetic Recall | False Negatives |
|---|---|---|---|
| KNN (k=5) | 86.4% | 0.74 | 174 |
| SVM (RBF kernel) | 87.85% | 0.79 | 140 |
| **Random Forest** | **93.8%** | **0.89** | **75** |

### Random Forest — Best Model

| | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | 1282 (TN) | 49 (FP) |
| **Actual 1** | 75 (FN) | 594 (TP) |

Random Forest misses **fewer diabetic cases** than KNN (FN=75 vs 174) and SVM (FN=140) — critical in medical diagnosis where false negatives carry the highest risk.

---

## Model Results — With Cross-Validation (5-Fold)

All three models retrained using 5-fold cross-validation for more reliable performance estimates.

| Model | CV Mean Accuracy |
|---|---|
| KNN | ~86% |
| SVM | ~88% |
| **Random Forest** | **~94%** |

Random Forest consistently outperforms across both evaluation approaches.

---

## Repository Structure

```
├── Diabetes_Classification_Analysis.ipynb    # Main analysis notebook
├── diabetes dataset.csv                      # Patient records dataset
└── README.md
```

---

## Methodology

### 1. Data Preprocessing
- Dropped `PatientID` (non-predictive)
- 80/20 stratified train-test split
- `StandardScaler` applied for KNN and SVM (distance-based models)
- Random Forest uses raw features (scale-invariant)

### 2. Models
- **KNN** (`n_neighbors=5`) — distance-based, sensitive to scale
- **Random Forest** (`n_estimators=100`) — ensemble, handles non-linearity
- **SVM** (`kernel='rbf'`) — maximum margin classifier

### 3. Evaluation
- Accuracy, Precision, Recall, F1-score
- Confusion matrix heatmaps
- 5-fold cross-validation for each model

---

## Key Insight

In medical diagnosis, **False Negatives are more costly** than False Positives — missing a diabetic patient is worse than a false alarm. Random Forest achieves the lowest FN count (75 vs 140–174 for other models), making it the most clinically appropriate choice.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas / NumPy | Data manipulation |
| Matplotlib / Seaborn | Confusion matrix heatmaps |
| Scikit-learn | KNN, Random Forest, SVM, CV, metrics |
| Jupyter Notebook | Analysis and reporting |

---

## How to Run

### 1. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 2. Update dataset path
In the notebook, update the CSV path to match your local file location.

### 3. Run the notebook
```bash
jupyter notebook Diabetes_Classification_Analysis.ipynb
```
