# 🏥 Task 4 — Disease Prediction from Medical Data

A machine learning system that predicts 3 diseases using patient medical data.

## 🎯 Objective
Predict the possibility of Heart Disease, Diabetes, and Breast Cancer using classification algorithms.

## 🚀 Features
- 3 datasets: Heart Disease (1000), Diabetes (768), Breast Cancer (569 real UCI data)
- 5 ML models compared per disease
- EDA with class distribution, feature correlations, distributions
- ROC curves, confusion matrices, accuracy comparison
- Real-time patient prediction
- Cross validation

## 📊 Results

| Disease | Best Model | Accuracy | AUC |
|---------|-----------|----------|-----|
| Heart Disease | Gradient Boosting | 89.5% | 95.5% |
| Diabetes | Random Forest | 82.5% | 87.3% |
| Breast Cancer | Logistic Regression | **98.3%** | **99.5%** |

## 🛠️ Tech Stack
- Python 3
- Scikit-learn
- Pandas & NumPy
- Matplotlib & Seaborn

## ▶️ Run
```bash
pip install -r requirements.txt
python disease_prediction.py
```

## 📁 Output Files
- `eda_disease.png` — EDA visualizations
- `disease_results.png` — Model results, ROC curves, confusion matrices

## 👨‍💻 Author
Allen Stivanson Christian
CodeAlpha Machine Learning Internship — June 2026
Student ID: CA/DF1/110227
