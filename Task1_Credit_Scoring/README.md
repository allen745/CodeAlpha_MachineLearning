# 💳 Task 1 — Credit Scoring Model

A machine learning model to predict an individual's creditworthiness using past financial data.

## 🎯 Objective
Predict whether a loan applicant is creditworthy based on their financial history.

## 🚀 Features
- Synthetic realistic dataset generation (1000 samples, 10 features)
- Exploratory Data Analysis with visualizations
- 4 ML models compared: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
- Evaluation metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- 5-Fold Cross Validation
- ROC curves comparison
- Feature importance visualization
- Confusion matrix
- Real-time prediction for new applicants

## 📊 Results

| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| Logistic Regression | 92.0% | 91.0% | 98.5% |
| Gradient Boosting | 90.5% | 89.3% | 96.8% |
| Random Forest | 89.0% | 87.1% | 96.5% |
| Decision Tree | 75.5% | 72.0% | 79.7% |

🏆 **Best Model: Logistic Regression (F1: 91.0%, AUC: 98.5%)**

## 🛠️ Tech Stack
- Python 3
- Scikit-learn
- Pandas & NumPy
- Matplotlib & Seaborn

## ▶️ Run
```bash
pip install -r requirements.txt
python credit_scoring.py
```

## 📁 Output Files
- `eda_analysis.png` — Exploratory Data Analysis plots
- `model_results.png` — Model comparison, ROC curves, confusion matrix

## 👨‍💻 Author
Allen Stivanson Christian
CodeAlpha Machine Learning Internship — June 2026
Student ID: CA/DF1/110227
