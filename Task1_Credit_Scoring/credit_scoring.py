"""
Credit Scoring Model — CodeAlpha Machine Learning Internship
Task 1: Predict an individual's creditworthiness using past financial data.
Author: Allen Stivanson Christian
Student ID: CA/DF1/110227
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)

# ── 1. Generate Synthetic Dataset ─────────────────────────────────────────────
def generate_dataset(n=1000, seed=42):
    """
    Generate a realistic synthetic credit scoring dataset.
    Features: age, income, debt, payment history, credit utilization, etc.
    """
    np.random.seed(seed)
    n = 1000

    age              = np.random.randint(18, 70, n)
    income           = np.random.randint(15000, 150000, n)
    debt             = np.random.randint(0, 80000, n)
    payment_history  = np.random.randint(0, 100, n)      # 0=poor, 100=perfect
    credit_util      = np.random.uniform(0, 1, n)        # credit utilization ratio
    num_accounts     = np.random.randint(1, 15, n)
    months_employed  = np.random.randint(0, 300, n)
    num_late_payments= np.random.randint(0, 20, n)
    loan_amount      = np.random.randint(1000, 50000, n)
    existing_loans   = np.random.randint(0, 5, n)

    # Creditworthy logic: higher score = more likely creditworthy
    score = (
        payment_history * 0.35 +
        (1 - credit_util) * 25 +
        (income / 150000) * 20 +
        (1 - debt / 80000) * 10 +
        (months_employed / 300) * 10 -
        num_late_payments * 2
    )
    noise = np.random.normal(0, 5, n)
    creditworthy = ((score + noise) > 35).astype(int)

    df = pd.DataFrame({
        "age":              age,
        "income":           income,
        "debt":             debt,
        "payment_history":  payment_history,
        "credit_utilization": credit_util.round(4),
        "num_accounts":     num_accounts,
        "months_employed":  months_employed,
        "num_late_payments":num_late_payments,
        "loan_amount":      loan_amount,
        "existing_loans":   existing_loans,
        "creditworthy":     creditworthy
    })
    return df

# ── 2. Exploratory Data Analysis ──────────────────────────────────────────────
def run_eda(df):
    print("\n" + "="*60)
    print("   EXPLORATORY DATA ANALYSIS")
    print("="*60)
    print(f"\n Dataset shape     : {df.shape}")
    print(f" Creditworthy (1)  : {df['creditworthy'].sum()} ({df['creditworthy'].mean()*100:.1f}%)")
    print(f" Not worthy   (0)  : {(df['creditworthy']==0).sum()} ({(df['creditworthy']==0).mean()*100:.1f}%)")
    print(f"\n Missing values    : {df.isnull().sum().sum()}")
    print("\n Feature Statistics:")
    print(df.describe().round(2).to_string())

    # Plot class distribution
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Credit Scoring — Exploratory Data Analysis", fontsize=14, fontweight="bold")

    # Class distribution
    df["creditworthy"].value_counts().plot(kind="bar", ax=axes[0,0],
        color=["#e74c3c", "#2ecc71"], edgecolor="black")
    axes[0,0].set_title("Class Distribution")
    axes[0,0].set_xticklabels(["Not Creditworthy", "Creditworthy"], rotation=0)
    axes[0,0].set_ylabel("Count")

    # Income distribution by class
    df.groupby("creditworthy")["income"].plot(kind="hist", ax=axes[0,1],
        alpha=0.6, bins=30, legend=True)
    axes[0,1].set_title("Income Distribution by Class")
    axes[0,1].set_xlabel("Income")
    axes[0,1].legend(["Not Creditworthy", "Creditworthy"])

    # Payment history
    df.groupby("creditworthy")["payment_history"].plot(kind="hist", ax=axes[0,2],
        alpha=0.6, bins=30)
    axes[0,2].set_title("Payment History by Class")
    axes[0,2].set_xlabel("Payment History Score")
    axes[0,2].legend(["Not Creditworthy", "Creditworthy"])

    # Credit utilization
    df.groupby("creditworthy")["credit_utilization"].plot(kind="hist", ax=axes[1,0],
        alpha=0.6, bins=30)
    axes[1,0].set_title("Credit Utilization by Class")
    axes[1,0].set_xlabel("Credit Utilization Ratio")
    axes[1,0].legend(["Not Creditworthy", "Creditworthy"])

    # Correlation heatmap
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                ax=axes[1,1], linewidths=0.5, annot_kws={"size": 7})
    axes[1,1].set_title("Feature Correlation Heatmap")

    # Age distribution
    df.groupby("creditworthy")["age"].plot(kind="hist", ax=axes[1,2],
        alpha=0.6, bins=20)
    axes[1,2].set_title("Age Distribution by Class")
    axes[1,2].set_xlabel("Age")
    axes[1,2].legend(["Not Creditworthy", "Creditworthy"])

    plt.tight_layout()
    plt.savefig("eda_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n ✔ EDA plots saved to 'eda_analysis.png'")

# ── 3. Preprocessing ──────────────────────────────────────────────────────────
def preprocess(df):
    X = df.drop("creditworthy", axis=1)
    y = df["creditworthy"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    return X_train_sc, X_test_sc, y_train, y_test, scaler, X.columns.tolist()

# ── 4. Train & Evaluate Models ────────────────────────────────────────────────
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_proba)
    cv   = cross_val_score(model, X_train, y_train, cv=5, scoring="f1").mean()

    return {
        "Model": name, "Accuracy": acc, "Precision": prec,
        "Recall": rec, "F1-Score": f1, "ROC-AUC": auc,
        "CV F1": cv, "predictions": y_pred, "probabilities": y_proba
    }

# ── 5. Plot Results ───────────────────────────────────────────────────────────
def plot_results(results, y_test, feature_names, best_model):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("Credit Scoring Model — Results", fontsize=14, fontweight="bold")

    # Model comparison bar chart
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    model_names = [r["Model"] for r in results]
    x = np.arange(len(metrics))
    width = 0.2
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"]

    for i, (res, color) in enumerate(zip(results, colors)):
        vals = [res[m] for m in metrics]
        axes[0,0].bar(x + i*width, vals, width, label=res["Model"],
                      color=color, alpha=0.85, edgecolor="black")

    axes[0,0].set_xticks(x + width*1.5)
    axes[0,0].set_xticklabels(metrics, fontsize=9)
    axes[0,0].set_ylim(0.5, 1.05)
    axes[0,0].set_title("Model Performance Comparison")
    axes[0,0].legend(fontsize=8)
    axes[0,0].set_ylabel("Score")

    # ROC curves
    colors_roc = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"]
    for res, color in zip(results, colors_roc):
        fpr, tpr, _ = roc_curve(y_test, res["probabilities"])
        axes[0,1].plot(fpr, tpr, color=color,
                       label=f"{res['Model']} (AUC={res['ROC-AUC']:.3f})")
    axes[0,1].plot([0,1],[0,1],"k--", alpha=0.5)
    axes[0,1].set_title("ROC Curves — All Models")
    axes[0,1].set_xlabel("False Positive Rate")
    axes[0,1].set_ylabel("True Positive Rate")
    axes[0,1].legend(fontsize=8)

    # Confusion matrix for best model
    best_res = max(results, key=lambda r: r["F1-Score"])
    cm = confusion_matrix(y_test, best_res["predictions"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1,0],
                xticklabels=["Not Creditworthy","Creditworthy"],
                yticklabels=["Not Creditworthy","Creditworthy"])
    axes[1,0].set_title(f"Confusion Matrix — {best_res['Model']}")
    axes[1,0].set_xlabel("Predicted")
    axes[1,0].set_ylabel("Actual")

    # Feature importance (Random Forest)
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        axes[1,1].bar(range(len(feature_names)),
                      importances[indices], color="#3498db", edgecolor="black")
        axes[1,1].set_xticks(range(len(feature_names)))
        axes[1,1].set_xticklabels([feature_names[i] for i in indices],
                                   rotation=45, ha="right", fontsize=8)
        axes[1,1].set_title("Feature Importance — Random Forest")
        axes[1,1].set_ylabel("Importance")

    plt.tight_layout()
    plt.savefig("model_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(" ✔ Results plots saved to 'model_results.png'")

# ── 6. Prediction Function ────────────────────────────────────────────────────
def predict_credit(model, scaler, feature_names):
    print("\n" + "="*60)
    print("   CREDIT SCORE PREDICTOR")
    print("="*60)
    print(" Enter applicant details:\n")

    sample = {}
    prompts = {
        "age": ("Age", 18, 70),
        "income": ("Annual Income ($)", 0, 500000),
        "debt": ("Total Debt ($)", 0, 500000),
        "payment_history": ("Payment History Score (0-100)", 0, 100),
        "credit_utilization": ("Credit Utilization (0.0-1.0)", 0.0, 1.0),
        "num_accounts": ("Number of Accounts", 0, 30),
        "months_employed": ("Months Employed", 0, 600),
        "num_late_payments": ("Number of Late Payments", 0, 50),
        "loan_amount": ("Requested Loan Amount ($)", 0, 500000),
        "existing_loans": ("Number of Existing Loans", 0, 20),
    }

    for feat, (label, lo, hi) in prompts.items():
        while True:
            try:
                val = float(input(f"  {label} [{lo}-{hi}]: ").strip())
                sample[feat] = val
                break
            except ValueError:
                print("  ⚠ Please enter a valid number.")

    X_input = pd.DataFrame([sample])[feature_names]
    X_scaled = scaler.transform(X_input)
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0][1]

    print(f"\n {'─'*58}")
    print(f"  🏦 CREDIT ASSESSMENT RESULT")
    print(f" {'─'*58}")
    if prediction == 1:
        print(f"  ✅ CREDITWORTHY")
        print(f"  Approval Probability : {probability*100:.1f}%")
        print(f"  Recommendation       : APPROVE LOAN")
    else:
        print(f"  ❌ NOT CREDITWORTHY")
        print(f"  Approval Probability : {probability*100:.1f}%")
        print(f"  Recommendation       : DECLINE LOAN")
    print(f" {'─'*58}\n")

# ── 7. Main ───────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("   CREDIT SCORING MODEL")
    print("   CodeAlpha ML Internship — Task 1")
    print("   Allen Stivanson Christian | CA/DF1/110227")
    print("="*60)

    # Generate data
    print("\n [1/5] Generating dataset...")
    df = generate_dataset()
    print(f"  ✔ Dataset created: {df.shape[0]} samples, {df.shape[1]-1} features")

    # EDA
    print("\n [2/5] Running Exploratory Data Analysis...")
    run_eda(df)

    # Preprocess
    print("\n [3/5] Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess(df)
    print(f"  ✔ Train: {len(X_train)} | Test: {len(X_test)}")

    # Train models
    print("\n [4/5] Training & Evaluating Models...")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    results = []
    trained_models = {}
    for name, model in models.items():
        print(f"  Training {name}...", end=" ")
        res = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        results.append(res)
        trained_models[name] = model
        print(f"✔ Accuracy: {res['Accuracy']:.3f} | F1: {res['F1-Score']:.3f} | AUC: {res['ROC-AUC']:.3f}")

    # Results summary
    print("\n" + "="*60)
    print("   MODEL COMPARISON RESULTS")
    print("="*60)
    print(f"\n {'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC':>8} {'CV F1':>8}")
    print(f" {'─'*22} {'─'*9} {'─'*10} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for r in results:
        print(f" {r['Model']:<22} {r['Accuracy']:>9.4f} {r['Precision']:>10.4f} "
              f"{r['Recall']:>8.4f} {r['F1-Score']:>8.4f} {r['ROC-AUC']:>8.4f} {r['CV F1']:>8.4f}")

    best = max(results, key=lambda r: r["F1-Score"])
    print(f"\n 🏆 Best Model: {best['Model']} (F1-Score: {best['F1-Score']:.4f})")

    # Best model report
    print(f"\n Classification Report — {best['Model']}:")
    print(classification_report(y_test, best["predictions"],
                                target_names=["Not Creditworthy", "Creditworthy"]))

    # Plots
    print("\n [5/5] Generating visualizations...")
    best_model = trained_models[best["Model"]]
    plot_results(results, y_test, feature_names, best_model)

    # Predict for new applicant
    print("\n" + "="*60)
    again = input(" Would you like to predict for a new applicant? (y/n): ").strip().lower()
    if again == "y":
        predict_credit(best_model, scaler, feature_names)

    print("\n" + "="*60)
    print("  ✅ Credit Scoring Model — Complete!")
    print("  📊 Check 'eda_analysis.png' and 'model_results.png'")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
