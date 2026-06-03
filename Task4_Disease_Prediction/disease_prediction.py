"""
Disease Prediction from Medical Data — CodeAlpha Machine Learning Internship
Task 4: Predict the possibility of diseases based on patient data.
Datasets: Heart Disease, Diabetes, Breast Cancer (UCI ML Repository)
Author: Allen Stivanson Christian
Student ID: CA/DF1/110227
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)


def load_heart_disease():
    """Heart Disease Dataset — UCI Repository (synthetic realistic version)"""
    np.random.seed(42)
    n = 1000

    age         = np.random.randint(29, 77, n)
    sex         = np.random.randint(0, 2, n)
    cp          = np.random.randint(0, 4, n)        # chest pain type
    trestbps    = np.random.randint(94, 200, n)     # resting blood pressure
    chol        = np.random.randint(126, 564, n)    # cholesterol
    fbs         = np.random.randint(0, 2, n)        # fasting blood sugar
    restecg     = np.random.randint(0, 3, n)        # resting ECG
    thalach     = np.random.randint(71, 202, n)     # max heart rate
    exang       = np.random.randint(0, 2, n)        # exercise induced angina
    oldpeak     = np.random.uniform(0, 6.2, n)      # ST depression
    slope       = np.random.randint(0, 3, n)
    ca          = np.random.randint(0, 5, n)        # major vessels
    thal        = np.random.randint(0, 4, n)

    
    risk = (
        (age > 55) * 0.3 +
        (sex == 1) * 0.15 +
        (cp > 1) * 0.2 +
        (trestbps > 140) * 0.1 +
        (chol > 240) * 0.1 +
        (thalach < 150) * 0.1 +
        (exang == 1) * 0.15 +
        (oldpeak > 2) * 0.1 +
        (ca > 1) * 0.2
    )
    noise  = np.random.normal(0, 0.1, n)
    target = ((risk + noise) > 0.5).astype(int)

    df = pd.DataFrame({
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
        "exang": exang, "oldpeak": oldpeak.round(1), "slope": slope,
        "ca": ca, "thal": thal, "target": target
    })
    return df, "Heart Disease", ["No Disease", "Disease"]

def load_diabetes():
    """Diabetes Dataset — Pima Indians style (synthetic realistic version)"""
    np.random.seed(123)
    n = 768

    pregnancies = np.random.randint(0, 17, n)
    glucose     = np.random.randint(44, 199, n)
    bp          = np.random.randint(24, 122, n)
    skin        = np.random.randint(7, 99, n)
    insulin     = np.random.randint(14, 846, n)
    bmi         = np.random.uniform(18.2, 67.1, n)
    dpf         = np.random.uniform(0.08, 2.42, n)  # diabetes pedigree function
    age         = np.random.randint(21, 81, n)

    risk = (
        (glucose > 140) * 0.35 +
        (bmi > 30) * 0.2 +
        (age > 40) * 0.15 +
        (dpf > 0.5) * 0.15 +
        (bp > 80) * 0.1 +
        (pregnancies > 5) * 0.05
    )
    noise  = np.random.normal(0, 0.1, n)
    target = ((risk + noise) > 0.4).astype(int)

    df = pd.DataFrame({
        "Pregnancies": pregnancies, "Glucose": glucose,
        "BloodPressure": bp, "SkinThickness": skin,
        "Insulin": insulin, "BMI": bmi.round(1),
        "DiabetesPedigreeFunction": dpf.round(3),
        "Age": age, "Outcome": target
    })
    return df, "Diabetes", ["No Diabetes", "Diabetes"]

def load_cancer():
    """Breast Cancer Dataset — from sklearn (real UCI dataset)"""
    data   = load_breast_cancer()
    df     = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df, "Breast Cancer", ["Malignant", "Benign"]


def run_eda(datasets):
    print("\n [2/5] Running Exploratory Data Analysis...")
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    fig.suptitle("Disease Prediction — Exploratory Data Analysis",
                 fontsize=14, fontweight="bold")

    for row, (df, name, labels) in enumerate(datasets):
        target_col = "target" if "target" in df.columns else "Outcome"

        
        counts = df[target_col].value_counts()
        colors = ["#e74c3c", "#2ecc71"]
        axes[row, 0].pie(counts.values, labels=labels,
                         colors=colors, autopct="%1.1f%%",
                         startangle=90, wedgeprops={"edgecolor": "white"})
        axes[row, 0].set_title(f"{name}\nClass Distribution")

        
        numeric_df = df.select_dtypes(include=[np.number])
        corr_with_target = numeric_df.corr()[target_col].drop(target_col).abs()
        top_features = corr_with_target.nlargest(8)
        axes[row, 1].barh(range(len(top_features)), top_features.values,
                          color=plt.cm.RdYlGn(top_features.values), edgecolor="black")
        axes[row, 1].set_yticks(range(len(top_features)))
        axes[row, 1].set_yticklabels(top_features.index, fontsize=7)
        axes[row, 1].set_title(f"{name}\nTop Feature Correlations")
        axes[row, 1].set_xlabel("Correlation |value|")

        
        top_feat = top_features.index[0]
        for cls, color, label in zip([0, 1], colors, labels):
            subset = df[df[target_col] == cls][top_feat]
            axes[row, 2].hist(subset, bins=25, alpha=0.6,
                              color=color, label=label, edgecolor="black")
        axes[row, 2].set_title(f"{name}\n{top_feat} by Class")
        axes[row, 2].set_xlabel(top_feat)
        axes[row, 2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("eda_disease.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✔ EDA saved to 'eda_disease.png'")


def train_evaluate(df, name, labels):
    target_col = "target" if "target" in df.columns else "Outcome"
    X = df.drop(target_col, axis=1).select_dtypes(include=[np.number])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler      = StandardScaler()
    X_train_sc  = scaler.fit_transform(X_train)
    X_test_sc   = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM":                 SVC(probability=True, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
        "KNN":                 KNeighborsClassifier(n_neighbors=5),
    }

    results = []
    for mname, model in models.items():
        model.fit(X_train_sc, y_train)
        y_pred  = model.predict(X_test_sc)
        y_proba = model.predict_proba(X_test_sc)[:, 1]
        cv      = cross_val_score(model, X_train_sc, y_train, cv=5, scoring="f1").mean()

        results.append({
            "Model":     mname,
            "Accuracy":  accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall":    recall_score(y_test, y_pred, zero_division=0),
            "F1-Score":  f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC":   roc_auc_score(y_test, y_proba),
            "CV F1":     cv,
            "predictions":   y_pred,
            "probabilities": y_proba,
            "model_obj":     model
        })

    best = max(results, key=lambda r: r["F1-Score"])
    return results, best, X_test_sc, y_test, scaler, X.columns.tolist(), X_train_sc, y_train


def plot_all_results(all_results):
    print("\n [4/5] Generating result visualizations...")
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle("Disease Prediction — Model Results", fontsize=14, fontweight="bold")

    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"]

    for row, (results, best, X_test, y_test, _, _, _, _) in enumerate(all_results):
        name = ["Heart Disease", "Diabetes", "Breast Cancer"][row]

        # Accuracy comparison
        model_names = [r["Model"] for r in results]
        accs = [r["Accuracy"] for r in results]
        bars = axes[row, 0].bar(range(len(model_names)), accs,
                                color=colors, edgecolor="black", alpha=0.85)
        axes[row, 0].set_xticks(range(len(model_names)))
        axes[row, 0].set_xticklabels(model_names, rotation=30, ha="right", fontsize=7)
        axes[row, 0].set_ylim(0.5, 1.05)
        axes[row, 0].set_title(f"{name}\nModel Accuracy")
        axes[row, 0].set_ylabel("Accuracy")
        for bar, acc in zip(bars, accs):
            axes[row, 0].text(bar.get_x() + bar.get_width()/2,
                              bar.get_height() + 0.005,
                              f"{acc:.3f}", ha="center", va="bottom", fontsize=7)

        
        for res, color in zip(results, colors):
            fpr, tpr, _ = roc_curve(y_test, res["probabilities"])
            axes[row, 1].plot(fpr, tpr, color=color, linewidth=1.5,
                              label=f"{res['Model'][:8]} ({res['ROC-AUC']:.2f})")
        axes[row, 1].plot([0,1],[0,1],"k--", alpha=0.4)
        axes[row, 1].set_title(f"{name}\nROC Curves")
        axes[row, 1].set_xlabel("FPR")
        axes[row, 1].set_ylabel("TPR")
        axes[row, 1].legend(fontsize=6)

        # Confusion matrix of best model
        cm = confusion_matrix(y_test, best["predictions"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    ax=axes[row, 2], linewidths=0.5)
        axes[row, 2].set_title(f"{name}\nBest: {best['Model']}\nAcc:{best['Accuracy']:.3f}")
        axes[row, 2].set_xlabel("Predicted")
        axes[row, 2].set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig("disease_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✔ Results saved to 'disease_results.png'")


def predict_disease(best_model_obj, scaler, feature_names, disease_name, labels):
    print(f"\n {'─'*55}")
    print(f"  🏥 {disease_name.upper()} PREDICTOR")
    print(f" {'─'*55}")
    print(f"  Enter patient details:\n")

    sample = {}
    for feat in feature_names:
        while True:
            try:
                val = float(input(f"  {feat}: ").strip())
                sample[feat] = val
                break
            except ValueError:
                print("  ⚠ Enter a valid number.")

    X_input = pd.DataFrame([sample])[feature_names]
    X_scaled = scaler.transform(X_input)
    pred  = best_model_obj.predict(X_scaled)[0]
    proba = best_model_obj.predict_proba(X_scaled)[0][1]

    print(f"\n  Result: {'⚠️ ' + labels[1] if pred == 1 else '✅ ' + labels[0]}")
    print(f"  Probability: {proba*100:.1f}%")
    print(f" {'─'*55}\n")


def main():
    print("\n" + "="*60)
    print("   DISEASE PREDICTION FROM MEDICAL DATA")
    print("   CodeAlpha ML Internship — Task 4")
    print("   Allen Stivanson Christian | CA/DF1/110227")
    print("="*60)

    
    print("\n [1/5] Loading Datasets...")
    heart_df,   heart_name,   heart_labels   = load_heart_disease()
    diabetes_df, diabetes_name, diabetes_labels = load_diabetes()
    cancer_df,  cancer_name,  cancer_labels  = load_cancer()

    datasets = [
        (heart_df,    heart_name,    heart_labels),
        (diabetes_df, diabetes_name, diabetes_labels),
        (cancer_df,   cancer_name,   cancer_labels),
    ]

    print(f"  ✔ Heart Disease  : {heart_df.shape[0]} samples, {heart_df.shape[1]-1} features")
    print(f"  ✔ Diabetes       : {diabetes_df.shape[0]} samples, {diabetes_df.shape[1]-1} features")
    print(f"  ✔ Breast Cancer  : {cancer_df.shape[0]} samples, {cancer_df.shape[1]-1} features")

    
    run_eda(datasets)

    
    print("\n [3/5] Training & Evaluating Models...")
    all_results = []
    best_models = []

    for df, name, labels in datasets:
        print(f"\n  ── {name} ──")
        results, best, X_test, y_test, scaler, feat_names, X_train, y_train = \
            train_evaluate(df, name, labels)
        all_results.append((results, best, X_test, y_test, scaler, feat_names, X_train, y_train))
        best_models.append((best, scaler, feat_names, name, labels))

        print(f"  {'Model':<22} {'Acc':>7} {'F1':>7} {'AUC':>7}")
        print(f"  {'─'*22} {'─'*7} {'─'*7} {'─'*7}")
        for r in results:
            marker = " ← BEST" if r["Model"] == best["Model"] else ""
            print(f"  {r['Model']:<22} {r['Accuracy']:>7.4f} {r['F1-Score']:>7.4f} {r['ROC-AUC']:>7.4f}{marker}")

    
    plot_all_results(all_results)

    
    print("\n" + "="*60)
    print("   FINAL SUMMARY — BEST MODELS")
    print("="*60)
    print(f"\n  {'Disease':<20} {'Best Model':<22} {'Accuracy':>9} {'F1':>8} {'AUC':>8}")
    print(f"  {'─'*20} {'─'*22} {'─'*9} {'─'*8} {'─'*8}")
    for (results, best, *_), (_, _, _, name, _) in zip(all_results, best_models):
        print(f"  {name:<20} {best['Model']:<22} {best['Accuracy']:>9.4f} "
              f"{best['F1-Score']:>8.4f} {best['ROC-AUC']:>8.4f}")

    
    print("\n" + "="*60)
    choice = input("\n  Predict for a new patient? (y/n): ").strip().lower()
    if choice == "y":
        print("\n  Which disease to predict?")
        print("  1. Heart Disease")
        print("  2. Diabetes")
        print("  3. Breast Cancer")
        d = input("  Enter choice (1/2/3): ").strip()
        idx = int(d) - 1 if d in ["1","2","3"] else 0
        best_obj, scaler, feat_names, dname, labels = best_models[idx]
        predict_disease(best_obj, scaler, feat_names, dname, labels)

    print("\n" + "="*60)
    print("  ✅ Disease Prediction — Complete!")
    print("  📊 Check 'eda_disease.png' and 'disease_results.png'")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
