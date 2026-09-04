# ============================================================
# Student Placement Eligibility Prediction
# Machine Learning Internship Project
# Algorithm: Logistic Regression
# ============================================================

# 1. Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)


# ============================================================
# 2. Load the dataset
# ============================================================

df = pd.read_csv("dataset/dataset_07_student_placement_eligibility.csv")

print("\n================ DATASET LOADED ================\n")

print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())


# ============================================================
# 3. Basic Data Inspection
# ============================================================

print("\n================ DATA INFORMATION ================\n")

print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nTarget class distribution:")
print(df["target"].value_counts())

print("\nTarget class percentage:")
print(df["target"].value_counts(normalize=True) * 100)

print("\nStatistical summary:")
print(df.describe())


# ============================================================
# 4. Separate Features and Target
# ============================================================

X = df.drop("target", axis=1)
y = df["target"]

print("\n================ FEATURES AND TARGET ================\n")

print("Features:")
print(X.columns.tolist())

print("\nTarget:")
print("0 = Not Eligible")
print("1 = Eligible")


# ============================================================
# 5. Train-Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n================ TRAIN-TEST SPLIT ================\n")

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ============================================================
# 6. Feature Scaling
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 7. Train Logistic Regression Model
# ============================================================

model = LogisticRegression(
    random_state=42,
    max_iter=1000
)

model.fit(X_train_scaled, y_train)

print("\n================ MODEL TRAINED ================\n")

print("Algorithm: Logistic Regression")
print("Random State: 42")
print("Maximum Iterations: 1000")


# ============================================================
# 8. Make Predictions
# ============================================================

y_pred = model.predict(X_test_scaled)

# Probability of being eligible
y_probability = model.predict_proba(X_test_scaled)[:, 1]


# ============================================================
# 9. Model Evaluation
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_probability)

print("\n================ MODEL PERFORMANCE ================\n")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")


# ============================================================
# 10. Classification Report
# ============================================================

print("\n================ CLASSIFICATION REPORT ================\n")

print(classification_report(
    y_test,
    y_pred,
    target_names=["Not Eligible", "Eligible"]
))


# ============================================================
# 11. Confusion Matrix
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("\n================ CONFUSION MATRIX ================\n")

print(cm)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Eligible", "Eligible"],
    yticklabels=["Not Eligible", "Eligible"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Student Placement Eligibility")

plt.tight_layout()
plt.savefig("results/confusion_matrix.png")
plt.show()


# ============================================================
# 12. Feature Coefficients
# ============================================================

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

coefficients["Absolute_Coefficient"] = (
    coefficients["Coefficient"].abs()
)

coefficients = coefficients.sort_values(
    by="Absolute_Coefficient",
    ascending=False
)

print("\n================ FEATURE COEFFICIENTS ================\n")

print(coefficients)


# ============================================================
# 13. Feature Coefficient Visualization
# ============================================================

plt.figure(figsize=(9, 6))

sns.barplot(
    data=coefficients,
    x="Coefficient",
    y="Feature"
)

plt.axvline(
    x=0,
    color="black",
    linestyle="--"
)

plt.title("Logistic Regression Feature Coefficients")
plt.xlabel("Coefficient")
plt.ylabel("Feature")

plt.tight_layout()
plt.savefig("feature_coefficients.png")
plt.show()


# ============================================================
# 14. ROC Curve
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {roc_auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Student Placement Eligibility")

plt.legend()
plt.tight_layout()

plt.savefig("roc_curve.png")
plt.show()


# ============================================================
# 15. Final Summary
# ============================================================

print("\n================ FINAL SUMMARY ================\n")

print("Student Placement Eligibility Prediction")
print("------------------------------------------")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

print("\nProject completed successfully!")
print("The model predicts whether a student is eligible for placement.")