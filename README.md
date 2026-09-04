# Student Placement Eligibility Prediction

## 📌 Project Overview

This project uses Machine Learning to predict whether a student is eligible for placement based on academic performance, attendance, coding skills, projects, internship experience, and backlogs.

The project is developed as part of an ML internship at Learn Depth Academy.

## 🎯 Objective

The main objective is to build a binary classification model that predicts:

- `1` → Eligible for placement
- `0` → Not eligible for placement

## 📊 Dataset

The dataset contains 1,000 student records and 6 input features.

### Features

| Feature | Description |
|---|---|
| `cgpa` | Student's CGPA |
| `attendance_pct` | Attendance percentage |
| `coding_score` | Coding assessment score |
| `projects_completed` | Number of completed projects |
| `internship_months` | Internship experience in months |
| `backlogs` | Number of academic backlogs |

### Target

`target`

- `1` = Eligible
- `0` = Not Eligible

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Logistic Regression

## 🔄 Machine Learning Workflow

1. Load the dataset
2. Inspect the dataset
3. Check missing values
4. Check duplicate records
5. Check target class balance
6. Separate features and target
7. Split data into training and testing sets
8. Scale the features using StandardScaler
9. Train Logistic Regression model
10. Make predictions
11. Evaluate model performance
12. Analyze feature coefficients
13. Generate confusion matrix
14. Generate ROC curve

## ⚙️ Model Configuration

**Algorithm:** Logistic Regression

**Train-Test Split:** 80% training / 20% testing

**Random State:** 42

**Maximum Iterations:** 1000

**Feature Scaling:** StandardScaler

## 📈 Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 70.00% |
| Precision | 68.87% |
| Recall | 73.00% |
| F1-Score | 70.87% |
| ROC-AUC | 78.74% |

## 🔍 Confusion Matrix

The model produced the following confusion matrix:

```text
[[67 33]
 [27 73]]