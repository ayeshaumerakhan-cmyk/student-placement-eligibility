# Student Placement Eligibility Prediction

## Overview

This project is a Machine Learning classification project that predicts whether a student is eligible for campus placement using **Logistic Regression**.

The model uses academic performance, attendance, coding skills, projects, internship experience, and backlogs to classify students as eligible or not eligible for placement.

---

## Objective

The main objective of this project is to build a binary classification model that predicts:

- `1` → Student is eligible for placement
- `0` → Student is not eligible for placement

The project also evaluates the model using multiple classification metrics and visualizations.

---

## Dataset

The dataset contains **1,000 student records** with 6 input features and 1 target variable.

### Features

| Feature | Description |
|---|---|
| `cgpa` | Student's CGPA |
| `attendance_pct` | Attendance percentage |
| `coding_score` | Coding assessment score |
| `projects_completed` | Number of completed projects |
| `internship_months` | Internship experience in months |
| `backlogs` | Number of academic backlogs |
| `target` | Placement eligibility (0 or 1) |

### Dataset Quality

- Total records: **1,000**
- Features: **6**
- Target variable: **1**
- Missing values: **0**
- Duplicate records: **0**
- Target classes: **500 eligible, 500 not eligible**
- Dataset is balanced.

---

## Machine Learning Algorithm

The project uses:

**Logistic Regression**

Logistic Regression is a supervised machine learning algorithm commonly used for binary classification problems.

In this project, it predicts the probability that a student belongs to the eligible placement class.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## Machine Learning Workflow

The project follows these steps:

1. Load the dataset
2. Inspect the dataset
3. Check for missing values
4. Check for duplicate records
5. Analyze target class balance
6. Separate input features and target
7. Split the data into training and testing sets
8. Train the Logistic Regression model
9. Make predictions on the test data
10. Evaluate model performance
11. Generate visualizations
12. Interpret the model results

---

## Train-Test Split

The dataset was divided into:

- **80% Training Data:** 800 records
- **20% Testing Data:** 200 records

The test set was kept separate during training to evaluate the model on unseen data.

---

## Model Configuration

The model used is:

```python
LogisticRegression(max_iter=1000)
The model was trained using the training dataset and evaluated using the testing dataset.

Model Performance

The trained model achieved the following results:

Metric	Score
Accuracy	70.00%
Precision	68.87%
Recall	73.00%
F1-Score	70.87%
ROC-AUC	78.74%

These results show that the model provides a reasonable baseline for predicting student placement eligibility.

Confusion Matrix

The confusion matrix obtained from the test data was:

[[67 33]
 [27 73]]

The matrix can be interpreted as:

67 → Correctly predicted not eligible
33 → Incorrectly predicted eligible
27 → Incorrectly predicted not eligible
73 → Correctly predicted eligible

The model correctly classified 140 out of 200 test records.

Model Outputs and Visualizations

The project generates the following visualizations:

1. Confusion Matrix

Displays the number of correct and incorrect predictions made by the model.

2. Feature Coefficients

Shows how each feature influences the Logistic Regression prediction.

3. ROC Curve

Shows the model's ability to distinguish between eligible and non-eligible students.

The generated plots are available in the results/ folder.

Feature Interpretation

The Logistic Regression coefficients help understand how the input features influence placement eligibility.

In general:

Higher CGPA can contribute to better placement eligibility.
Better attendance can contribute positively.
Higher coding scores can improve eligibility.
More completed projects can provide a positive contribution.
More internship experience can support eligibility.
Higher number of backlogs can negatively affect eligibility.

The exact effect of each feature can be observed from the generated feature coefficient visualization.

Project Structure
student-placement-eligibility/
│
├── dataset/
│   └── dataset_07_student_placement_eligibility.csv
│
├── results/
│   ├── confusion_matrix.png
│   ├── feature_coefficients.png
│   └── roc_curve.png
│
├── placement_prediction.py
├── requirements.txt
├── README.md
└── .gitignore

The .venv virtual environment is used locally for development and is not included in the GitHub repository.

How to Run the Project
1. Clone the repository
git clone <your-github-repository-url>
2. Open the project folder
cd student-placement-eligibility
3. Create a virtual environment
python -m venv .venv
4. Activate the virtual environment

For Windows:

.venv\Scripts\activate
5. Install the required libraries
pip install -r requirements.txt
6. Run the project
python placement_prediction.py

The model performance metrics and evaluation results will be displayed in the terminal, and the visualization files will be generated inside the results/ folder.

Limitations

This project uses a synthetic educational dataset, so the results may not represent real-world student placement outcomes.

Real placement decisions can depend on many additional factors, such as:

Communication skills
Interview performance
Technical interview results
Company requirements
Resume quality
Soft skills
Industry-specific requirements

Therefore, this model should be considered a learning project and baseline classification system rather than a real-world placement decision-making system.

Future Improvements

Possible improvements include:

Testing additional classification algorithms
Hyperparameter tuning
Feature scaling and preprocessing pipelines
Cross-validation
Feature selection
Testing on real-world datasets
Improving model interpretability
Building a simple user interface for predictions
Conclusion

This project demonstrates how Logistic Regression can be applied to a binary classification problem.

The model achieved an accuracy of 70% and a ROC-AUC score of 78.74% on the test dataset.

The project also demonstrates the complete machine learning workflow, from data inspection and preprocessing to model training, evaluation, visualization, and interpretation.

Author

Ayesha Umera

B.Tech – Artificial Intelligence & Machine Learning
