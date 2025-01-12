# Stacking Model for Classification

This repository contains a machine learning project that demonstrates the implementation of a stacking classifier using `RandomForestClassifier`, `GradientBoostingClassifier`, and `LogisticRegression` as the meta-model for classification tasks. The project includes model training, evaluation, and optimization to minimize overfitting and achieve better generalization.

## Table of Contents

1. [Project Description](#project-description)
2. [Data](#data)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Model Evaluation](#model-evaluation)
6. [License](#license)

## Project Description

This project focuses on stacking models to improve classification performance. The stacking classifier is an ensemble method that combines the predictions of several base learners (Random Forest and Gradient Boosting) using a logistic regression model as a meta-model. The model is optimized to reduce overfitting and improve generalization performance using cross-validation, hyperparameter tuning, and proper evaluation metrics.

### Key Features:

- **Stacked Classifier**: Combines multiple base classifiers (Random Forest and Gradient Boosting).
- **Meta-Model**: Uses Logistic Regression as a final estimator.
- **Hyperparameter Tuning**: Reduced complexity of base models to prevent overfitting.
- **Model Evaluation**: Measures accuracy, precision, recall, and F1-score, and outputs confusion matrices.
- **Cross-Validation**: To evaluate the model's generalization performance.

## Data

The dataset used in this project contains various features that describe objects and a categorical target variable (`class`). The features include metrics like compactness, circularity, aspect ratio, etc., while the target variable `class` contains categorical labels.

Please ensure that the dataset is loaded properly and preprocessed before training the models.

## Installation

To get started with this project, clone the repository and install the required dependencies.

### Clone the repository:
```bash
git clone https://github.com/your-username/stacking-model-classification.git
cd stacking-model-classification
```

### Install dependencies:
```bash
pip install -r requirements.txt
```

### Requirements:
- Python 3.x
- `sklearn`
- `matplotlib`
- `seaborn`
- `pandas`
- `numpy`

## Usage

### Training the Stacking Classifier:
```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

# Define base learners
base_learners = [
    ('rf', RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_split=10, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=50, learning_rate=0.05, random_state=42))
]

# Define meta-model
meta_model = LogisticRegression(max_iter=1000, C=0.1)

# Create stacking classifier
stacking_model = StackingClassifier(estimators=base_learners, final_estimator=meta_model)

# Train the model
stacking_model.fit(X_train, y_train)
y_pred = stacking_model.predict(X_test)
```

### Model Evaluation:
```python
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

def evaluate_model(model, X_train, y_train, X_test, y_test):
    # Predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Accuracy
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)

    # Confusion Matrix
    train_conf_matrix = confusion_matrix(y_train, train_pred)
    test_conf_matrix = confusion_matrix(y_test, test_pred)

    # Precision, Recall, F1-Score
    train_precision = precision_score(y_train, train_pred, average='macro', zero_division=1)
    train_recall = recall_score(y_train, train_pred, average='macro', zero_division=1)
    train_f1_score = f1_score(y_train, train_pred, average='macro', zero_division=1)

    test_precision = precision_score(y_test, test_pred, average='macro', zero_division=1)
    test_recall = recall_score(y_test, test_pred, average='macro', zero_division=1)
    test_f1_score = f1_score(y_test, test_pred, average='macro', zero_division=1)

    # Print results
    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)
    print("\nTrain Confusion Matrix:", train_conf_matrix)
    print("\nTest Confusion Matrix:", test_conf_matrix)
    print("\nTrain Precision:", train_precision)
    print("Train Recall:", train_recall)
    print("Train F1 Score:", train_f1_score)
    print("\nTest Precision:", test_precision)
    print("Test Recall:", test_recall)
    print("Test F1 Score:", test_f1_score)
```

Run the function `evaluate_model` to get a comprehensive evaluation of the model's performance.

## Model Evaluation

- **Accuracy**: Overall classification performance.
- **Precision**: Precision of positive class predictions.
- **Recall**: Recall of positive class predictions.
- **F1-Score**: Harmonic mean of precision and recall.
- **Confusion Matrix**: True positives, false positives, true negatives, and false negatives.


### Notes:

- **Cross-Validation**: You may also want to include a section on cross-validation, as it's an important step in evaluating model performance in a more robust way. If you're using it in your project, you can explain how it's used.
- **Model Tuning**: In case you're doing hyperparameter tuning, you can mention that this repository includes the process of finding the best parameters for your models.
