## Supervised Learning Project

### 1. **Grid Search with AdaBoostClassifier**
The project now includes a **GridSearchCV** implementation to tune the hyperparameters of the `AdaBoostClassifier`. The following steps were added:
- **Parameter Grid**:
  ```python
  param_grid = {
      'estimator': [DecisionTreeClassifier(max_depth=2, max_features=None, class_weight=None),
                    DecisionTreeClassifier(max_depth=2, max_features='sqrt', class_weight='balanced')],
      'n_estimators': [200, 250, 300],
      'learning_rate': [0.02, 0.04, 0.06],
      'random_state': [42]
  }
  ```
- **Training and Evaluation**:
  - Used `GridSearchCV` with cross-validation (`cv=3`) to find the best hyperparameters.
  - Measured and printed the total time taken for grid search.
  - Evaluated the best estimator on training and testing data using the `evaluate_model` function.

---

### 2. **Stacking Classifier**
A new **StackingClassifier** was implemented to combine multiple base learners and a meta-model. Details:
- **Base Learners**:
  - Random Forest (`RandomForestClassifier`)
  - Gradient Boosting (`GradientBoostingClassifier`)
- **Meta-Model**:
  - Logistic Regression (`LogisticRegression` with `C=0.1`)
- **Workflow**:
  - Trained the stacking model on `X_train` and `y_train`.
  - Evaluated its performance using `evaluate_model`.

---

### 3. **Evaluation of Multiple Models**
The following machine learning models were trained and evaluated:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Classifier (SVC)
- Decision Tree
- Random Forest
- AdaBoostClassifier

Each model was evaluated using a loop, and the results were printed for training and testing sets. The `evaluate_model` function was used for consistency in metric reporting.

---

### Updated Workflow Overview:
1. Preprocessing and scaling (if applicable).
2. Hyperparameter tuning using `GridSearchCV` for AdaBoost.
3. Building and evaluating a **StackingClassifier** with multiple base models.
4. Iterative evaluation of individual models for comparison.


## Summary of Results

### 1. **Grid Search with AdaBoostClassifier**
- **Train Accuracy**: 94.76%  
- **Test Accuracy**: 87.25%  
- **Train Confusion Matrix**:
  ```
  [[182   5   5]
   [ 13 354  10]
   [  0   6 169]]
  ```
- **Test Confusion Matrix**:
  ```
  [[25  1  0]
   [ 6 40  6]
   [ 0  0 24]]
  ```
- **Precision (Train/Test)**: 94.06% / 86.07%  
- **Recall (Train/Test)**: 95.08% / 91.03%  
- **F1 Score (Train/Test)**: 94.54% / 87.54%  

### 2. **Stacking Model**
- **Train Accuracy**: 98.52%  
- **Test Accuracy**: 95.10%  
- **Train Confusion Matrix**:
  ```
  [[189   1   2]
   [  2 372   3]
   [  0   3 172]]
  ```
- **Test Confusion Matrix**:
  ```
  [[26  0  0]
   [ 1 48  3]
   [ 0  1 23]]
  ```
- **Precision (Train/Test)**: 98.35% / 94.24%  
- **Recall (Train/Test)**: 98.47% / 96.05%  
- **F1 Score (Train/Test)**: 98.41% / 95.05%  

---

### 3. **Comparison of Simple Models**

| Model                     | Train Accuracy | Test Accuracy | Train F1 Score | Test F1 Score |
|---------------------------|----------------|---------------|----------------|---------------|
| Logistic Regression       | 95.30%         | 97.06%        | 95.04%         | 97.01%        |
| K-Nearest Neighbors       | 95.83%         | 92.16%        | 95.40%         | 92.20%        |
| Support Vector Classifier | 97.98%         | 98.04%        | 97.78%         | 98.01%        |
| Decision Tree             | 95.70%         | 88.24%        | 95.41%         | 88.13%        |
| Random Forest             | 97.31%         | 93.14%        | 97.21%         | 93.14%        |
| AdaBoost                  | 85.22%         | 75.49%        | 84.25%         | 73.98%        |

---

### Conclusion
1. **Stacking Classifier** delivered the best overall performance, achieving high test accuracy (95.10%) with balanced precision, recall, and F1 scores.
2. **Logistic Regression** and **SVC** also performed well, showing excellent generalization with test accuracies above 97%.
3. **AdaBoostClassifier**, despite hyperparameter tuning, underperformed relative to other models, highlighting the importance of model selection based on data characteristics.
4. **Decision Tree** had lower test accuracy compared to ensemble models, indicating susceptibility to overfitting without additional regularization.

This analysis demonstrates the effectiveness of ensemble techniques like stacking and the necessity of hyperparameter tuning to optimize model performance.
