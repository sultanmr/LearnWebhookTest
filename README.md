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
