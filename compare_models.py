import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# Load the MNIST dataset
mnist = datasets.fetch_openml("mnist_784")
X, y = mnist.data, mnist.target
X = X / 255.0  # Normalize the pixel values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning for SVM
svm_param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
}
svm = SVC()
svm_grid_search = GridSearchCV(svm, svm_param_grid, cv=3)
svm_grid_search.fit(X_train, y_train)
svm_best_model = svm_grid_search.best_estimator_

# Hyperparameter tuning for Decision Tree
dt_param_grid = {
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
}
dt = DecisionTreeClassifier()
dt_grid_search = GridSearchCV(dt, dt_param_grid, cv=3)
dt_grid_search.fit(X_train, y_train)
dt_best_model = dt_grid_search.best_estimator_

# Model evaluation
production_predictions = svm_best_model.predict(X_test)
candidate_predictions = dt_best_model.predict(X_test)

production_accuracy = accuracy_score(y_test, production_predictions)
candidate_accuracy = accuracy_score(y_test, candidate_predictions)

production_confusion_matrix = confusion_matrix(y_test, production_predictions)
candidate_confusion_matrix = confusion_matrix(y_test, candidate_predictions)

macro_f1_production = f1_score(y_test, production_predictions, average='macro')
macro_f1_candidate = f1_score(y_test, candidate_predictions, average='macro')

print("Production Model Accuracy:", production_accuracy)
print("Candidate Model Accuracy:", candidate_accuracy)
print("Production Model Confusion Matrix:")
print(production_confusion_matrix)
print("Candidate Model Confusion Matrix:")
print(candidate_confusion_matrix)
print("Macro-average F1 Score (Production Model):", macro_f1_production)
print("Macro-average F1 Score (Candidate Model):", macro_f1_candidate)
