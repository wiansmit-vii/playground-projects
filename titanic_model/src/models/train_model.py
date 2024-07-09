import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load processed data
train_data_cleaned = pd.read_pickle("../../data/processed/train_data_engineered.pkl")
test_data_cleaned = pd.read_pickle("../../data/processed/test_data_engineered.pkl")

# --------------------------------------------------------------------------------------------
# BUILD AND TRAIN THE MODEL
# --------------------------------------------------------------------------------------------

# Train / Validation Split
train_data_split, validation_data_split = train_test_split(train_data_cleaned, test_size=0.2, random_state=42)

# Separate features and target
X_train_data = train_data_split.drop('Survived', axis=1)
y_train_data = train_data_split['Survived']
X_validation_data = validation_data_split.drop('Survived', axis=1)
y_validation_data = validation_data_split['Survived']

# Random Forest Model
model = RandomForestClassifier(random_state=42)

# Define parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['log2', 'sqrt', 0.5],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}

# Cross-validation setup
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=kf, scoring='accuracy', n_jobs=-1)

# Fit the GridSearchCV to the data
grid_search.fit(X_train_data, y_train_data)

# Print the best parameters found by GridSearchCV
print(f'Best Parameters: {grid_search.best_params_}')

# Get the best model from GridSearchCV
best_model = grid_search.best_estimator_

# Perform cross-validation and get the accuracy scores of the best model
cv_scores = cross_val_score(best_model, X_train_data, y_train_data, cv=kf, scoring='accuracy')

# Print the cross-validation scores and mean score
print(f'Cross-Validation Scores: {cv_scores}')
print(f'Mean Cross-Validation Accuracy: {np.mean(cv_scores)}')

'''
Cross-Validation Scores: [0.86713287 0.83802817 0.85211268 0.83802817 0.87323944]
Mean Cross-Validation Accuracy: 0.8537082635674185

Good overall performance with some variability: ~ 4%
'''

# WHEN MODEL IS TUNED TO CROSS-VALIDATION, TEST ON UNSEEN VALIDATION DATA 
# Evaluate the model on the validation set
y_val_pred = best_model.predict(X_validation_data)
val_accuracy = accuracy_score(y_validation_data, y_val_pred)
print(f'Validation Accuracy: {val_accuracy}')

# Train on the entire training dataset for final predictions
X_train_data_cleaned = train_data_cleaned.drop('Survived', axis=1)
y_train_data_cleaned = train_data_cleaned['Survived']

best_model.fit(X_train_data_cleaned, y_train_data_cleaned)

# Fit final model for submission
X_test = test_data_cleaned

# Make predictions on the test set
predictions = best_model.predict(X_test)

# Submit results 
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)

