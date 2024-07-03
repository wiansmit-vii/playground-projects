import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score

'''
The following relative import works because Python allows relative imports 
within packages. Because random_forest_model.py and data_prep.py are within the 
same package structure, Python can resolve the import based on relative paths

For example:

If random_forest_model.py is in a directory named models/ and data_prep.py is 
in a sibling directory named data/, Python treats models/ and data/ as parts 
of the same package.

Python treats directories with __init__.py files as packages. If models/ and 
data/ are structured as packages (i.e., they contain __init__.py files), Python 
considers them part of the same package namespace.
'''
from data_prep import train_data_cleaned, test_data_cleaned

# BUILD AND TRAIN THE MODEL
# Train / Validatoin Split
train_data_split, validation_data_split = train_test_split(train_data_cleaned, test_size=0.2, random_state=42)

# Confirming splits
train_data_split.info()
validation_data_split.info()

# Separate features and target
X_train_data = train_data_split.drop('Survived', axis=1)
y_train_data = train_data_split['Survived']
X_validation_data = validation_data_split.drop('Survived', axis=1)
y_validation_data = validation_data_split['Survived']

# Random Forest Model
#rfc = RandomForestClassifier(n_estimators=200)
#rfc.fit(X_train_data,y_train_data)

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

'''
Best Parameters:    {'max_depth': 10, 
                    'max_features': 'log2', 
                    'min_samples_leaf': 2, 
                    'min_samples_split': 2, 
                    'n_estimators': 100}
'''

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


# WHEN MODEL IS TUNED TO CROSS-VALIDATION, TEST ON UNSEEN VALIDATION ATA 
# Evaluate the model on the validation set
y_val_pred = best_model.predict(X_validation_data)
val_accuracy = accuracy_score(y_validation_data, y_val_pred)
print(f'Validation Accuracy: {val_accuracy}')

'''
Validation Accuracy: 0.8202247191011236

An accuracy of 82.02% on the Titanic dataset is quite good and indicates a 
well-performing model. To further improve, focus on refined feature engineering, 
thorough hyperparameter tuning, and potentially leveraging ensemble methods. Keep in 
mind that incremental improvements can sometimes be achieved through careful 
experimentation and validation.

'''

# Train on the entire training dataset for final predictions
best_model.fit(X, y)

# Fit final model for submission. 
X_test = test_data_cleaned.drop('PassengerId', axis=1)

# Make predictions on the test set
predictions = best_model.predict(X_test)

# Submit results to edit // Check this out TODO
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")