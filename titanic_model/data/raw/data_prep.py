# Import train_data manipulation libraies
import numpy as np
import pandas as pd

# Import visualisation libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Import the train_data
train_data = pd.read_csv("./train.csv")
test_data = pd.read_csv("./test.csv")

# EXPLORATORY DATA ANALYSIS ON TRAIN_DATA DATA ONLY TO:
# Avoid Data Leakage
# Realistic Modeling

# Use a heatmap to visualise missing train_data
sns.heatmap(train_data.isnull(), yticklabels=False, cbar=False, cmap='viridis')

'''
Age: Small enough for reasonable replacement with some form of imputation. 
Cabin: Too little train_data to work with. 
'''

sns.set_style('whitegrid')

# Ratio of target labels
sns.countplot(x="Survived", data = train_data)

# Survival with a hue of sex
sns.countplot(x="Survived", hue= "Sex", data = train_data)

# Survival with a hue of passanger class
sns.countplot(x="Survived", hue= "Pclass", data = train_data)

# Idea of age of people of the Titanic
sns.distplot(train_data['Age'].dropna(), kde=False, bins=30)
'''
Bimodel distribution
'''

train_data.info()

# Investigate the SibSp feature
sns.countplot(x='SibSp', data=train_data)
'''
Most people did not have spouces or children on board.
'''

# Fare Distribution
train_data['Fare'].hist(bins=40, figsize=(10,4))
'''
As most passangers were in the third class it makes sense that is is skewed to
the the cheaper side. 
'''

# We could also do this as an interactive plot

import cufflinks as cf 
cf.go_offline()

train_data["Fare"].iplot(kind='hist', bins=50)

# DATA CLEANING
# We are assuming there is consistency between the two data sets 
# Applying imputation from training_data over both data sets

# Use a heatmap to visualise missing data
sns.heatmap(train_data.isnull(), yticklabels=False, cbar=False, cmap='viridis')

# IMPUTATION

# Let's first visualise this to see what we can expect
plt.figure(figsize=(10,7))
sns.boxplot(x="Pclass", y="Age", data=train_data)

# Function returning the mean age by Pclass for all the missing values. 
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return train_data[train_data["Pclass"] == 1]['Age'].mean()
        elif Pclass == 2:
            return train_data[train_data["Pclass"] == 2]['Age'].mean()
        else:
            return train_data[train_data["Pclass"] == 3]['Age'].mean()
    else:
        return Age

# Applying imputation from training_data over both data sets
train_data['Age'] = train_data[['Age','Pclass']].apply(impute_age, axis=1)
test_data['Age'] = train_data[['Age','Pclass']].apply(impute_age, axis=1)

# Chacking the null identifying heat map again
sns.heatmap(train_data.isnull(), yticklabels=False, cbar=False, cmap='viridis')
sns.heatmap(test_data.isnull(), yticklabels=False, cbar=False, cmap='viridis')

# Drop Missing Data over both datasets except for the target column in test_data 
train_data.drop('Cabin', axis=1, inplace=True)
test_data.drop('Cabin', axis=1, inplace=True)

train_data.dropna(inplace=True)
test_data.dropna(inplace=True)



# ONE-HOT ENCODING
# Prepare train_data for train_dataing by setting dummy variables

train_sex = pd.get_dummies(train_data['Sex'], drop_first=True)
test_sex = pd.get_dummies(test_data['Sex'], drop_first=True)
train_embark = pd.get_dummies(train_data['Embarked'], drop_first=True)
test_embark = pd.get_dummies(test_data['Embarked'], drop_first=True)

# Convert the boolean True/False to integers 1/0
train_sex = sex.astype(int)
test_sex = sex.astype(int)
train_embark = embark.astype(int)
test_embark = embark.astype(int)

# Concatenate encoded columns to train_dataframe
train_data_cleaned = pd.concat([train_data, train_sex, train_embark], axis=1)
test_data_cleaned = pd.concat([test_data, test_sex, test_embark], axis=1)

# Drop unneseccsary columns
train_data_cleaned.drop(['Sex', 'Embarked', 'Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)
test_data_cleaned.drop(['Sex', 'Embarked', 'Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)


# TODO
# Check get_dummies on Pclass and effect on results ---------------------- 
# Scale features

# BUILD AND TRAIN THE MODEL

X_train_data = train_data_cleaned.drop('Survived', axis=1)
y_train_data = train_data_cleaned['Survived']

# Random Forest Model
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train_data,y_train_data)

# Predictions
predictions = rfc.predict(test_data_cleaned)

# Submit results to edit
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")