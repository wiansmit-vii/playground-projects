# Import train_data manipulation libraies
import numpy as np
import pandas as pd

# Import visualisation libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Import the train_data
train_data = pd.read_csv("./train.csv")
test_data = pd.read_csv("./test.csv")

train_data.head()
test_data.head()

# DATA CLEANING
# We are assuming there is consistency between the two data sets 
# Applying imputation from training_data over both data sets

# Use a heatmap to visualise missing data
sns.heatmap(train_data.isnull(), yticklabels=False, cbar=False, cmap='viridis')
sns.heatmap(test_data.isnull(), yticklabels=False, cbar=False, cmap='viridis')

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
            return train_data[train_data["Pclass"] == 1]['Age'].mean().astype(int)
        elif Pclass == 2:
            return train_data[train_data["Pclass"] == 2]['Age'].mean().astype(int)
        else:
            return train_data[train_data["Pclass"] == 3]['Age'].mean().astype(int)
    else:
        return Age

# Applying imputation from training_data over both data sets
train_data['Age'] = train_data[['Age','Pclass']].apply(impute_age, axis=1)
test_data['Age'] = test_data[['Age','Pclass']].apply(impute_age, axis=1)

# Chacking the null identifying heat map again
sns.heatmap(train_data.isnull(), yticklabels=False, cbar=False, cmap='viridis')
sns.heatmap(test_data.isnull(), yticklabels=False, cbar=False, cmap='viridis')

# Drop Missing Data over both datasets except for the target column in test_data 
train_data.drop('Cabin', axis=1, inplace=True)
test_data.drop('Cabin', axis=1, inplace=True)

train_data.dropna(inplace=True)
test_data.dropna(inplace=True)

train_data.head()
test_data.head()

# ONE-HOT ENCODING
# Prepare train_data for train_dataing by setting dummy variables

train_sex = pd.get_dummies(train_data['Sex'], drop_first=True).astype(int)
test_sex = pd.get_dummies(test_data['Sex'], drop_first=True).astype(int)
train_embark = pd.get_dummies(train_data['Embarked'], drop_first=True).astype(int)
test_embark = pd.get_dummies(test_data['Embarked'], drop_first=True).astype(int)
# testing --
train_pclass = pd.get_dummies(train_data['Pclass'], drop_first=True).astype(int)
test_pclass = pd.get_dummies(test_data['Pclass'], drop_first=True).astype(int)

# Concatenate encoded columns to train_dataframe
train_data_cleaned = pd.concat([train_data, train_sex, train_embark, train_pclass], axis=1)
test_data_cleaned = pd.concat([test_data, test_sex, test_embark, test_pclass], axis=1)

# Drop unneseccsary columns
train_data_cleaned.drop(['Sex', 'Embarked', 'Name', 'Ticket', 'PassengerId', 'Pclass'], axis=1, inplace=True)
test_data_cleaned.drop(['Sex', 'Embarked', 'Name', 'Ticket', 'PassengerId', 'Pclass'], axis=1, inplace=True)

# Set all column names to strings
train_data_cleaned.columns = train_data_cleaned.columns.astype(str)
test_data_cleaned.columns = test_data_cleaned.columns.astype(str)