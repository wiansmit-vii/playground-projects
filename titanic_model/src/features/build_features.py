import pandas as pd

# --------------------------------------------------------------------------------------------
# BUILD FEATURES FOR MODELLING
# --------------------------------------------------------------------------------------------

# Load processed train data
train_data = pd.read_pickle("../../data/interim/train_data_processed.pkl")
test_data = pd.read_pickle("../../data/interim/test_data_processed.pkl")

# ONE-HOT ENCODING
# Prepare data for training by setting dummy variables

def one_hot_encode_and_concat(data, columns):
    """One-hot encode specified columns and concatenate to the dataframe."""
    encoded_columns = []
    for column in columns:
        encoded = pd.get_dummies(data[column], drop_first=True).astype(int)
        encoded_columns.append(encoded)
    return pd.concat([data] + encoded_columns, axis=1)

# Encode specified columns
columns_to_encode = ['Sex', 'Embarked', 'Pclass']
train_data_cleaned = one_hot_encode_and_concat(train_data, columns_to_encode)
test_data_cleaned = one_hot_encode_and_concat(test_data, columns_to_encode)

# Drop unnecessary columns
columns_to_drop = ['Sex', 'Embarked', 'Name', 'Ticket', 'PassengerId', 'Pclass']
train_data_cleaned.drop(columns_to_drop, axis=1, inplace=True)
test_data_cleaned.drop(columns_to_drop, axis=1, inplace=True)

# Set all column names to strings
train_data_cleaned.columns = train_data_cleaned.columns.astype(str)
test_data_cleaned.columns = test_data_cleaned.columns.astype(str)

# Save the cleaned data to processed folder
train_data_cleaned.to_pickle("../../data/processed/train_data_engineered.pkl")
test_data_cleaned.to_pickle("../../data/processed/test_data_engineered.pkl")
