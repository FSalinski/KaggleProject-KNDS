import pandas as pd

data = pd.read_csv('./bank-data.csv', sep=';') # importing raw data

#  Dropping columns we decided to drop in EDA

print("Shape of data:", data.shape)
data.drop(columns=['duration', 'default'], axis=1, inplace=True)
print("*Columns dropped*\nNew shape of data:", data.shape)

# Inputting missing data

data['pdays'] = data['pdays'] < 999
# pdays is True when a client was contacted in the previous campaign and False otherwise

data.rename({'pdays':'contacted.in.previous'}, inplace=True)

# Now we will get rid of 'unknown' values in categorical columns
# How many records has any 'unknown' value?
data_with_unknowns = data[(data['marital'] == 'unknown') |
                            (data['housing'] == 'unknown') |
                            (data['loan'] == 'unknown') |
                            (data['job'] == 'unknown') ]
print("Records with any 'unknown':", len(data_with_unknowns))
print("Describtion of data with any unknowns:\n", data_with_unknowns.describe())

# Dropping records with unknowns

data.drop(axis=0, index=data_with_unknowns.index, inplace=True)

# Exporting cleaned data to csv file

data.to_csv('./Data cleaning and feature engineering/cleaned-bank-data.csv', index=False, sep=';')