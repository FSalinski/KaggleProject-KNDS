import pandas as pd

data = pd.read_csv('C:/Users/frane/kaggleproject/bank-data.csv', sep=';') # importing raw data

#  Dropping columns we decided to drop in EDA

print(data.shape)
data.drop(columns=['duration'], axis=1, inplace=True)
print(data.shape)

# Imputting missing data
