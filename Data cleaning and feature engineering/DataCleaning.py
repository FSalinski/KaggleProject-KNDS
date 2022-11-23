import pandas as pd

data = pd.read_csv('C:/Users/frane/kaggleproject/bank-data.csv', sep=';') # importing raw data

#  Dropping columns we decided to drop in EDA

data.drop(columns=['duration', 'default'], axis=1, inplace=True)

# Inputting missing data

data['pdays'] = data['pdays'] < 999

# pdays is True when a client was contacted in the previous campaign and False otherwise
data.rename({'pdays':'contacted.in.previous'}, inplace=True)

# Now we will get rid off 'unknown' values in categorical columns