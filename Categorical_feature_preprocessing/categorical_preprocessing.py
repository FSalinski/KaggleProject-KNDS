import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def main():
    train_data = pd.read_csv('./Data/preprocessed-train-bank-data.csv', sep=';')
    test_data = pd.read_csv('./Data/preprocessed-test-bank-data.csv', sep=';')

    features_to_encode = ['job','marital','education','housing','loan','contact','poutcome']

    for column in features_to_encode:
        train_data[column] = train_data[column].apply(lambda x: f"{column}_{x}") # Making all categories unique among all data 
        test_data[column] = test_data[column].apply(lambda x: f"{column}_{x}") 

    # One hot encoding some of categorical features

    print("Data description:\n", train_data.describe())
    print("\n*One hot encoding*")
    onehot = OneHotEncoder()

    onehot.fit(train_data[features_to_encode])
    train_one_hotted = onehot.transform(train_data[features_to_encode]).toarray()
    test_one_hotted = onehot.transform(test_data[features_to_encode]).toarray()

    categories = onehot.categories_
    categories = np.concatenate(categories) # array of all categories

    train_encoded_data = pd.DataFrame(columns=categories, data=train_one_hotted)
    test_encoded_data = pd.DataFrame(columns=categories, data=test_one_hotted)

    # dropping categorical columns and adding encoded categories

    train_data.drop(labels=features_to_encode, axis=1, inplace=True)
    test_data.drop(labels=features_to_encode, axis=1, inplace=True)
    train_data = train_data.drop('y', axis=1).join(train_encoded_data).join(train_data['y'])
    test_data = test_data.drop('y', axis=1).join(test_encoded_data).join(test_data['y'])

    print("\nData description:\n", train_data.describe())

    # Sine cosine encoding cyclical features

    print("\nUnique values in 'month':", train_data['month'].unique())
    print("Unique values in 'day_of_week':", train_data['day_of_week'].unique())

    print("\n*Sine/Cosine encoding*")

    for data in [train_data, test_data]: 

        data['month'].replace({'mar' : 3, 
                               'apr' : 4,
                               'may' : 5,
                               'jun' : 6,
                               'jul' : 7,
                               'aug' : 8,
                               'sep' : 9,
                               'oct' : 10,
                               'nov' : 11,
                               'dec' : 12}, inplace=True)

        data['day_of_week'].replace({'mon' : 1,
                                     'tue' : 2,
                                     'wed' : 3,
                                     'thu' : 4,
                                     'fri' : 5}, inplace=True)

        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12.0)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12.0)
        data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 5.0)
        data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 5.0)

        data.drop(['month', 'day_of_week'], axis=1, inplace=True)

    print("\nDescription of new sin/cos features:\n", train_data[['month_sin', 'month_cos', 'day_sin', 'day_cos']].describe())

    # Replacing True/False with 1/0 in 'contacted.in.previous'

    train_data['contacted.in.previous'] = train_data['contacted.in.previous'].apply(lambda x: 1. if x == True else 0.)
    test_data['contacted.in.previous'] = test_data['contacted.in.previous'].apply(lambda x: 1. if x == True else 0.)

    # Replacing yes/no with 1/0 in 'y'

    train_data['y'] = train_data['y'].apply(lambda x: 1. if x == 'yes' else 0.)
    test_data['y'] = test_data['y'].apply(lambda x: 1. if x == 'yes'else 0.)
    
    train_data = train_data.drop('y', axis=1).join(train_data['y'])
    test_data = test_data.drop('y', axis=1).join(test_data['y'])

    print("\n*Checking data*")
    print("Train data description:\n", train_data.describe(), "\n")
    print("Train data head:\n", train_data.head(), "\n")
    print("Test data description:\n", test_data.describe(), "\n")
    print("Test data head:\n", test_data.head())

    print("*Exporting data to csv files*")
    train_data.to_csv('./Data/preprocessed2-train-bank-data.csv', index=False, sep=';')
    test_data.to_csv('./Data/preprocessed2-test-bank-data.csv', index=False, sep=';')
    

if __name__ == '__main__':
    main()