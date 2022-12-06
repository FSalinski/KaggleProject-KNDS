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


if __name__ == '__main__':
    main()