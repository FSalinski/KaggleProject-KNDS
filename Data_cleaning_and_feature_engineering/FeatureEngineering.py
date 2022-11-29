import pandas as pd
import numpy as np
from sklearn import preprocessing
import scipy.stats

import warnings
warnings.filterwarnings("ignore")

def calculate_iqr(data, column):
    q1 = np.percentile(data[column], 25)
    q3 = np.percentile(data[column], 75)
    return q1, q3, 1.5 * (q3 - q1)

def drop_outliers(train_data, data, column):
    """
    Function drops outliers in data based on InterQuartile Range
    """
    q1, q3, iqr = calculate_iqr(train_data, column)
    outliers_data = data.loc[(data[column] < q1 - iqr) | (data[column] > q3 + iqr)]

    data.drop(axis=0, index=outliers_data.index, inplace=True)
    

def main():
    train_data = pd.read_csv('./Data/cleaned-train-bank-data.csv', sep=';')
    test_data = pd.read_csv('./Data/cleaned-test-bank-data.csv', sep=';')

    numerical_to_drop_outliers = ['age','campaign','previous','nr.employed']

    print("Shape of data: ", train_data.shape, test_data.shape)
    print("*Dropping records with outliers*")
    for column in numerical_to_drop_outliers:
        drop_outliers(train_data, train_data, column)
        drop_outliers(train_data, test_data, column)
    print("New shape of data: ", train_data.shape, test_data.shape)

    # Previous column has only 0 values after dropping outliers
    print("*Dropping 'previous' column*")
    train_data.drop(columns=['previous'], axis=1, inplace=True)
    test_data.drop(columns=['previous'], axis=1, inplace=True)
    
    features_to_scale = ['age','campaign','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']

    # Checking and removing skewness in data 

    for column in features_to_scale:
        print(f"\nSkewness in {column}: {scipy.stats.skew(train_data[column])}")

    print("\n*Normalizing distributions*")

    pt = preprocessing.PowerTransformer(method='yeo-johnson')

    for column in features_to_scale:
        train_data[column] = pt.fit_transform(train_data[[column]], pt.fit(train_data[[column]]))
        test_data[column] = pt.fit_transform(test_data[[column]], pt.fit(train_data[[column]]))

    for column in features_to_scale:
        print(f"\nNew skewness in {column}: {scipy.stats.skew(train_data[column])}")

    # Scaling features to [0,1] range

    print("\n*Scaling features*")

    scaler = preprocessing.MinMaxScaler()
    
    for column in features_to_scale:
        train_data[column] = scaler.fit_transform(train_data[[column]], scaler.fit(train_data[[column]]))
        test_data[column] = scaler.fit_transform(test_data[[column]], scaler.fit(train_data[[column]]))
    
    print("\n*Checking data*")
    print("Train data description:\n", train_data.describe(), "\n")
    print("Train data head:\n", train_data.head(), "\n")
    print("Test data description:\n", test_data.describe(), "\n")
    print("Test data head:\n", test_data.head())

    print("*Exporting data to csv files*")

    train_data.to_csv('./Data/preprocessed-train-bank-data.csv', index=False, sep=';')
    test_data.to_csv('./Data/preprocessed-test-bank-data.csv', index=False, sep=';')
      

if __name__ == '__main__':
    main()