import pandas as pd
import numpy as np
from sklearn import preprocessing
import scipy.stats

import warnings
warnings.filterwarnings("ignore")    

def main():
    train_data = pd.read_csv('./Data/cleaned-train-bank-data.csv', sep=';')
    test_data = pd.read_csv('./Data/cleaned-test-bank-data.csv', sep=';')

    # Previous column has only 0 values and outliers
    print("*Dropping 'previous' column*")
    train_data.drop(columns=['previous'], axis=1, inplace=True)
    test_data.drop(columns=['previous'], axis=1, inplace=True)

    features_to_scale = ['age','campaign','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']

    # Replacing outliers with winsorized mean

    print("\n*Winsorizing features*") 
    for column in features_to_scale:
        iqr = scipy.stats.iqr(train_data[column])
        q1 = scipy.stats.scoreatpercentile(train_data[column], 25)
        q3 = scipy.stats.scoreatpercentile(train_data[column], 75)

        floor = q1 - iqr
        ceil = q3 + iqr

        min_limit = scipy.stats.percentileofscore(train_data[column], floor) / 100
        max_limit = scipy.stats.percentileofscore(train_data[column], ceil) / 100

        scipy.stats.mstats.winsorize(train_data[column], limits=[min_limit, 1 - max_limit], inplace=True)
        scipy.stats.mstats.winsorize(test_data[column], limits=[min_limit, 1 - max_limit], inplace=True)
    
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

    scalers = [preprocessing.MinMaxScaler() for column in features_to_scale]

    for i, column in enumerate(features_to_scale):
        fit = scalers[i].fit(train_data[[column]])
        train_data[column] = scalers[i].fit_transform(train_data[[column]], fit)
        test_data[column] = scalers[i].fit_transform(test_data[[column]], fit)
    
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