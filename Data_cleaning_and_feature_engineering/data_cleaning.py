import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    data = pd.read_csv('./Data/bank-data.csv', sep=';') 
    X = data.drop('y', axis=1)
    y = data['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

    train_data = X_train.join(y_train)
    test_data = X_test.join(y_test)

    #  Dropping columns we decided to drop in EDA

    print("Shape of data:", train_data.shape, test_data.shape)
    train_data.drop(columns=['duration', 'default'], axis=1, inplace=True)
    test_data.drop(columns=['duration', 'default'], axis=1, inplace=True)
    print("*Columns dropped*\nNew shape of data:", train_data.shape, test_data.shape)

    # Binarizing the 'pdays' column

    train_data['pdays'] = train_data['pdays'] < 999
    test_data['pdays'] = test_data['pdays'] < 999
    
    # pdays is True when a client was contacted in the previous campaign and False otherwise

    train_data.rename({'pdays':'contacted.in.previous'}, inplace=True, axis=1)
    test_data.rename({'pdays':'contacted.in.previous'}, inplace=True, axis=1)

    # Now we will get rid of 'unknown' values in categorical columns
    
    train_data = train_data[(train_data['marital'] != 'unknown') &
                            (train_data['housing'] != 'unknown') &
                            (train_data['loan'] != 'unknown') &
                            (train_data['job'] != 'unknown') ]
    test_data = test_data[(test_data['marital'] != 'unknown') &
                            (test_data['housing'] != 'unknown') &
                            (test_data['loan'] != 'unknown') &
                            (test_data['job'] != 'unknown') ]    

    print("*Records with unknowns dropped*\nNew shape of data:", train_data.shape, test_data.shape)

    print("*Checking data*")
    print("Train data description:\n", train_data.describe(), "\n")
    print("Train data head:\n", train_data.head(), "\n")
    print("Test data description:\n", test_data.describe(), "\n")
    print("Test data head:\n", test_data.head())

    print("*Exporting data to csv files*")
    train_data.to_csv('./Data/cleaned-train-bank-data.csv', index=False, sep=';')
    test_data.to_csv('./Data/cleaned-test-bank-data.csv', index=False, sep=';')


if __name__ == '__main__':
    main()