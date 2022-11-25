import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    data = pd.read_csv('./Data/bank-data.csv', sep=';') 
    X = data.drop('y', axis=1)
    y = data['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

    #  Dropping columns we decided to drop in EDA

    print("Shape of data:", X_train.shape, X_test.shape)
    X_train.drop(columns=['duration', 'default'], axis=1, inplace=True)
    X_test.drop(columns=['duration', 'default'], axis=1, inplace=True)
    print("*Columns dropped*\nNew shape of data:", X_train.shape, X_test.shape)

    # Inputting missing data

    X_train['pdays'] = X_train['pdays'] < 999
    X_test['pdays'] = X_test['pdays'] < 999
    # pdays is True when a client was contacted in the previous campaign and False otherwise

    X_train.rename({'pdays':'contacted.in.previous'}, inplace=True)
    X_test.rename({'pdays':'contacted.in.previous'}, inplace=True)

    # Now we will get rid of 'unknown' values in categorical columns
    # How many records has any 'unknown' value?
    X_train_with_unknowns = X_train[(X_train['marital'] == 'unknown') |
                                (X_train['housing'] == 'unknown') |
                                (X_train['loan'] == 'unknown') |
                                (X_train['job'] == 'unknown') ]
    X_test_with_unknowns = X_test[(X_test['marital'] == 'unknown') |
                                (X_test['housing'] == 'unknown') |
                                (X_test['loan'] == 'unknown') |
                                (X_test['job'] == 'unknown') ]
    print("Records with any 'unknown':", len(X_train_with_unknowns + X_test_with_unknowns))
    print("Description of data with any unknowns:\n", X_train_with_unknowns.describe())
    print("\nCompared to all data:\n", X_train.describe())

    # Dropping records with unknowns

    X_train.drop(axis=0, index=X_train_with_unknowns.index, inplace=True)
    y_train.drop(axis=0, index=X_train_with_unknowns.index, inplace=True)
    X_test.drop(axis=0, index=X_test_with_unknowns.index, inplace=True)
    y_test.drop(axis=0, index=X_test_with_unknowns.index, inplace=True)
    print("*Records with unknowns dropped*")

    train_data = X_train.join(y_train)
    test_data = X_test.join(y_test)

    print("*Checking if the data is valid*")
    print("Train data:", train_data)
    print("Test data:", test_data)

    print("*Exporting data to csv files*")
    train_data.to_csv('./Data/cleaned-train-bank-data.csv', index=False, sep=';')
    test_data.to_csv('./Data/cleaned-test-bank-data.csv', index=False, sep=';')


if __name__ == '__main__':
    main()