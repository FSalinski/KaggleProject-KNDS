import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression

train_data = pd.read_csv('./Data/preprocessed2-train-bank-data.csv', sep=';')
X_train, y_train = train_data.drop('y', axis=1), train_data['y']

logr = LogisticRegression(class_weight='balanced', max_iter=90, random_state=1)
logr.fit(X_train, y_train)

with open('./Models/Serialized_models/logistic_regression_gs.pickle', 'wb') as file:
    pickle.dump(logr, file)