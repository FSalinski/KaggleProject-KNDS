import pickle
import pandas as pd
from sklearn.neural_network import MLPClassifier

train_data = pd.read_csv('C:/Users/frane/kaggleproject/Data/preprocessed2-train-bank-data.csv', sep=';')
X_train, y_train = train_data.drop('y', axis=1), train_data['y']

mlpc = MLPClassifier(max_iter=10)
mlpc.fit(X_train, y_train)

with open('./Models/Serialized_models/MLP_classifier_prototype.pickle', 'wb') as file:
    pickle.dump(mlpc, file)