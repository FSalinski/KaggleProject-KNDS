import pickle
import pandas as pd
from sklearn.neural_network import MLPClassifier

train_data = pd.read_csv('./Data/preprocessed2-train-bank-data.csv', sep=';')
X_train, y_train = train_data.drop('y', axis=1), train_data['y']

mlpc = MLPClassifier(hidden_layer_sizes=(110,), max_iter=13, power_t=0.4,
              random_state=1)
mlpc.fit(X_train, y_train)

with open('./Models/Serialized_models/MLP_classifier_gs.pickle', 'wb') as file:
    pickle.dump(mlpc, file)