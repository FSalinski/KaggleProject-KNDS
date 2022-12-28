import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv('./Data/preprocessed2-train-bank-data.csv', sep=';')
X_train, y_train = train_data.drop('y', axis=1), train_data['y']

rfc = RandomForestClassifier(max_depth=8, class_weight='balanced')
rfc.fit(X_train, y_train)

with open('./Models/Serialized_models/random_forest_classifier_prototype.pickle', 'wb') as file:
    pickle.dump(rfc, file)