import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv('./Data/preprocessed2-train-bank-data.csv', sep=';')
X_train, y_train = train_data.drop('y', axis=1), train_data['y']

rfc = RandomForestClassifier(class_weight='balanced', criterion='entropy',
                       max_depth=9, min_samples_leaf=3, n_estimators=80,
                       random_state=1)
rfc.fit(X_train, y_train)

with open('./Models/Serialized_models/random_forest_classifier_gs.pickle', 'wb') as file:
    pickle.dump(rfc, file)