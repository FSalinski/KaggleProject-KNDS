import pickle
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

train_data = pd.read_csv('./Data/preprocessed2-train-bank-data.csv', sep=';')
X_train, y_train = train_data.drop('y', axis=1), train_data['y']

dtc = DecisionTreeClassifier(class_weight='balanced', max_depth=6,
                       min_samples_split=4, random_state=1, splitter='random')
dtc.fit(X_train, y_train)

with open('./Models/Serialized_models/decision_tree_classifier_gs.pickle', 'wb') as file:
    pickle.dump(dtc, file)