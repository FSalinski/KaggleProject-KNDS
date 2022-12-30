import pickle
import pandas as pd
from sklearn.svm import LinearSVC

train_data = pd.read_csv('./Data/preprocessed2-train-bank-data.csv', sep=';')
X_train, y_train = train_data.drop('y', axis=1), train_data['y']

svc = LinearSVC(class_weight='balanced', dual=False, penalty='l1', random_state=1)
svc.fit(X_train, y_train)

with open('./Models/Serialized_models/SVC_gs.pickle', 'wb') as file:
    pickle.dump(svc, file)