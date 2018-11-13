import warnings

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics, linear_model
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import platform
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")

pathWindows = 'C:/Users/Pc/PycharmProjects/TCC/dataset/'
pathLinux = '/home/gianluca/Documentos/Projetos/Pycharm Projects/TCC_IA/dataset/'
path = pathLinux if platform.system() == 'Linux' else pathWindows
dataset_Train = pd.read_csv(path + 'pdbbind-2007-refined-core-yx36i.csv')
le = LabelEncoder()
le.fit(dataset_Train['PDB'].astype(str))
dataset_Train['PDB'] = le.transform(dataset_Train['PDB'].astype(str))
# le.fit(dataset_Test['PDB'].astype(str))
# dataset_Test['PDB'] = le.transform(dataset_Test['PDB'].astype(str))

params, labels = dataset_Train.loc[:, dataset_Train.columns != 'pbindaff'], dataset_Train.loc[:, 'pbindaff']

X_train, X_test, Y_train, Y_test = train_test_split(params, labels, train_size=0.5, random_state=1)

svc = svm.SVR()
svc.fit(X_train, Y_train)
y_pred = svc.predict(X_test)
print('Accuracy Score: ')
print(svc.score(Y_test, y_pred))

lm = linear_model.LinearRegression()
model = lm.fit(X_train, Y_train)
predictions = lm.predict(X_test)

print(predictions[0:5])
plt.scatter(Y_test, predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()
print("Score Before FIT:", model.score(X_test, Y_test))

# print(metrics.accuracy_score(Y_test, y_pred))

# svc = svm.SVC(kernel='linear')
# svc.fit(X_train, Y_train)
# y_pred = svc.predict(X_test)
# print('Accuracy Score : ')
# print(metrics.accuracy_score(Y_test, y_pred))
#
# svc = svm.SVC(kernel='rbf')
# svc.fit(X_train, Y_train)
# y_pred = svc.predict(X_test)
# print('Accuracy Score : ')
# print(metrics.accuracy_score(Y_test, y_pred))
#
# svc = svm.SVC(kernel='poly')
# svc.fit(X_train, Y_train)
# y_pred = svc.predict(X_test)
# print('Accuracy Score : ')
# print(metrics.accuracy_score(Y_test, y_pred))
#
