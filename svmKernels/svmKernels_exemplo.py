import warnings

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import platform

warnings.filterwarnings("ignore")

pathWindows = 'C:/Users/Pc/PycharmProjects/TCC/dataset/'
pathLinux = '/home/gianluca/Documentos/Projetos/Pycharm Projects/TCC_IA/dataset/'
path = pathLinux if platform.system() == 'Linux' else pathWindows

dataset = pd.read_csv(path + 'pdbbind-2007-refined-core-yx36i.csv')

params = dataset.iloc[0:1105, :37]
labels = dataset.iloc[0:1105, 37:]

X_train, X_test, Y_train, Y_test = train_test_split(params, labels, train_size=0.5, random_state=1)

svc = svm.SVC()
svc.fit(X_train, Y_train)
y_pred = svc.predict(X_test)
print('Accuracy Score: ')
print(metrics.accuracy_score(Y_test, y_pred))

svc = svm.SVC(kernel='linear')
svc.fit(X_train, Y_train)
y_pred = svc.predict(X_test)
print('Accuracy Score : ')
print(metrics.accuracy_score(Y_test, y_pred))

svc = svm.SVC(kernel='rbf')
svc.fit(X_train, Y_train)
y_pred = svc.predict(X_test)
print('Accuracy Score : ')
print(metrics.accuracy_score(Y_test, y_pred))

svc = svm.SVC(kernel='poly')
svc.fit(X_train, Y_train)
y_pred = svc.predict(X_test)
print('Accuracy Score : ')
print(metrics.accuracy_score(Y_test, y_pred))

