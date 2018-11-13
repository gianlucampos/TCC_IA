import warnings
import platform
import pandas as pd
import numpy as np
from sklearn import svm, utils
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt


warnings.filterwarnings("ignore")
pathWindows = 'C:/Users/Pc/PycharmProjects/TCC/dataset/'
pathLinux = '/home/gianluca/Documentos/Projetos/Pycharm Projects/TCC_IA/dataset/'
path = pathLinux if platform.system() == 'Linux' else pathWindows
dataset_Train = pd.read_csv(path + 'pdbbind-2007-refined-core-yx36i.csv')
dataset_Test = pd.read_csv(path + 'pdbbind-2007-core-yx36i.csv')
le = LabelEncoder()
le.fit(dataset_Train['PDB'].astype(str))
dataset_Train['PDB'] = le.transform(dataset_Train['PDB'].astype(str))
le.fit(dataset_Test['PDB'].astype(str))
dataset_Test['PDB'] = le.transform(dataset_Test['PDB'].astype(str))
# =========================================================================================
# Treinamento: Prevendo complexo de acordo com parâmetros passados:
print('=' * 240)
print('TREINAMENTO, sem validação (decorando)')
params, labels = dataset_Train.loc[:, dataset_Train.columns != 'pbindaff'], dataset_Train.loc[:, 'pbindaff']
x = np.array(params.values)
y = np.array(labels.values.tolist())

print('Informação: ', x[0].tolist())
print('Resposta Correta: ', y[0])

y_encoded = le.fit_transform(y)

clf = svm.SVC(kernel='rbf', C=1.0, gamma=0.1)
# clf = svm.SVC(kernel='linear', C=1.0, gamma=0.1)
# clf = svm.SVC(kernel='poly', C=1.0, gamma=0.1)
clf.fit(x, y)
print('Predição: ', clf.predict([x[0]]))
print('Score: ', clf.score(x, y))

# =========================================================================================
# Balanceamento de base com train_test_split e k-fold validation # A FAZER
# print('Score', clf.score(X_test, Y_test))
# =========================================================================================
# Teste:
print('=' * 240)
print('TESTE')
params_test, labels_test = dataset_Test.loc[:, dataset_Test.columns != 'pbindaff'], dataset_Test.loc[:, 'pbindaff']
x_test = np.array(params_test.values)
y_test = np.array(labels_test.values.tolist())

print('Informação: ', x_test[0].tolist())
print('Resposta Correta: ', y_test[0])

# clf.fit(x, y_test)
print('Predição: ', clf.predict([x_test[0]]))
print('Score: ', clf.score(x_test, y_test))
