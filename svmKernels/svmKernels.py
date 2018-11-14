import warnings
import timeit
import platform
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")
# Setando timer para avaliar tempo de treino
inicio = timeit.default_timer()
# Caminho para datasets
pathWindows = 'C:/Users/Pc/PycharmProjects/TCC/dataset/'
pathLinux = '/home/gianluca/Documentos/Projetos/Pycharm Projects/TCC_IA/dataset/'
path = pathLinux if platform.system() == 'Linux' else pathWindows
# Convertendo datasets para dataframes
dataset_Train = pd.read_csv(path + 'pdbbind-2007-refined-core-yx36i.csv')
dataset_Test = pd.read_csv(path + 'pdbbind-2007-core-yx36i.csv')
# Convertendo Strings
le = LabelEncoder()
le.fit(dataset_Train['PDB'])
dataset_Train['PDB'] = le.transform(dataset_Train['PDB'])
le.fit(dataset_Test['PDB'])
dataset_Test['PDB'] = le.transform(dataset_Test['PDB'])
# =========================================================================================
# Treinamento: Prevendo complexo de acordo com parâmetros passados:
print('=' * 240)
print('TREINAMENTO, sem validação (decorando)')
params, labels = dataset_Train.loc[:, dataset_Train.columns != 'pbindaff'], dataset_Train.loc[:, 'pbindaff']
# Passando valores flutuantes para inteiros e
params = params.astype(int)
labels *= 1000
labels = labels.astype(int)
x = np.array(params.values)
y = np.array(labels.values.tolist())

print('Informação: ', x[0].tolist())
print('Resposta Correta: ', y[0])

clf = svm.SVC()
# clf = svm.SVC(kernel='linear')
# clf = svm.SVC(kernel='poly')
# clf = svm.SVC(kernel='rbf')
# clf = svm.SVC(kernel='sigmoid')

clf = svm.SVR(kernel='poly', degree=3, gamma=0.01, coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True,
              cache_size=200, verbose=False)

clf.fit(x, y)
print('Predição: ', clf.predict([x[0]]))
print('Score: ', clf.score(x, y))

fim = timeit.default_timer()
print('Tempo: %f segundos' % (fim - inicio))
# =========================================================================================
# Balanceamento de base com train_test_split e k-fold validation # A FAZER
# print('Score', clf.score(X_test, Y_test))
# =========================================================================================
# Teste:
print('=' * 240)
print('TESTE')
params_test, labels_test = dataset_Test.loc[:, dataset_Test.columns != 'pbindaff'], dataset_Test.loc[:, 'pbindaff']
params_test = params_test.astype(int)
labels_test *= 1000
labels_test = labels_test.astype(int)
x_test = np.array(params_test.values)
y_test = np.array(labels_test.values.tolist())

print('Informação: ', x_test[0].tolist())
print('Resposta Correta: ', y_test[0])

print('Predição: ', clf.predict([x_test[0]]))
print('Score: ', clf.score(x_test, y_test))
