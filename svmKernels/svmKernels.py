import platform
import timeit
import warnings
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from matplotlib import style
from math import sqrt

style.use("ggplot")

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

# Definindo valores de parâmetros de entrada (Distância euclidiana e nome do composto) e Labels (Afinidade)
params, labels = dataset_Train.loc[:, dataset_Train.columns != 'pbindaff'], dataset_Train.loc[:, 'pbindaff']

# Multiplicando valores por 1000 e convertendo pra inteiros (SVC obrigatorio, SVR não precisa)
# params = params.astype(int)
# labels *= 1000
# labels = labels.astype(int)

# Criando Arranjos desses valores para jogar no SVM (Exemplo: X[0,1,2,3] ∈ Y[0])
x = np.array(params.values)
y = np.array(labels.values.tolist())

# print('Informação: ', x[0].tolist())
print('Resposta Correta: ', y)

# Criando Classificador SVM (para testar comente descomentar um)
# clf = svm.SVC(kernel='linear')
# clf = svm.SVC(kernel='poly')
# clf = svm.SVC(kernel='rbf')
# clf = svm.SVC(kernel='sigmoid')

# clf = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False,
#               tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1,
#               decision_function_shape='ovr', random_state=0)

# Regressor
# clf = svm.SVR(kernel='linear')
# clf = svm.SVR(kernel='poly')
# clf = svm.SVR(kernel='rbf')
# clf = svm.SVR(kernel='sigmoid')


clf = svm.SVR(kernel='rbf', gamma=0.00001, C=5.1, epsilon=0.001)

# Treinando SVM
clf.fit(x, y)

# Predizendo e Medindo resultado
print('Predição: ', clf.predict(x))
print('Score: ', clf.score(x, y))
fim = timeit.default_timer()
print('Tempo: %f segundos' % (fim - inicio))

y_pred = clf.predict(x)
mse = mean_squared_error(y, y_pred, multioutput='raw_values')
print('RMSE', sqrt(mse))

# Plotando os Resultados
plt.subplot(1, 2, 1)
plt.scatter(y, y, label="Afinidade Correta")
plt.scatter(y, clf.predict(x), label="Afinidade Prevista")
plt.title('Treino')
plt.legend()
# plt.show()
# =========================================================================================
# Balanceamento de base com train_test_split ou k-fold validation
# print('=' * 240)
# print('Validando Dataset de Treino')
# =========================================================================================
# Teste, segue o mesmo processo visto acima:
print('=' * 240)
print('TESTE')
params_test, labels_test = dataset_Test.loc[:, dataset_Test.columns != 'pbindaff'], dataset_Test.loc[:, 'pbindaff']
# params_test = params_test.astype(int)
# labels_test *= 1000
# labels_test = labels_test.astype(int)
x_test = np.array(params_test.values)
y_test = np.array(labels_test.values.tolist())

# print('Informação: ', x_test[0].tolist())
# print('Resposta Correta: ', y_test)
# print('Predição: ', clf.predict(x_test))
print('Score: ', clf.score(x_test, y_test))

y_pred = clf.predict(x_test)
mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
print('RMSE', sqrt(mse))

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test, label="Afinidade Correta")
plt.scatter(y_test, clf.predict(x_test), label="Afinidade Prevista")
plt.title('Teste')
plt.legend()
plt.show()
print('=' * 240)
