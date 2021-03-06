import platform
import timeit
import warnings
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn import metrics
from matplotlib import style
from math import sqrt

style.use("ggplot")
warnings.filterwarnings("ignore")

# Setando timer para avaliar tempo de treino e teste
inicio = timeit.default_timer()

# Caminho para datasets
pathWindows = 'C:/Users/Pc/PycharmProjects/TCC/dataset/'
pathLinux = '/home/gianluca/Documentos/Projetos/Pycharm Projects/TCC_IA/dataset/'
path = pathLinux if platform.system() == 'Linux' else pathWindows

# Convertendo dataset para dataframe
dataset = pd.read_csv(path + 'pdbbind-2007-refined-full.csv', index_col=0)

# Convertendo Strings
le = LabelEncoder()
le.fit(dataset['PDB'])
dataset['PDB'] = le.transform(dataset['PDB'])

# Definindo valores de parâmetros de entrada (Distância euclidiana e nome do composto) e Labels (Afinidade)
params, labels = dataset.loc[:, dataset.columns != 'pbindaff'], dataset.loc[:, 'pbindaff']

# Criando Arranjos desses valores para jogar no SVM (Exemplo: X[0,1,2,3] ∈ Y[0])
x = np.array(params.values)
y = np.array(labels.values.tolist())

# Definindo modelo SVM com regressão e parâmetros do kernel rbf
clf = svm.SVR(kernel='rbf', gamma=0.00001, C=100, epsilon=0.001)

# Para
# Múltiplos de 1300: 1, 2, 4, 5, 10, 13, 20, 25, 26, 50, 52, 65, 100, 130, 260, 325, 650, 1300
kn_values = [2, 4, 5, 10, 13, 20, 25, 26, 50, 52, 65, 100, 130, 260, 325, 650, 1300]
for i in range(17):
    kn = kn_values[i]
    print('K = ', kn)
    # Aplicando K-Fold
    kf = KFold(n_splits=kn, random_state=True, shuffle=True)
    # Avaliando tempo de treino e teste
    inicio = timeit.default_timer()
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Treinando SVM
        clf.fit(x_train, y_train)

    y_pred = clf.predict(x_train)
    mse = metrics.mean_squared_error(y_train, y_pred, multioutput='uniform_average')
    print('RMSE Treino: ', sqrt(mse))

    # Teste
    # Predizendo e avaliando resultados da primeira amostra
    y_pred = clf.predict(x_test)
    mse = metrics.mean_squared_error(y_test, y_pred, multioutput='uniform_average')
    print('RMSE Teste: ', sqrt(mse))
    fim = timeit.default_timer()
    print('Tempo: %f segundos' % (fim - inicio))
    print(clf.score(x_test, y_test))
    print('-' * 240)
