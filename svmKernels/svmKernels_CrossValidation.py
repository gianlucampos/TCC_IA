import platform
import timeit
import warnings
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn import metrics
from matplotlib import pyplot as plt
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

kn = 0
for i in range(10):
    kn = kn + 10
    # Aplicando K-Fold
    kf = KFold(n_splits=kn, random_state=True, shuffle=True)
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

    # =========================================================================================
    # Treinamento
    print('=' * 240)
    print('TREINAMENTO, com Cross Validation')
    print('Kn: ', kn)
    # Definindo modelo SVM com regressão e parâmetros do kernel rbf
    clf = svm.SVR(kernel='rbf', gamma=0.00001, C=100, epsilon=0.001)

    # Treinando SVM
    clf.fit(x_train, y_train)
    print('Tamanho de amostras pra treino: ', len(x_train))

    # Predizendo e avaliando resultados (TREINO)
    # print('Resposta Correta: ', y_train)
    # print('Predição: ', clf.predict(x_train))
    inicio = timeit.default_timer()
    print('Score: ', clf.score(x_train, y_train))
    fim = timeit.default_timer()
    print('Tempo: %f segundos' % (fim - inicio))
    y_pred = clf.predict(x_train)
    mse = metrics.mean_squared_error(y_train, y_pred, multioutput='raw_values')
    print('RMSE', sqrt(mse))

    # Plotando os Resultados
    plt.subplot(1, 2, 1)
    plt.scatter(y_train, y_train, label="Afinidade Correta")
    plt.scatter(clf.predict(x_train), y_train, label="Afinidade Prevista")
    plt.title('Treino')
    plt.legend()
    # =========================================================================================
    # Teste
    print('-' * 240)
    print('TESTE, com Cross Validation')

    # Predizendo e avaliando resultados (TESTE)
    # print('Resposta Correta: ', y_test)
    # print('Predição: ', clf.predict(x_test))
    inicio = timeit.default_timer()
    print('Score: ', clf.score(x_test, y_test))
    fim = timeit.default_timer()
    print('Tempo: %f segundos' % (fim - inicio))
    y_pred = clf.predict(x_test)
    mse = metrics.mean_squared_error(y_test, y_pred, multioutput='raw_values')
    print('RMSE', sqrt(mse))

    # Plotando os Resultados
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_test, label="Afinidade Correta")
    plt.scatter(clf.predict(x_test), y_test, label="Afinidade Prevista")
    plt.title('Teste')
    plt.legend()
    # plt.show()
