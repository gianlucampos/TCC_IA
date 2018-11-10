import warnings
import platform
import pandas as pd
import numpy as np
from sklearn import svm
from matplotlib import style
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")

style.use("ggplot")
pathWindows = 'C:/Users/Pc/PycharmProjects/TCC/dataset/'
pathLinux = '/home/gianluca/Documentos/Projetos/Pycharm Projects/TCC_IA/dataset/'
path = pathLinux if platform.system() == 'Linux' else pathWindows
dataset = pd.read_csv(path + 'pdbbind-2007-refined-core-yx36i.csv')
# =========================================================================================
# Código para personalizar:
params = dataset.iloc[0:1105, :37]
labels = dataset.iloc[0:1105, 37:]
x = np.array(params.values)
y = np.array(labels.values.tolist())

print(x[3])
print(y[3])

clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(x, y)
print('Predição: ', clf.predict([x[3]]))
print('Score', clf.score(x, y))

# O certo seria balancear com train_test_split e k-fold validation
# print('Score', clf.score(X_test, Y_test))


