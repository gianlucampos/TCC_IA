# Manter estes imports por padrão
import platform
import warnings
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Caminho para datasets
warnings.filterwarnings("ignore")
pathWindows = 'C:/Users/Pc/PycharmProjects/TCC/dataset/'
pathLinux = '/home/gianluca/Documentos/Projetos/Pycharm Projects/TCC_IA/dataset/'
path = pathLinux if platform.system() == 'Linux' else pathWindows
dataTrain = pd.read_csv(path + 'pdbbind-2007-refined-core-yx36i.csv')

# le = LabelEncoder()
# le.fit(dataTrain['PDB'].astype(str))
# dataTrain['PDB'] = le.transform(dataTrain['PDB'].astype(str))
# print(dataTrain.head())
# dataTrain['PDB'] = le.inverse_transform(dataTrain['PDB'])
# print(dataTrain.head())
#
# mylist = []
# mylist.append(12)
# mylist.append(12)
# print(mylist)
#
# dataset_Train = pd.read_csv(path + 'pdbbind-2007-refined-core-yx36i.csv')
# dataset_Test = pd.read_csv(path + 'pdbbind-2007-core-yx36i.csv')
# dataset = dataset_Train.append(dataset_Test)
# dataset = dataset.reset_index(drop=True)
# dataset.to_csv('pdbbind-2007-refined-full.csv', header=True)
# dataset2 = pd.read_csv('pdbbind-2007-refined-full.csv', index_col=0)
# dataset2.to_csv('pdbbind-2007-refidasdasned-full.csv', header=True)

# Salvando o modelo svm em um arquivo para testa-lo depois
from sklearn import svm
from sklearn import datasets

# clf = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target
# clf.fit(X, y)
# print(clf.score(X, y))

from joblib import dump, load

# dump(clf, 'filename.svm')  # Escreve modelo
clf3 = load('filename.Svm')  # Lê modelo
print(clf3.score(X, y))

