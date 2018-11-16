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

le = LabelEncoder()
le.fit(dataTrain['PDB'].astype(str))
dataTrain['PDB'] = le.transform(dataTrain['PDB'].astype(str))
print(dataTrain.head())
dataTrain['PDB'] = le.inverse_transform(dataTrain['PDB'])
print(dataTrain.head())

mylist = []
mylist.append(12)
mylist.append(12)
print(mylist)