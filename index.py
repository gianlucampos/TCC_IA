import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import platform
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
pathWindows = 'C:/Users/Pc/PycharmProjects/TCC/dataset/'
pathLinux = '/home/gianluca/Documentos/Projetos/Pycharm Projects/TCC_IA/dataset/'
path = pathLinux if platform.system() == 'Linux' else pathWindows

data = pd.read_csv(path + 'pdbbind-2007-refined-core-yx36i.csv')
le = LabelEncoder()
le.fit(data['PDB'].astype(str))
data['PDB'] = le.transform(data['PDB'].astype(str))
print(data.head())
data['PDB'] = le.inverse_transform(data['PDB'])
print(data.head())

