import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import platform

warnings.filterwarnings("ignore")
pathWindows = 'C:/Users/Pc/PycharmProjects/TCC/dataset/'
pathLinux = '/home/gianluca/Documentos/Projetos/Pycharm Projects/TCC_IA/dataset/'
path = pathLinux if platform.system() == 'Linux' else pathWindows

# read csv (comma separated value) into data
data = pd.read_csv(path + 'pdbbind-2007-refined-core-yx36i.csv')
plt.style.use('ggplot')
sns.countplot(x="PDB", data=data)
data.loc[:, 'PDB'].value_counts()

# KNN
from sklearn.neighbors import KNeighborsClassifier

# K é um hiperparametro
# K > n = overfiting, modelo complexo
# K < n = underfiting, modelo simples
knn = KNeighborsClassifier(n_neighbors=3)
# data = data.iloc[0:310]
x, y = data.loc[:, data.columns != 'PDB'], data.loc[:, 'PDB']
knn.fit(x, y)
prediction = knn.predict(x)
print('Prediction: {}'.format(prediction))
print(knn.score(x, y))

# Printando Gráficos e Valores dos DataFrames
df = pd.DataFrame(prediction)
df.index.name = 'ID'
df.columns = ['PDB']
df.to_csv('results.csv', header=True)
df2 = pd.DataFrame(df['PDB'].value_counts())
df2.index.name = 'PDB'
df2.columns = ['COUNT']
bigdata = pd.merge(df, df2, on="PDB")

sns.lmplot(x='PDB', y='COUNT', data=bigdata)
