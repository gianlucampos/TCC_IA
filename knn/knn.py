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

# read csv (comma separated value) into data
data = pd.read_csv(path + 'pdbbind-2007-refined-core-yx36i.csv')
# le = LabelEncoder()
# le.fit(data['PDB'].astype(str))
# data['PDB'] = le.transform(data['PDB'].astype(str))
plt.style.use('ggplot')
# sns.countplot(x="PDB", data=data)
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

# Printando Predição
predicao = pd.DataFrame(prediction)
predicao.index.name = 'ID'
predicao.columns = ['PDB']
print(predicao)
# df.to_csv('results.csv', header=True)

# Printando a frequência de PDB's
# n_ocorrencias = pd.DataFrame(predicao['PDB'].value_counts())
# n_ocorrencias.index.name = 'PDB'
# n_ocorrencias.columns = ['COUNT']
# bigdata = pd.merge(predicao, n_ocorrencias, on="PDB")
# bigdata = bigdata.drop_duplicates(keep='first')
# bigdata['PDB'] = le.inverse_transform(bigdata['PDB'])
# print(bigdata)

# sns.countplot(x='PDB', data=bigdata)
# plt.show()

# train test split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
# print('Prediction: {}'.format(prediction))
print('With KNN (K=3) accuracy is: ', knn.score(x_test, y_test))  # accuracy
