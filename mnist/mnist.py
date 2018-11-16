# 1 etapa: pegar base de treino do rf-score
# 2 etapa: criar amostras de treino e teste
# 3 etapa: utilizar K-fold para cross-validation

import warnings

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import svm, linear_model
import platform

warnings.filterwarnings("ignore")

pathWindows = 'C:/Users/Pc/PycharmProjects/TCC/dataset/'
pathLinux = '/home/gianluca/Documentos/Projetos/Pycharm Projects/TCC_IA/dataset/'
path = pathLinux if platform.system() == 'Linux' else pathWindows

# Parâmetros

# pbindaff,
# 6.6, 7.6, 8.6, 16.6,
# 6.7, 7.7, 8.7, 16.7,
# 6.8, 7.8, 8.8, 16.8,
# 6.9, 7.9, 8.9, 16.9,
# 6.15,7.15,8.15,16.15,
# 6.16,7.16,8.16,16.16,
# 6.17,7.17,8.17,16.17,
# 6.35,7.35,8.35,16.35,
# 6.53,7.53,8.53,16.53,

# Label
# PDB

dataset = pd.read_csv(path + 'pdbbind-2007-refined-core-yx36i.csv')
le = LabelEncoder()
le.fit(dataset['PDB'].astype(str))
dataset['PDB'] = le.transform(dataset['PDB'].astype(str))

params = dataset.iloc[0:1105, :37]
labels = dataset.iloc[0:1105, 37:]

print('X Data')
print('==================================================')
print(params.head())
print('==================================================')
print('Y Data')
print(labels.head())
print('==================================================')

X_train, X_test, Y_train, Y_test = train_test_split(params, labels, train_size=0.2, random_state=0)

lm = linear_model.LinearRegression()
model = lm.fit(X_train, Y_train)
predictions = lm.predict(X_test)

print(predictions[0:5])
plt.scatter(Y_test, predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()
print("Score Before FIT:", model.score(X_test, Y_test))

clf = svm.SVC()
clf.fit(X_train, Y_train)
score = clf.score(X_test, Y_test)
print('Final Score: ')
print(score)
# ==============================================================================
# TestePDB:

test_data = pd.read_csv(path + 'pdbbind-2007-core-yx36i.csv')
le.fit(test_data['PDB'].astype(str))
test_data['PDB'] = le.transform(test_data['PDB'].astype(str))

results = clf.predict(X_test)
print(results)
df = pd.DataFrame(results)
