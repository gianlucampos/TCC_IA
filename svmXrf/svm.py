import platform
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics

pathWindows = 'C:/Users/Pc/PycharmProjects/TCC/dataset/'
pathLinux = '/home/gianluca/Documentos/Projetos/Pycharm Projects/TCC_IA/dataset/'
path = pathLinux if platform.system() == 'Linux' else pathWindows
dftrain = pd.read_csv(path + 'pdbbind-2007-refined-core-yx36i.csv')
dftest = pd.read_csv(path + 'pdbbind-2007-core-yx36i.csv')

train = pd.concat([dftrain, pd.get_dummies(dftrain['PDB'], prefix='PDB')], axis=1)
params = train.iloc[0:1105, :37]
labels = train.iloc[0:1105, 38:]
X_train, X_test, Y_train, Y_test = train_test_split(params, labels, train_size=0.2, random_state=0)

print('')
print('Training SVM...')
svmmodel = svm.SVC(degree=3, gamma='auto', kernel='rbf')
svmmodel.fit(X_train, X_test)

# predict on normalized test data
print('')
print('Predicting using svm...')
svmoutput = svmmodel.predict(train)
