# Reconhecimento de Dígitos via SVM utilizando vetores classificação
# Link: https://www.kaggle.com/archaeocharlie/a-beginner-s-approach-to-classification

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm

# Carregando dados
labeled_images = pd.read_csv('dataset/digits.csv')
images = labeled_images.iloc[0:5000, 1:]
labels = labeled_images.iloc[0:5000, :1]
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)

# Plotando imagem e histograma
i = 5
img = train_images.iloc[i].values
img = img.reshape((28, 28))
plt.imshow(img, cmap='gray')
plt.title(train_labels.iloc[i, 0])
plt.show()
plt.hist(train_images.iloc[i])

# Medindo desempenho (Apenas 10%)
clf = svm.SVC()
forma = clf.fit(train_images, train_labels.values.ravel())
pontuacao = clf.score(test_images, test_labels)

# Escala binária (preto de branco)
test_images[test_images > 0] = 1
train_images[train_images > 0] = 1
img = train_images.iloc[i].values.reshape((28, 28))
plt.imshow(img, cmap='binary')
plt.title(train_labels.iloc[i])
# plt.show()
# plt.hist(train_images.iloc[i])
