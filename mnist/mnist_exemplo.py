# Reconhecimento de Dígitos via SVM utilizando vetores classificação
# Link:
# https://www.kaggle.com/archaeocharlie/a-beginner-s-approach-to-classification
import warnings
import platform
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm

warnings.filterwarnings("ignore")

# Caminho para datasets
pathWindows = 'C:/Users/Pc/PycharmProjects/TCC/dataset/'
pathLinux = '/home/gianluca/Documentos/Projetos/Pycharm Projects/TCC_IA/dataset/'
path = pathLinux if platform.system() == 'Linux' else pathWindows

labeled_images = pd.read_csv(path + 'train.csv')
images = labeled_images.iloc[0:5000, 1:]
labels = labeled_images.iloc[0:5000, :1]

print(images.head(), labels.head())
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)

i = 1
img = train_images.iloc[i].as_matrix()
img = img.reshape((28, 28))
plt.imshow(img, cmap='gray')
plt.title(train_labels.iloc[i, 0])
plt.hist(train_images.iloc[i])

clf = svm.SVC()
clf.fit(train_images, train_labels)
score = clf.score(test_images, test_labels)
print('Score com treino: ', score)

test_images[test_images > 0] = 1
train_images[train_images > 0] = 1

img = train_images.iloc[i].as_matrix().reshape((28, 28))
plt.imshow(img, cmap='binary')
plt.title(train_labels.iloc[i])
plt.hist(train_images.iloc[i])
plt.show()

clf = svm.SVC()
clf.fit(train_images, train_labels)
score = clf.score(test_images, test_labels)

print('Score Com Teste: ', score)

test_data = pd.read_csv(path + 'test.csv')
test_data[test_data > 0] = 1
results = clf.predict(test_data[0:5000])
# df = pd.DataFrame(results)
# df.index.name = 'ImageId'
# df.index += 1
# df.columns = ['Label']
# df.to_csv('results.csv', header=True)
