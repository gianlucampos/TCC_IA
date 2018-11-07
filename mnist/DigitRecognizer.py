import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm

labeled_images = pd.read_csv('dataset/train.csv')
images = labeled_images.iloc[0:5000, 1:]
labels = labeled_images.iloc[0:5000, :1]
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)

labeled_images.info()
print(labeled_images)
train_labels.info()
print(train_labels)

i = 0
img = train_images.iloc[i].values
img = img.reshape((28, 28))
plt.imshow(img, cmap='gray')
plt.title(train_labels.iloc[i, 0])
plt.show()
# plt.hist(train_images.iloc[i])
# plt.show()

# clf = svm.SVC()
# clf.fit(train_images, train_labels.values.ravel())
# clf.score(test_images, test_labels)
#
# test_images[test_images > 0] = 1
# train_images[train_images > 0] = 1
# img = train_images.iloc[i].as_matrix().reshape((28, 28))
# plt.imshow(img, cmap='binary')
# plt.title(train_labels.iloc[i])
# plt.show()
#
# clf = svm.SVC()
# clf.fit(train_images, train_labels.values.ravel())
# clf.score(test_images,test_labels)
