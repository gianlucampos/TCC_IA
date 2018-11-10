import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# read csv (comma separated value) into data
data = pd.read_csv('C:/Users/Pc/PycharmProjects/TCC/dataset/column_2C_weka.csv')
plt.style.use('ggplot')
sns.countplot(x="class", data=data)
data.loc[:, 'class'].value_counts()

# KNN
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
x, y = data.loc[:, data.columns != 'class'], data.loc[:, 'class']
knn.fit(x, y)
prediction = knn.predict(x)
# print('Prediction: {}'.format(prediction))
print('Without Test Split, Score: ', knn.score(x, y))

# train test split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
knn = KNeighborsClassifier(n_neighbors=3)
x, y = data.loc[:, data.columns != 'class'], data.loc[:, 'class']
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
# print('Prediction: {}'.format(prediction))
print('With KNN (K=3) accuracy is: ', knn.score(x_test, y_test))  # accuracy

# create data1 that includes pelvic_incidence that is feature and sacral_slope that is target variable
data1 = data[data['class'] == 'Abnormal']
x = np.array(data1.loc[:, 'pelvic_incidence']).reshape(-1, 1)
y = np.array(data1.loc[:, 'sacral_slope']).reshape(-1, 1)
# Scatter
plt.figure(figsize=[10, 10])
plt.scatter(x=x, y=y)
plt.xlabel('pelvic_incidence')
plt.ylabel('sacral_slope')

# LinearRegression
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
# Predict space
predict_space = np.linspace(min(x), max(x)).reshape(-1, 1)
# Fit
reg.fit(x, y)
# Predict
predicted = reg.predict(predict_space)
# R^2
print('R^2 score: ', reg.score(x, y))
# Plot regression line and scatter
plt.plot(predict_space, predicted, color='black', linewidth=3)
plt.scatter(x=x, y=y)
plt.xlabel('pelvic_incidence')
plt.ylabel('sacral_slope')
plt.show()
