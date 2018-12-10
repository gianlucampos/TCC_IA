import warnings
import numpy as np
from sklearn.model_selection import KFold,cross_val_score,cross_val_predict
from sklearn import metrics

warnings.filterwarnings("ignore")

X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [8, 9]])
y = np.array([1, 2, 3, 4, 5])

kf = KFold(n_splits=2, random_state=True, shuffle=True)

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

print("Train X-Y")
print(X_train, y_train)
print("Test X-Y")
print(X_test, y_test)

# scores = cross_val_score(model, df, y, cv=6)
# print ("Cross-validated scores: ", scores)
#
# predictions = cross_val_predict(model, df, y, cv=6)
# plt.scatter(y, predictions)
# accuracy = metrics.r2_score(y, predictions)
# print “Cross-Predicted Accuracy:”, accuracy
#
