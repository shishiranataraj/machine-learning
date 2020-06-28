from sklearn import datasets
import numpy as np

iris = datasets.load_iris()

#split in features and labels
X = iris.data
y = iris.target
print(X.shape)
print(y.shape)

#divide training and testing
from sklearn.model_selection import train_test_split
X_Train, X_Test, y_train, y_test = train_test_split(X, y, test_size=0.2)
