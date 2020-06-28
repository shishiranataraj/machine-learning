#Getting data
import pandas as pd

data = pd.read_csv('car.data')

#classifying to features and labels
X = data[[
    'buying',
    'maint',
    'safety'
]].values

y = data[['class']]

#convert the features/labels into numbers
#this process is called pre_processing

#preprocessing X
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for i in range(len(X[0])):
    X[:, i] = le.fit_transform(X[:, i])


#preprocessing y
label_mapping = {
    'unacc': 0,
    'acc': 1,
    'good': 2,
    'vgood': 3
}
y['class'] = y['class'].map(label_mapping)

#converting to numpy arrays
import numpy as np

y = np.array(y)

print(y)

#create a model
from sklearn import neighbors

knn = neighbors.KNeighborsClassifier(n_neighbors=25, weights='uniform') #n_neightbors implies to k value

#split the dataset into training and testing
from sklearn.model_selection import train_test_split

X_Train, X_Test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#train model
clf = knn.fit(X_Train, y_train)

#check accuracy
from sklearn import metrics

prediction = knn.predict(X_Test)
accuracy = metrics.accuracy_score(y_test, prediction)

#test
# a = 1272
# print("Actual_Value: " , y[a])
# print("Predicted_Value: ", knn.predict(X)[a])