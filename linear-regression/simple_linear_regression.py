import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#generate dataset
dataset = pd.read_csv('simple_linear_regression.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#split it into train and test
from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(x, y, test_size = 1/3, random_state = 0)

#train model
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_Train, Y_Train)

#predict
y_pred = regression.predict(X_Test)


#visualizing
plt.scatter(X_Train, Y_Train, color="red")
plt.plot(X_Train, regression.predict(X_Train), color="blue")

plt.scatter(X_Test, regression.predict(X_Test), color="yellow")

plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()  