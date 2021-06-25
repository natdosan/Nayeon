"""
  Linear regression visualization of user data
  Copyright (c) 2021, Nate d. (sjkuksee)
"""
import numpy as np
import pandas as pd
import tensorflow
import keras
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import csv
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style
data = pd.read_csv("restaurant.csv", sep=",")
print(data.head())

# reassign to get attributes that you want here
data = data[["Items ordered", "Order total", "Age", "Returning (0 = f, 1 = t)"]]

length = len(data)
print(data.head())

# what you want to predict
predict = "Order total"

# return new data frame without our prediction value as training data
X = np.array(data.drop([predict], 1))
y = np.array(data[predict])
# x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

# find the best model
best = 0
for i in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    accuracy = linear.score(x_test, y_test)
    print(accuracy)

    if accuracy > best:
        best = accuracy
    with open("restaurant_model.pickle", "wb") as f:
        pickle.dump(linear, f)

# saving model
pickle_in = open("restaurant_model.pickle", "rb")
linear = pickle.load(pickle_in)

print("Coefficients of ", length, " dimensions:\n", linear.coef_)
print("Intercept: \n", linear.intercept_)

# predictions
predictions = linear.predict(x_test)

for i in range(len(predictions)):
    print(predictions[i], x_test[i], y_test[i])

# visualizing results
# set your own independent variable
correlationVariable = 'Age'
style.use("ggplot")
pyplot.scatter(data[correlationVariable], data["Order total"])
pyplot.xlabel(correlationVariable)
pyplot.ylabel("Order total ($)")
pyplot.show()
