"""
    Data visualization and regression
    Copyright (c) 2021, sjuksee
    Author: [sjkuksee]
    Date Created: 210625
    Last updated: 210703
"""

import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
import csv
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("../datasets/formatted.csv", sep=",")

# reassign to get attributes that you want here
data = data[["Visits", "Items", "Total", "Age", "Solo?", "Duplicates?"]]

length = len(data)

# what you want to predict
predict = "Total"

# return new data frame without our prediction value as training data
# design matrix
X = np.array(data.drop([predict], 1))
# prediction vector
y = np.array(data[predict])

# find the best model
best = 0
accuracies = []

# set sample size
n = 40
for i in range(n):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    accuracy = linear.score(x_test, y_test)
    print(accuracy)
    accuracies.append(accuracy)

    if accuracy > best:
        best = accuracy
    with open("restaurant_model.pickle", "wb") as f:
        pickle.dump(linear, f)

average = sum(accuracies) / len(accuracies)

print('Best accuracy: ', best)
print('Average accuracy: ', average)

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
pyplot.scatter(data[correlationVariable], data["Total"])
# pyplot.hist(data[correlationVariable], histtype='bar', align='mid', orientation='vertical')

# Correlation coefficient, r
x = pd.Series(data[correlationVariable])
y1 = pd.Series(data["Visits"])
y2 = pd.Series(data["Total"])

r1 = x.corr(y1)
r2 = x.corr(y2)
print("r of Visits = ", r1, "r of Total $ Spent = ", r2)

"""
    3d visualization
    docu:
    https://matplotlib.org/stable/tutorials/toolkits/mplot3d.html#toolkit-mplot3d-tutorial
    https://matplotlib.org/stable/gallery/mplot3d/2dcollections3d.html#sphx-glr-gallery-mplot3d-2dcollections3d-py
    https://matplotlib.org/stable/gallery/index.html#mplot3d-examples-index
    https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
"""
ax1 = pyplot.figure().add_subplot(projection='3d')
ax1.scatter(data[correlationVariable], data["Total"], data["Visits"])
ax1.scatter(data[correlationVariable], data["Total"], zs=0, zdir='y', label='Visits with respect to Age')

# Make legend, set axes limits and labels
ax1.legend()
ax1.set_xlim(0, 6)
ax1.set_ylim(0, 5)
ax1.set_zlim(0, 5)
ax1.set_xlabel('Age')
ax1.set_ylabel('Total')
ax1.set_zlabel('Visits')

# Customize the view angle so it's easier to see that the scatter points lie
# on the plane y=0
ax1.view_init(elev=20., azim=-35)

'''
    3d bar charts
    docu: 
    https://matplotlib.org/stable/gallery/mplot3d/3d_bars.html#sphx-glr-gallery-mplot3d-3d-bars-py
'''

figure = pyplot.figure(figsize=(8, 3))
ax2 = figure.add_subplot(121, projection='3d')

# top = data[correlationVariable] + data["Total"]
top = data["Visits"]
bottom = np.zeros_like(top)
width = depth = 1

ax2.bar3d(data[correlationVariable], data["Total"], bottom, width, depth, top, shade=True)
ax2.set_title('Shaded')
ax2.set_zlabel('Visits')

pyplot.xlabel(correlationVariable)
pyplot.ylabel("Total ($)")
pyplot.show()

