#Ewan M., Nina G., Megan W., Jane K.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# imports the data and sets x and y values
data = pd.read_csv("dairy_dataset.csv")
x = data[["Total Land Area (acres)", "Number of Cows"]].values
# print(x)
y = data["Quantity (liters/kg)"].values

# separates the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = .2)

# # reshape the xtrain data into a 2D array
# xtrain = xtrain.reshape(-1, 1)

# create the linear regression model using the training data
model = LinearRegression().fit(xtrain, ytrain)


# get the coef_, intercept_ valuesm and r^2 values
# use float() to turn the arrays into a single float value
coef = np.around(model.coef_, 2)
intercept = round(float(model.intercept_), 2)
r_squared = round(model.score(x, y),2)

# print out the linear equation and r^2 value
print(f"Model's Linear Equation: y={coef[0]}x1 + {coef[1]}x2  + {intercept}")
print("R Squared value:", r_squared)

# get the predicted y values for the xtest values - returns an array of the results
predict = model.predict(xtest)
# round the value in the np array to 2 decimal places
predict = np.around(predict, 2)
print(predict)

# compare the actual and predicted values
print("\nTesting Multivariable Model with Testing Data:")
for index in range(len(xtest)):
    actual = round(ytest[index]) # gets the actual y value from the ytest dataset
    predicted_y = predict[index] # gets the predicted y value from the predict variable
    x_coord = xtest[index] # gets the x value from the xtest dataset
    x_coord=np.around(x_coord,2)
    print(f"Total Land Area (acres)n: {x_coord[0]} Number of Cows: {x_coord[1]}  Actual: {actual} Predicted: {predicted_y}")
