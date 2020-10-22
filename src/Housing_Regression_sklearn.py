import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# import dataset from housing_RT.csv
def importData():
    house_dt = pd.read_csv("../data_set/housing_RT.csv", delimiter=",", index_col=0)
    return house_dt

# split dataset: data training and data testing (hold-out method)
def splitDataset(house_dt):
    X = house_dt.iloc[:, 1:5]
    Y = house_dt.iloc[:, 0]
    # print(X)
    # print(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=1/3.0, random_state=100)
    return X, Y, X_train, X_test, y_train, y_test
    
# Decision Tree model
def train(X_train, y_train):
    # creating the regressor object
    regressor = DecisionTreeRegressor(max_depth=3, random_state=100)
    # performing training
    regressor.fit(X_train, y_train)
    return regressor

# Prediction
def prediction(X_test, regressor):
    y_pred = regressor.predict(X_test)
    #y_pred = regressor.predict([[2000, 3, 2, 2]])
    #print(y_pred)
    return y_pred

# Calculating accuracy
def cal_accuracy(y_test, y_pred):
    # Calculating accuracy using MAE, MSE and RMSE
    print("Mean Absolute Error: ", mean_absolute_error(y_test, y_pred));
    print("Mean Squared Error: ", mean_squared_error(y_test, y_pred));
    print("Root Mean Squared Error: ", np.sqrt(mean_squared_error(y_test, y_pred)));

def main():
    # 1. Building phrase
    data = importData()
    X, Y, X_train, X_test, y_train, y_test = splitDataset(data) # Split the dataset from train and test using Python sklearn package.
    regressor = train(X_train, y_train) # Train the regressor

    # 2. Opeational phrase
    # Prediction
    y_pred = prediction(X_test, regressor)
    #print(X_test)
    #print(y_test)
    #print(y_pred)
    # Calculate the accuracy
    cal_accuracy(y_test, y_pred)

# calling main function
if __name__=="__main__":
    main()