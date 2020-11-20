import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

def readFile():
    data = pd.read_csv("../data_set/Housing_2019.csv", delimiter=",", index_col=0)
    return data

def splitData(data):
    X = data.iloc[:, [1, 3, 4, 10]]
    y = data.price
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3.0, random_state=13)
    return X, y, X_train, X_test, y_train, y_test

def train(X_train, y_train):
    lm = linear_model.LinearRegression()
    lm.fit(X_train, y_train)
    return lm

def predict(lm, X_test):
    y_pred = lm.predict(X_test)
    return y_pred

def main():
    data = readFile()
    X, y, X_train, X_test, y_train, y_test = splitData(data)
    linearModel = train(X_train, y_train)
    y_pred = predict(linearModel, X_test)
    # print(y_test)
    # print(y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print("MSE: ", round(mse, 3))
    print("RMSE: ", round(rmse, 3))


# calling main function
if __name__=="__main__":
    main()