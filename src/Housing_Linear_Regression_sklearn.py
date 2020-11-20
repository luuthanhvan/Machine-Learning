import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

def readFile():
    data = pd.read_csv("../data_set/Housing_2019.csv", delimiter=",", index_col=0)
    return data

def main():
    data = readFile()
    #print(data)
    X = data.iloc[:, [1, 2, 3, 4, 10]]
    y = data.price
    print(X)
    #print(y)
    plt.scatter(data.lotsize, data.price)
    plt.xlabel("lotsize")
    plt.ylabel("price")
    #plt.show()

    lm = linear_model.LinearRegression()
    lm.fit(X[0:520], y[0:520]) 
    #print(lm.intercept_) # theta0
    #print(lm.coef_) # the rest of theta

    # predict the last 20 instances
    y_test = y[-20:]
    X_test = X[-20:]
    y_pred = lm.predict(X_test)

    #print(y_test)
    #print(y_pred)
    err = mean_squared_error(y_test, y_pred)
    rmse_err = np.sqrt(err)
    print(err)
    print(round(rmse_err, 3))

# calling main function
if __name__=="__main__":
    main()