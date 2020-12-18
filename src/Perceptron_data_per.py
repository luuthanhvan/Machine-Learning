import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def readFile():
    data = pd.read_csv("../data_set/data_per.csv")
    return data

def myPerceptron(X, y, eta, loop):
    n = len(X.iloc[0,:]) # number of columns (attributes)
    m = len(X) # number of rows (instances)
    # print("m =", m, "n =", n)

    np.random.seed(0)
    rd = np.random.rand(5)
    w0 = rd[0]
    w = rd[1:len(rd)]
    
    for t in range(0, loop):
        print("Loop", t+1)
        for i in range(0, m):
            
            gx = w0 + sum(X.values[i,0:4]*w)

            # print("gx =", gx)

            if(gx > 0):
                output = 1
            else:
                output = 0
            
            w0 = w0 + eta * (y.values[i] - output)
            w = w + eta * (y.values[i] - output)*X.values[i,0:4]
            # print(" w0 =", w0)
            # print(" w =", w)

    return (np.round(w0, 3), np.round(w, 3))

def main():
    data = readFile()
    # print(data)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # print(X.values)
    w = myPerceptron(X, y, 0.1, 5)
    # print(w)
    # print(len(X))
    # print(y)
    # np.random.seed(0)
    # wtest = np.random.rand(5)
    # w0 = wtest[0]
    # w = wtest[1:len(wtest)]
    # print(w0)
    # print(w)
    

if __name__=="__main__":
    main()