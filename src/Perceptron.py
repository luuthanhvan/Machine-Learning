import numpy as np
import matplotlib.pyplot as plt

def initData():
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])
    Y = np.array([0, 0, 0, 1])
    return X, Y

def showData(X, Y):
    colorMap = np.array(["red", "green"])
    plt.axis([0, 1.5, 0, 2])
    # print(X[0])
    plt.scatter(X[0], X[1], c=colorMap[Y], s=150)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()

def myPerceptron(X, y, eta, loop):
    n = len(X[0])
    m = len(X[:,0])
    # print("m =", m, "n =", n)
    w0 = -0.2
    # w1 = 0.5
    # w2 = 0.5
    w = (0.5, 0.5)
    # print("w0 =", w0)
    # print("w1 =", w1)
    # print("w2 =", w2)
    for t in range(0, loop):
        print("Loop", t+1)
        for i in range(0, m):
            gx = w0 + sum(X[i,]*w)
            print("gx =", gx)

            if(gx > 0):
                output = 1
            else:
                output = 0
            
            w0 = w0 + eta * (y[i] - output)
            w = w + eta * (y[i] - output)*X[i,]
            print(" w0 =", w0)
            print(" w =", w)
    return (np.round(w0, 3), np.round(w, 3))

def main():
    X, Y = initData()
    # print(X)
    # print(Y)
    # showData(X, Y)
    w = myPerceptron(X, Y, 0.15, 2)
    print(w)
    

if __name__=="__main__":
    main()