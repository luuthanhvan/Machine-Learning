import numpy as np
import matplotlib.pyplot as plt

def initData():
    X = np.array([1, 2, 4])
    y = np.array([2, 3, 6])
    return X, y

def linearRegression1(X, y, eta, loop, theta0, theta1):
    m = len(X) # number of instances
    for j in range(0, loop):
        #print("Lần lặp: ", j)
        for i in range(0, m):
            h_i = theta0 + theta1*X[i]
            theta0 = theta0 + eta*(y[i] - h_i)*1
            theta1 = theta1 + eta*(y[i] - h_i)*X[i]

            #print("Phần tử ", i, ", X = ", X[i],  ", y = ", y[i], "h = ", h_i, ": ")
            #print("\ttheta0 = ", round(theta0, 3), ", theta1 = ", round(theta1, 3))
    return [round(theta0, 3), round(theta1, 3)]

def main():
    X, y = initData()

    # training speed (eta) = 0.2, loop = 1, theta0 = 0, theta1 = 1
    theta1 = linearRegression1(X, y, 0.2, 1, 0, 1)
    #print("theta1: ", theta1) # theta1:  [0.336, 1.584]
    X1 = np.array([1, 6])
    y1 = theta1[0] + theta1[1]*X1

    # training speed (eta) = 0.2, loop = 2, theta0 = 0, theta1 = 1
    theta2 = linearRegression1(X, y, 0.2, 2, 0, 1)
    #print("theta2: ", theta2) # theta2:  [0.29, 1.572]
    X2 = np.array([1, 6])
    y2 = theta2[0] + theta2[1]*X2

    # training speed (eta) = 0.1, loop = 1, theta0 = 0, theta1 = 1
    theta3 = linearRegression1(X, y, 0.1, 1, 0, 1)
    #print("theta3: ", theta3) # theta3:  [0.257, 1.588]
    X3 = np.array([1, 6])
    y3 = theta3[0] + theta3[1]*X3

    # training speed (eta) = 0.1, loop = 2, theta0 = 0, theta1 = 1
    theta4 = linearRegression1(X, y, 0.1, 2, 0, 1)
    #print("theta4: ", theta4) # theta4:  [0.199, 1.406]
    X4 = np.array([1, 6])
    y4 = theta4[0] + theta4[1]*X4

    plt.axis([0, 5, 0, 8])
    plt.plot(X, y, "ro", color="blue") # show data

    # plt.plot(X1, y1, color="red") # regression line 1
    # plt.plot(X2, y2, color="black") # regression line 2
    plt.plot(X3, y3, color="pink") # regression line 3
    plt.plot(X4, y4, color="green") # regression line 4

    plt.xlabel("X")
    plt.ylabel("y")
    plt.show()

    # predict
    X_pred = np.array([0, 3, 5])
    for i in range(0, 3):
        y_pred = theta2[0] + theta2[1]*X_pred[i]
        print("X =", X_pred[i], "=> y =", round(y_pred, 3))

# calling main function
if __name__=="__main__":
    main()