import numpy as np

def initData():
    X = np.array([1, 2, 4])
    y = np.array([2, 3, 6])
    return X, y

def linearRegression2(X, y, eta, loop, theta0, theta1):
    m = len(X) # number of instances
    X_j = [np.array([1, 1, 1]), X]

    for j in range(0, loop):
        sum1 = 0
        sum2 = 0

        for i in range(0, m):
            h_i = theta0 + theta1*X[i]
            sum1 = sum1 + (y[i] - h_i)*1
            sum2 = sum2 + (y[i] - h_i)*X[i]

        theta0 = theta0 + eta*sum1
        theta1 = theta1 + eta*sum2
    
    return round(theta0, 3), round(theta1, 3)

'''
    theta0 = 0 + 0.2 * ( (2 - (0*1 + 1*1))*1 + (3 - (0*1 + 1*2))*1 + (6 - (0*1 + 1*4))*1 )
    theta1 = 1 + 0.2 * ( (2 - (0*1 + 1*1))*1 + (3 - (0*1 + 1*2))*2 + (6 - (0*1 + 1*4))*4 )
'''

def main():
    X, y = initData()
    # training speed (eta) = 0.2, loop = 2, theta0 = 0, theta1 = 1
    theta0, theta1 = linearRegression2(X, y, 0.2, 2, 0, 1)
    print("theta0 = ", theta0)
    print("theta1 = ", theta1)

    # predict
    X_pred = np.array([0, 3, 5])
    for i in range(0, 3):
        y_pred = theta0 + theta1*X_pred[i]
        print("X =", X_pred[i], "=> y =", round(y_pred, 3))

# calling main function
if __name__=="__main__":
    main()