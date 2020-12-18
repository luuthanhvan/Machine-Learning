import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

def readFile():
    data = pd.read_csv("../data_set/iris_data.csv")
    return data

# split dataset: data training and data testing (hold-out method)
def splitDataset(dataset):
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
    
    return X, y, X_train, X_test, y_train, y_test
    
def trainPerceptron(X_train, y_train, eta, loop):
    net = Perceptron(max_iter=loop, eta0=eta, random_state=200)
    net.fit(X_train, y_train)
    return net

def prediction(X_test, net):
    y_pred = net.predict(X_test)
    return y_pred

def calAccuracy(y_test, y_pred):
    print("Accuracy is ", accuracy_score(y_test, y_pred)*100)

def main():    
    # read data from file
    data = readFile()
    # print(data)
    # split dataset (hold-out method)
    X, y, X_train, X_test, y_train, y_test = splitDataset(data)

    eta = [0.002, 0.02, 0.1, 0.2]
    loop = [5, 50, 100, 1000]

    for i in range(len(eta)):
        # training
        net = trainPerceptron(X_train, y_train, eta[i], loop[i])
        # prediction
        y_pred = prediction(X_test, net)
        # calculating accuracy
        calAccuracy(y_test, y_pred)


if __name__=="__main__":
    main()