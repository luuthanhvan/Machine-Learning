import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 

# get Iris file from sklearn
def loadData():
    iris_dt = load_iris()
    #print(iris_dt)
    #print("data original: ")
    #print(iris_dt.data[0:15])
    #iris_dt.target[1:5]
    return iris_dt

# split dataset: data training and data testing (hold-out method)
def splitDataset(iris_dt):
    X = iris_dt.data
    Y = iris_dt.target
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    #print("data training: ")
    #print(X_train)
    #print("data testing: ")
    #print(X_test)
    return X, Y, X_train, X_test, y_train, y_test
    
# Decision Tree model
def train_using_gaussian(X_train, y_train):
    model = GaussianNB()
    # performing training
    model.fit(X_train, y_train)
    return model

# Prediction
def prediction(X_test, model):
    y_pred = model.predict(X_test)
    #y_pred = clf_gini.predict([[8, 3, 7, 2]])
    #print(y_pred)
    return y_pred

# Calculating accuracy
def cal_accuracy(y_test, y_pred):
    # Calculating accuracy using confusion matrix
    print(confusion_matrix(y_test, y_pred, labels=[2,0,1])) # for iris data set in sklearn
    print("Accuracy is ", accuracy_score(y_test, y_pred)*100)

'''
While implementing the decision tree we will go through the following two phases:
    1. Building Phase
        Preprocess the dataset.
        Split the dataset from train and test using Python sklearn package.
        Train the classifier.
    2. Operational Phase
        Make predictions.
        Calculate the accuracy.
'''

def main():
    
    # 1. Building phrase
    irisData = loadData()
    # print(data)
    # X, Y, X_train, X_test, y_train, y_test = splitDataset(irisData) # Split the dataset from train and test using Python sklearn package.
    # model = train_using_gaussian(X_train, y_train) # Train the classifier using Gaussian

    # 2. Opeational phrase
    # Prediction
    # y_pred = prediction(X_test, model)
    # print(y_test)
    # print(y_pred)
    # Calculate the accuracy
    # cal_accuracy(y_test, y_pred)

    kf = KFold(n_splits=15) # chia tap du lieu thanh 15 phan
    X = irisData.data
    y = irisData.target

    scores = []
    # totalScore = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index, ], X[test_index, ]
        y_train, y_test = y[train_index], y[test_index]
        model = GaussianNB()
        model.fit(X_train, y_train)
        # y_pred = model.predict(X_test)
        # totalScore += accuracy_score(y_test, y_pred)

        scores.append(model.score(X_test, y_test))
    
    print(np.mean(scores))
    # print(totalScore/15)

# calling main function
if __name__=="__main__":
    main()