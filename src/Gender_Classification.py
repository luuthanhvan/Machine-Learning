import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def initData():
    X = [
        [180, 15, 0],
        [167, 42, 1],
        [136, 35, 1],
        [174, 15, 0],
        [141, 28, 1]
    ]
    Y = [0,1,1,0,1]
    return X, Y

def train_using_entropy(X_train, y_train):
    # decision tree with entropy
    clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=5, max_depth=2, min_samples_leaf=1)
    # performing training
    clf_entropy.fit(X_train, y_train)
    return clf_entropy

def prediction(X_test, clf_object):
    y_pred = clf_object.predict(X_test)
    #print(y_pred)
    return y_pred

# Calculating accuracy
def cal_accuracy(y_test, y_pred):
    # Calculating accuracy using confusion matrix
    #print(confusion_matrix(y_test, y_pred, labels=[0,1]))
    print("Accuracy is ", accuracy_score(y_test, y_pred)*100)

def main():
    X_train, y_train = initData()
    clf_entropy = train_using_entropy(X_train, y_train)
    y_pred = prediction([[182, 25, 0]], clf_entropy)
    print(y_pred)

# calling main function
if __name__=="__main__":
    main()