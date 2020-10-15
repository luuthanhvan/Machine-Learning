import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
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

# import dataset from iris_data.csv
def importData():
    iris_dt = pd.read_csv("../data_set/iris_data.csv", delimiter=",", header=None)
    return iris_dt

# split dataset: data training and data testing (hold-out method)
def splitDataset(iris_dt):
    X = iris_dt.data
    Y = iris_dt.target
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=1/3.0, random_state=100)
    #print("data training: ")
    #print(X_train)
    #print("data testing: ")
    #print(X_test)
    return X, Y, X_train, X_test, y_train, y_test

def splitDatasetFromFile(iris_dt):
    X = iris_dt.iloc[1:, 0:4]
    Y = iris_dt.iloc[1:, 4:]
    #print(X)
    #print(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=1/3.0, random_state=100)
    return X, Y, X_train, X_test, y_train, y_test
    
# Decision Tree model
def train_using_gini(X_train, y_train):
    # creating the classifier object
    clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)
    # performing training
    clf_gini.fit(X_train, y_train)
    return clf_gini

def train_using_entropy(X_train, y_train):
    # decision tree with entropy
    clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=3, min_samples_leaf=5)
    # performing training
    clf_entropy.fit(X_train, y_train)
    return clf_entropy

# Prediction
def prediction(X_test, clf_object):
    y_pred = clf_object.predict(X_test)
    #y_pred = clf_gini.predict([[8, 3, 7, 2]])
    #print(y_pred)
    return y_pred

# Calculating accuracy
def cal_accuracy(y_test, y_pred):
    # Calculating accuracy using confusion matrix
    #print(confusion_matrix(y_test, y_pred, labels=[2,0,1])) # for iris data set in sklearn
    print(confusion_matrix(y_test, y_pred, labels=["Iris-setosa", "Iris-versicolor", "Iris-virginica"]))
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
    '''
    # 1. Building phrase
    data = loadData()
    X, Y, X_train, X_test, y_train, y_test = splitDataset(data) # Split the dataset from train and test using Python sklearn package.
    clf_gini = train_using_gini(X_train, y_train) # Train the classifier using Gini index
    clf_entropy = train_using_entropy(X_train, y_train) # Train the classifier using Entropy

    # 2. Opeational phrase
    print("Results using Gini index: ")
    # Prediction using Gini index
    y_pred_gini = prediction(X_test, clf_gini)
    # Calculate the accuracy
    cal_accuracy(y_test, y_pred_gini)

    print("Result using Entropy: ")
    # prediction using Entropy
    y_pred_entropy = prediction(X_test, clf_entropy)
    # Calculate the accuracy
    cal_accuracy(y_test, y_pred_entropy) '''

    # 1. Building phrase
    data = importData()
    X, Y, X_train, X_test, y_train, y_test = splitDatasetFromFile(data) # Split the dataset from train and test using Python sklearn package.
    clf_gini = train_using_gini(X_train, y_train) # Train the classifier using Gini index
    clf_entropy = train_using_entropy(X_train, y_train) # Train the classifier using Entropy

    # 2. Opeational phrase
    print("Results using Gini index: ")
    # Prediction using Gini index
    y_pred_gini = prediction(X_test, clf_gini)
    # Calculate the accuracy
    cal_accuracy(y_test, y_pred_gini)

    print("Result using Entropy: ")
    # prediction using Entropy
    y_pred_entropy = prediction(X_test, clf_entropy)
    # Calculate the accuracy
    cal_accuracy(y_test, y_pred_entropy)

# calling main function
if __name__=="__main__":
    main()