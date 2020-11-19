import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def readFile():
    wineData = pd.read_csv("../data_set/winequality-red.csv", delimiter=";")
    return wineData

def splitDataset(wineData):
    X = wineData.iloc[:, 0:11]
    Y = wineData.iloc[:, 11:]
    #print(X)
    #print(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=4/10.0, random_state=200)
    return X, Y, X_train, X_test, y_train, y_test

def train_using_entropy(X_train, y_train):
    # decision tree with entropy
    clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=200, max_depth=11, min_samples_leaf=3)
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
    print(confusion_matrix(y_test, y_pred, labels=np.unique(y_test)))
    print("Accuracy is ", accuracy_score(y_test, y_pred)*100)

def main():
    # a. Đọc file
    wineData = readFile()
    
    ''' b. 
        - Tập dữ liệu có 4898 phần tử
        - Cột nhãn (quality) có 11 giá trị tương ứng từ 0-10
    '''
    
    # c. Phân chia tập dữ liệu: 4 phần test, 6 phần train
    X, Y, X_train, X_test, y_train, y_test = splitDataset(wineData)
    #print(y_test.value_counts())
    ''' 
    Số lượng phần tử và nhãn của các phần tử thuộc tập test
        nhãn        Số phần tử
        5           299
        6           251
        7           66
        4           19
        8            3
        3            2
    '''

    # d. Xây dựng mô hình cây quyết định dựa vào chỉ số độ lợi thông tin
    clf_entropy = train_using_entropy(X_train, y_train)

    # e. Đánh giá độ chính xác tổng thể và độ chính xác cho từng lớp cho toàn bộ dữ liệu trong tập test
    y_pred_all = prediction(X_test, clf_entropy)
    cal_accuracy(y_test, y_pred_all)

    # f. Đánh giá độ chính xác tổng thể và độ chính xác cho từng lớp cho 6 phần tử đầu tiên trong tập test
    y_pred_6 = prediction(X_test.iloc[1:7], clf_entropy)
    cal_accuracy(y_test.iloc[1:7], y_pred_6)

# calling main function
if __name__=="__main__":
    main()