import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix 

wineWhite = pd.read_csv("../data_set/winequality-white.csv", delimiter=";")
# print(wineWhite)
# print(wineWhite.quality.value_counts())
''' 
Câu 1:
- Tập dữ liệu có 11 thuộc tính: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol
- Cột nhãn: quality, có 7 giá trị lần lượt là 3, 4, 5, 6, 7, 8, 9, cụ thể như sau:
        Nhãn    Số lượng phần tử
        6       2198
        5       1457
        7       880
        8       175
        4       163
        3       20
        9       5
'''

# Câu 2:
X = wineWhite.iloc[:, 0:11]
y = wineWhite.iloc[:, 11:]
# print(X)
# print(Y)
kf = KFold(n_splits=40, shuffle=True) # chia tập dữ liệu thành 40 phần
model = GaussianNB()
scores = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index, ], X.iloc[test_index, ]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    model.fit(X_train, y_train) # Câu 3: sử dụng giải thuật Naive Bayes để dự đoán nhãn cho tập kiểm tra theo nghi thức k-fold với phân phối xác suất Gaussian 
    scores.append(model.score(X_test, y_test)) # tính độ chính xác tổng thể  cho mỗi lần lặp và lưu vào mảng scores
    # tính độ chính xác cho từng phân lớp của mỗi lần lặp
    y_pred = model.predict(X_test)
    cfm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))

# print("X_train")
# print(X_train)
# print("y_train")
# print(y_train)
# print("y_train")
# print(y_train)
# print("y_test")
# print(y_test)
'''
- Số lượng phần tử có trong tập test: X_test = 122 rows x 11 columns, y_test = 122 rows x 1 columns
- Số lượng phần tử có trong tập train: X_train = 4776 rows x 11 columns , y_train = 4776 rows x 1 columns
'''

# Câu 4+5:
# độ chính xác cho từng phần lớp ở lần lặp cuối cùng
print(cfm)
# tính độ chính xác tổng thể của trung bình 40 lần lặp
result = np.mean(scores)
print("Accuracy is ", result*100)

'''
Độ chính xác cho từng phần lớp ở lần lặp cuối cùng:
    [ 1  0  0  0  0  0]
    [ 0  2  2  0  2  0]
    [ 1  1 22 11  2  0]
    [ 0  1 16 23 19  0]
    [ 0  0  2  3 12  0]
    [ 0  0  0  0  2  0]
Độ chính xác tổng thể của trung bình 40 lần lặp: 44.56400773024124
'''


