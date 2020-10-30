import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Câu 6:
wineWhite = pd.read_csv("../data_set/winequality-white.csv", delimiter=";")
X = wineWhite.iloc[:, 0:11]
y = wineWhite.iloc[:, 11:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

print("Naive Bayes")
model_NB = GaussianNB()
model_NB.fit(X_train, y_train)
y_pred_NB = model_NB.predict(X_test)
print("Confusion matrix: ")
print(confusion_matrix(y_test, y_pred_NB, labels=np.unique(y_test)))
print("Accuracy is ", accuracy_score(y_test, y_pred_NB)*100)
print("========================================")

print("\n")
print("Decision Tree")
model_DT = DecisionTreeClassifier(criterion="entropy", random_state=10, max_depth=11, min_samples_leaf=3)
model_DT.fit(X_train, y_train)
y_pred_DT = model_DT.predict(X_test)
print("Confusion matrix: ")
print(confusion_matrix(y_test, y_pred_DT, labels=np.unique(y_test)))
print("Accuracy is ", accuracy_score(y_test, y_pred_DT)*100)
print("========================================")
