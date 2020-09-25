import pandas as pd

# read data from file bt1_sang6.csv
data = pd.read_csv("../data_set/graph.csv", delimiter = ";", header=None)

# display data
print(data)

# display data in column 3
print(data.iloc[:,2])

# display data in column from 8 to 14
print(data.iloc[7:14])

# display data in row 10, column 2 and 3
print(data.iloc[9:10, 1:3])

# data in column 2 assign to x
# data in column 2 assign to y
# represent data of x and y in coordinate plane
x = data.iloc[:, 1]
y = data.iloc[:, 2]
#print(x)
#print(y)
import matplotlib.pyplot as plt
plt.scatter(x, y)
plt.title("Graph")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.show()