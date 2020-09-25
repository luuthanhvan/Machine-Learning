import pandas as pd

tsv_file = open("../data_set/web_traffic.tsv")
data = pd.read_csv(tsv_file, delimiter = "\t", header=None)

#print(data)
#print(data.shape)

x = data.dropna(axis='rows').iloc[:, 0] # data in the first column \ nan elements
y = data.dropna(axis='rows').iloc[:, 1] # data in the second column \ nan elements
#print(x)
#print(y)

import matplotlib.pyplot as plt

plt.scatter(x, y)
plt.title("Web traffic over the last month")
plt.xlabel("Time")
plt.ylabel("Hits/hour")
plt.xticks([w*7*24 for w in range(10)], ['week %i'%w for w in range(10)])
plt.autoscale(tight=True)
plt.grid()
plt.show()


