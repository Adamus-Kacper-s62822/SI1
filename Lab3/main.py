import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv('data.csv')

x = data.iloc[:, 0:2].values
y = data.iloc[:, 2].values

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(x, y)

print(clf.predict(x))