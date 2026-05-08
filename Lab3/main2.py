import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')

x = data.iloc[:, 0:2].values
y = data.iloc[:, 2].values

sns.scatterplot(x=x[:, 0], y=x[:, 1], hue=y)

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(x, y)

print(clf.predict(x))

plt.show()