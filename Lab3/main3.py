import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

column_names = []

with open("communities.names", "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        line = line.strip()
        match = re.match(r"@attribute\s+(\S+)\s+", line)
        if match:
            column_names.append(match.group(1))

data = pd.read_csv(
    "communities.data",
    header=None,
    names=column_names,
    na_values="?"
)

target_col = "ViolentCrimesPerPop"
drop_cols = ["state", "county", "community", "communityname", "fold"]

X = data.drop(columns=drop_cols + [target_col]).apply(pd.to_numeric, errors="coerce")
y = data[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("knn", KNeighborsRegressor(n_neighbors=5))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2:", r2_score(y_test, y_pred))

cols = [
    "racePctWhite",
    "racepctblack",
    "racePctAsian",
    "racePctHisp",
    "ViolentCrimesPerPop"
]

df = data[cols].apply(pd.to_numeric, errors="coerce")
corr = df.corr()["ViolentCrimesPerPop"].drop("ViolentCrimesPerPop")

labels = {
    "racePctWhite": "Biali",
    "racepctblack": "Czarni",
    "racePctAsian": "Azjaci",
    "racePctHisp": "Latynosi"
}
corr.index = [labels.get(col, col) for col in corr.index]

group_colors = ["lightblue", "lightcoral", "gold", "lightgreen"]

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

fig.suptitle(
    "Analiza poziomu przestępczości w zbiorze Communities and Crime USA",
    fontsize=14
)

axes[0].scatter(y_test, y_pred, color="steelblue", alpha=0.7)
axes[0].set_xlabel("Wartości rzeczywiste")
axes[0].set_ylabel("Wartości przewidziane")
axes[0].set_title("Rzeczywiste vs przewidziane")

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
axes[0].plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")

residuals = y_test - y_pred
axes[1].scatter(y_pred, residuals, color="darkorange", alpha=0.7)
axes[1].axhline(y=0, color="red", linestyle="--")
axes[1].set_xlabel("Wartości przewidziane")
axes[1].set_ylabel("Błąd")
axes[1].set_title("Wykres reszt")

axes[2].bar(corr.index, corr.values, color=group_colors)
axes[2].set_title("Korelacja grup etnicznych")
axes[2].set_ylabel("Korelacja")
axes[2].tick_params(axis="x", rotation=45)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()