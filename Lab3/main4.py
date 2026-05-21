import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor, VotingRegressor, StackingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


RANDOM_STATE = 42
TARGET_COLUMN = "ViolentCrimesPerPop"

COLUMNS_TO_DROP = [
    "state",
    "county",
    "community",
    "communityname",
    "fold"
]


def load_column_names(file_path):
    column_names = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            match = re.match(r"@attribute\s+(\S+)\s+", line.strip())

            if match:
                column_names.append(match.group(1))

    return column_names


def load_data():
    column_names = load_column_names("communities.names")

    data = pd.read_csv(
        "communities.data",
        header=None,
        names=column_names,
        na_values="?"
    )

    return data


def prepare_data(data):
    X = data.drop(columns=COLUMNS_TO_DROP + [TARGET_COLUMN])
    y = data[TARGET_COLUMN]

    X = X.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE
    )


def create_pipeline(model, use_scaler=False):
    steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if use_scaler:
        steps.append(("scaler", StandardScaler()))

    steps.append(("model", model))

    return Pipeline(steps)


def get_models():
    simple_models = [
        ("knn", KNeighborsRegressor(n_neighbors=5)),
        ("ridge", Ridge(alpha=1.0)),
        ("linear", LinearRegression())
    ]

    return {
        "KNN": create_pipeline(
            KNeighborsRegressor(n_neighbors=5),
            use_scaler=True
        ),

        "Regresja liniowa": create_pipeline(
            LinearRegression(),
            use_scaler=True
        ),

        "Ridge": create_pipeline(
            Ridge(alpha=1.0),
            use_scaler=True
        ),

        "Drzewo decyzyjne": create_pipeline(
            DecisionTreeRegressor(max_depth=6, random_state=RANDOM_STATE)
        ),

        "Random Forest": create_pipeline(
            RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE)
        ),

        "Gradient Boosting": create_pipeline(
            GradientBoostingRegressor(random_state=RANDOM_STATE)
        ),

        "Bagging": create_pipeline(
            BaggingRegressor(
                estimator=DecisionTreeRegressor(random_state=RANDOM_STATE),
                n_estimators=100,
                random_state=RANDOM_STATE
            )
        ),

        "Voting": create_pipeline(
            VotingRegressor(simple_models),
            use_scaler=True
        ),

        "Stacking": create_pipeline(
            StackingRegressor(simple_models),
            use_scaler=True
        ),

        "Sieć neuronowa": create_pipeline(
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                max_iter=2000,
                random_state=RANDOM_STATE
            ),
            use_scaler=True
        )
    }


def evaluate_models(models, X_train, X_test, y_train, y_test):
    results = []
    predictions = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        predictions[name] = y_pred

        mse = mean_squared_error(y_test, y_pred)

        results.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, y_pred),
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "R2": r2_score(y_test, y_pred)
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="RMSE")

    return results_df, predictions


def calculate_ethnic_correlations(data):
    columns = [
        "racePctWhite",
        "racepctblack",
        "racePctAsian",
        "racePctHisp",
        TARGET_COLUMN
    ]

    labels = {
        "racePctWhite": "Biali",
        "racepctblack": "Czarni",
        "racePctAsian": "Azjaci",
        "racePctHisp": "Latynosi"
    }

    df = data[columns].apply(pd.to_numeric, errors="coerce")

    correlations = df.corr()[TARGET_COLUMN]
    correlations = correlations.drop(TARGET_COLUMN)
    correlations.index = [labels[column] for column in correlations.index]

    return correlations


def add_values_to_bars(ax, bars, decimals=4):
    for bar in bars:
        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.{decimals}f}",
            ha="center",
            va="bottom" if height >= 0 else "top"
        )


def show_charts(y_test, best_prediction, results_df, correlations):
    best_model = results_df.iloc[0]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    fig.suptitle(
        "Analiza poziomu przestępczości w zbiorze Communities and Crime USA",
        fontsize=16
    )

    # Wykres 1: wartości rzeczywiste vs przewidziane
    ax = axes[0, 0]

    ax.scatter(y_test, best_prediction, alpha=0.7)
    ax.set_title(f"Rzeczywiste vs przewidziane\nNajlepszy model: {best_model['Model']}")
    ax.set_xlabel("Wartości rzeczywiste")
    ax.set_ylabel("Wartości przewidziane")

    min_value = min(y_test.min(), best_prediction.min())
    max_value = max(y_test.max(), best_prediction.max())

    ax.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--",
        label="Idealna prognoza"
    )

    ax.legend()

    ax.text(
        0.05,
        0.95,
        f"MAE: {best_model['MAE']:.4f}\n"
        f"RMSE: {best_model['RMSE']:.4f}\n"
        f"R²: {best_model['R2']:.4f}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    # Wykres 2: reszty
    residuals = y_test - best_prediction

    ax = axes[0, 1]

    ax.scatter(best_prediction, residuals, alpha=0.7)
    ax.axhline(y=0, linestyle="--", label="Brak błędu")
    ax.axhline(
        y=residuals.mean(),
        linestyle=":",
        label=f"Średni błąd: {residuals.mean():.4f}"
    )

    ax.set_title("Wykres reszt")
    ax.set_xlabel("Wartości przewidziane")
    ax.set_ylabel("Błąd predykcji")
    ax.legend()

    ax.text(
        0.05,
        0.95,
        f"Największy błąd dodatni: {residuals.max():.4f}\n"
        f"Największy błąd ujemny: {residuals.min():.4f}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    # Wykres 3: korelacje
    ax = axes[1, 0]

    bars = ax.bar(correlations.index, correlations.values)
    ax.set_title("Korelacja grup etnicznych z poziomem przestępczości")
    ax.set_ylabel("Współczynnik korelacji")
    ax.tick_params(axis="x", rotation=45)
    ax.axhline(y=0, linewidth=0.8)

    add_values_to_bars(ax, bars, decimals=2)

    # Wykres 4: porównanie modeli
    ax = axes[1, 1]

    bars = ax.bar(results_df["Model"], results_df["RMSE"])
    ax.set_title("Porównanie modeli według RMSE")
    ax.set_ylabel("RMSE — im mniej, tym lepiej")
    ax.tick_params(axis="x", rotation=45)

    add_values_to_bars(ax, bars, decimals=4)

    ax.text(
        0.05,
        0.95,
        f"Najlepszy model:\n{best_model['Model']}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def main():
    data = load_data()

    X_train, X_test, y_train, y_test = prepare_data(data)

    models = get_models()

    results_df, predictions = evaluate_models(
        models,
        X_train,
        X_test,
        y_train,
        y_test
    )

    print(results_df)

    best_model_name = results_df.iloc[0]["Model"]
    best_prediction = predictions[best_model_name]

    correlations = calculate_ethnic_correlations(data)

    show_charts(
        y_test,
        best_prediction,
        results_df,
        correlations
    )


if __name__ == "__main__":
    main()