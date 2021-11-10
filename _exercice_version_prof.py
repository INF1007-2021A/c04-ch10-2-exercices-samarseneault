#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


WINE_DATA_PATH = "./data/winequality-white.csv"


def prepare_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset = load_wine_data()
    x, y = split_attributes(dataset)
    
    return train_test_split(x, y)


def load_wine_data() -> pd.DataFrame:
    return pd.read_csv(WINE_DATA_PATH, delimiter=";")


def split_attributes(dataset: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return dataset[dataset.columns.difference(["quality"])], dataset["quality"]


def train_models(x_train: pd.DataFrame, y_train: pd.DataFrame) -> List[Union[RandomForestRegressor, LinearRegression]]:
    return [
        train_random_forest(x_train, y_train),
        train_linear_regression(x_train, y_train)
    ]


def train_random_forest(x_train: pd.DataFrame, y_train: pd.DataFrame) -> RandomForestRegressor:
    model = RandomForestRegressor()
    model.fit(x_train, y_train)

    return model


def train_linear_regression(x_train: pd.DataFrame, y_train: pd.DataFrame) -> LinearRegression:
    model = LinearRegression()
    model.fit(x_train, y_train)

    return model


def predict_models(models: List[Union[RandomForestRegressor, LinearRegression]], x_test: pd.DataFrame) -> List[np.ndarray]:
    return [model.predict(x_test) for model in models]
    

def evaluate_models(predictions: List[np.ndarray], y_test: pd.DataFrame) -> List[float]:
    return [mean_squared_error(y_test, y_pred) for y_pred in predictions]


def main() -> None:
    x_train, x_test, y_train, y_test = prepare_data()
    print("Prepared data")

    trained_models = train_models(x_train, y_train)
    print("Trained")

    predictions = predict_models(trained_models, x_test)
    print("Predicted")
    
    model_performances = evaluate_models(predictions, y_test)
    print(f"Evaluated models. MSE: {model_performances}")


if __name__ == '__main__':
    main()
