from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


FEATURES = ("day_of_week", "month", "day_of_month", "is_weekend", "lag_1", "lag_7", "rolling_mean_7")


def _calendar_features(date: pd.Timestamp) -> dict[str, float]:
    return {
        "day_of_week": float(date.dayofweek),
        "month": float(date.month),
        "day_of_month": float(date.day),
        "is_weekend": float(date.dayofweek >= 5),
    }


def build_supervised_frame(daily: pd.DataFrame, target: str) -> pd.DataFrame:
    if target not in {"revenue", "quantity"}:
        raise ValueError("Forecast target must be `revenue` or `quantity`.")
    ordered = daily[["date", target]].copy()
    ordered["date"] = pd.to_datetime(ordered["date"])
    ordered = ordered.sort_values("date").reset_index(drop=True)
    if ordered["date"].duplicated().any():
        raise ValueError("Daily data must have one row per date before feature engineering.")
    values = ordered[target].astype(float)
    features = pd.DataFrame([_calendar_features(date) for date in ordered["date"]])
    features["lag_1"] = values.shift(1)
    features["lag_7"] = values.shift(7)
    features["rolling_mean_7"] = values.shift(1).rolling(window=7, min_periods=7).mean()
    frame = pd.concat([ordered, features], axis=1).dropna().reset_index(drop=True)
    if len(frame) < 10:
        raise ValueError("Not enough history to create lag features and a chronological test set. Provide at least 17 daily records.")
    return frame


def _new_model() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=250,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )


def evaluate_model(daily: pd.DataFrame, target: str) -> dict[str, Any]:
    supervised = build_supervised_frame(daily, target)
    split = int(len(supervised) * 0.8)
    if split < 2 or len(supervised) - split < 2:
        raise ValueError("Not enough feature-ready observations for a reliable chronological holdout. Add more daily history.")
    train = supervised.iloc[:split]
    test = supervised.iloc[split:]
    model = _new_model()
    model.fit(train.loc[:, FEATURES], train[target])
    predictions = np.maximum(0, model.predict(test.loc[:, FEATURES]))
    actual = test[target].to_numpy(dtype=float)
    r2 = float(r2_score(actual, predictions)) if len(actual) >= 2 else None
    results = pd.DataFrame({
        "date": test["date"].to_numpy(),
        "actual": actual,
        "predicted": predictions,
        "absolute_error": np.abs(actual - predictions),
    })
    return {
        "model": model,
        "target": target,
        "metrics": {
            "mae": float(mean_absolute_error(actual, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(actual, predictions))),
            "r2": r2,
        },
        "train_size": len(train),
        "test_size": len(test),
        "split_date": pd.Timestamp(test["date"].iloc[0]),
        "test_results": results,
    }


def forecast_future(
    daily: pd.DataFrame,
    result: dict[str, Any],
    target: str,
    horizon: int,
) -> pd.DataFrame:
    if horizon < 1:
        raise ValueError("Forecast horizon must be a positive number of days.")
    ordered = daily[["date", target]].copy()
    ordered["date"] = pd.to_datetime(ordered["date"])
    ordered = ordered.sort_values("date").reset_index(drop=True)
    if ordered["date"].duplicated().any():
        raise ValueError("Daily data must have one row per date before forecasting.")
    values = ordered[target].astype(float).tolist()
    if len(values) < 7:
        raise ValueError("At least 7 historical values are required for lagged forecasting.")
    dates: list[pd.Timestamp] = []
    predictions: list[float] = []
    model = result["model"]
    for offset in range(1, horizon + 1):
        date = ordered["date"].iloc[-1] + pd.Timedelta(days=offset)
        features = _calendar_features(date)
        features["lag_1"] = values[-1]
        features["lag_7"] = values[-7]
        features["rolling_mean_7"] = float(np.mean(values[-7:]))
        predicted = max(0.0, float(model.predict(pd.DataFrame([features], columns=FEATURES))[0]))
        dates.append(date)
        predictions.append(predicted)
        values.append(predicted)
    return pd.DataFrame({"date": dates, target: predictions})