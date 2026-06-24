from pathlib import Path
import json
import pickle
import sqlite3

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)


CLASSIFICATION_MODELS = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
}

REGRESSION_MODELS = {
    "linear_regression": LinearRegression,
    "ridge": Ridge,
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
}


def available_model_types(task_type: str):
    return CLASSIFICATION_MODELS if task_type == "classification" else REGRESSION_MODELS


def get_model_instance(task_type: str, model_type: str, params: dict):
    """Создать экземпляр модели по типу задачи и параметрам."""
    models = available_model_types(task_type)
    if model_type not in models:
        raise ValueError(
            f"Неизвестная модель '{model_type}' для задачи '{task_type}'. Доступно: {list(models.keys())}"
        )

    cls = models[model_type]

    if model_type == "logistic_regression":
        return cls(max_iter=params.get("max_iter", 1000), C=params.get("C", 1.0))
    elif model_type == "linear_regression":
        return cls()
    elif model_type == "ridge":
        return cls(alpha=params.get("alpha", 1.0))
    elif model_type == "random_forest":
        return cls(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", None),
            random_state=42,
        )
    elif model_type == "gradient_boosting":
        return cls(
            n_estimators=params.get("n_estimators", 100),
            learning_rate=params.get("learning_rate", 0.1),
            max_depth=params.get("max_depth", 3),
            random_state=42,
        )


def preprocess_features(df: pd.DataFrame, fill_strategy: str):
    """Заполнить пропуски и вернуть df и использованные значения (для предсказания)."""
    fill_values = {}
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                value = df[col].mean() if fill_strategy == "mean" else df[col].median()
            else:
                mode = df[col].mode()
                value = mode.iloc[0] if not mode.empty else ""
            fill_values[col] = value
            df[col] = df[col].fillna(value)
    return df, fill_values


def compute_feature_importance(model, feature_names):
    """Вернуть {признак: важность} для деревьев или линейных моделей."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        importances = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
    else:
        return {}
    return {f: float(i) for f, i in zip(feature_names, importances)}


def train_model(
    filename: str,
    model_name: str,
    model_type: str,
    task_type: str,
    target_column: str,
    train_size: float,
    params: dict,
    scale_features: bool = False,
    fill_strategy: str = "mean",
):
    """Обучить модель с препроцессингом и сохранить результаты в БД."""
    path = Path("data") / filename
    df = pd.read_csv(path)

    # Целевая переменная (по умолчанию — последняя колонка)
    if not target_column:
        target_column = df.columns[-1]
    if target_column not in df.columns:
        raise ValueError(f"Целевая колонка '{target_column}' отсутствует в данных")

    y = df[target_column]
    X_raw = df.drop(columns=[target_column])

    # Заполнение пропусков
    X_raw, fill_values = preprocess_features(X_raw, fill_strategy)

    # Кодирование категориальных признаков
    X = pd.get_dummies(X_raw, drop_first=True)
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=42
    )

    # Опциональное масштабирование
    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train), columns=feature_names, index=X_train.index
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test), columns=feature_names, index=X_test.index
        )

    model = get_model_instance(task_type, model_type, params)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    # Метрики зависят от типа задачи
    if task_type == "classification":
        labels = sorted([str(c) for c in pd.unique(y)])
        cm = confusion_matrix(y_test.astype(str), pred.astype(str), labels=labels)
        metrics = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "f1_score": float(f1_score(y_test, pred, average="weighted")),
            "confusion_matrix": cm.tolist(),
            "labels": labels,
        }
    else:
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        metrics = {
            "r2": float(r2_score(y_test, pred)),
            "mae": float(mean_absolute_error(y_test, pred)),
            "rmse": rmse,
        }

    metrics["feature_importance"] = compute_feature_importance(model, feature_names)

    # Сохраняем модель и параметры препроцессинга
    Path("models").mkdir(exist_ok=True)
    model_path = Path("models") / f"{model_name}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "model": model,
                "features": feature_names,
                "scaler": scaler,
                "fill_values": fill_values,
                "fill_strategy": fill_strategy,
                "task_type": task_type,
                "target": target_column,
            },
            f,
        )

    conn = sqlite3.connect("models.db")
    conn.execute(
        """INSERT INTO training_results
           (model_name, model_type, task_type, filename, target_column,
            train_size, params, metrics, model_path)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            model_name,
            model_type,
            task_type,
            filename,
            target_column,
            train_size,
            json.dumps(params),
            json.dumps(metrics),
            str(model_path),
        ),
    )
    conn.commit()
    conn.close()

    return {"model_path": str(model_path), "metrics": metrics}
