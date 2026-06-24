from pathlib import Path
import os
import json
import sqlite3

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel

# Загружаем переменные окружения из .env (ключи API и т.д.)
load_dotenv(override=True)

from backend.ml.train import (
    train_model,
    CLASSIFICATION_MODELS,
    REGRESSION_MODELS,
)
from backend.ml.predict import predict_with_model
from backend.ml.interpret import interpret_results

app = FastAPI(title="ML Service API", version="1.0.0")


def get_record(model_name: str):
    """Достать последнюю запись об обучении модели из БД."""
    conn = sqlite3.connect("models.db")
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM training_results WHERE model_name = ? ORDER BY id DESC LIMIT 1",
        (model_name,),
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail=f"Модель '{model_name}' не найдена")
    record = dict(row)
    record["metrics"] = json.loads(record.get("metrics") or "{}")
    record["params"] = json.loads(record.get("params") or "{}")
    return record


class TrainRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    filename: str
    model_name: str
    model_type: str = "logistic_regression"
    task_type: str = "classification"
    target_column: str = ""
    train_size: float = 0.8
    params: dict = {}
    scale_features: bool = False
    fill_strategy: str = "mean"


class PredictRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_name: str
    input_data: dict


class InterpretRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_name: str


@app.get("/")
def root():
    return {"message": "ML Service API работает"}


@app.get("/models/available")
def available_models():
    """Доступные модели по типам задач с параметрами по умолчанию."""
    return {
        "classification": {
            "logistic_regression": {"max_iter": 1000, "C": 1.0},
            "random_forest": {"n_estimators": 100, "max_depth": None},
            "gradient_boosting": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
        },
        "regression": {
            "linear_regression": {},
            "ridge": {"alpha": 1.0},
            "random_forest": {"n_estimators": 100, "max_depth": None},
            "gradient_boosting": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
        },
    }


@app.get("/models/trained")
def trained_models():
    """Список обученных моделей из БД."""
    conn = sqlite3.connect("models.db")
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM training_results ORDER BY id DESC").fetchall()
    conn.close()
    return {"results": [dict(r) for r in rows]}


@app.get("/data/files")
def list_data_files():
    """Список загруженных файлов данных."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    files = [f.name for f in data_dir.iterdir() if f.is_file() and f.suffix == ".csv"]
    return {"files": files}


@app.get("/data/columns")
def data_columns(filename: str):
    """Колонки CSV-файла (для выбора целевой переменной)."""
    path = Path("data") / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Файл '{filename}' не найден")
    df = pd.read_csv(path, nrows=5)
    return {"columns": list(df.columns)}


@app.post("/train")
def train(req: TrainRequest, background_tasks: BackgroundTasks):
    """Запустить обучение модели в фоне."""
    path = Path("data") / req.filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Файл данных '{req.filename}' не найден")

    if req.task_type not in ("classification", "regression"):
        raise HTTPException(status_code=400, detail=f"Неизвестный тип задачи '{req.task_type}'")

    valid = CLASSIFICATION_MODELS if req.task_type == "classification" else REGRESSION_MODELS
    if req.model_type not in valid:
        raise HTTPException(
            status_code=400,
            detail=f"Модель '{req.model_type}' недоступна для '{req.task_type}'. Доступно: {list(valid.keys())}",
        )

    background_tasks.add_task(
        train_model,
        req.filename,
        req.model_name,
        req.model_type,
        req.task_type,
        req.target_column,
        req.train_size,
        req.params,
        req.scale_features,
        req.fill_strategy,
    )
    return {
        "message": f"Модель '{req.model_name}' ({req.task_type}/{req.model_type}) отправлена на обучение"
    }


@app.post("/predict")
def predict(req: PredictRequest):
    """Предсказание обученной моделью."""
    try:
        result = predict_with_model(req.model_name, req.input_data)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/{model_name}/details")
def model_details(model_name: str):
    """Полная запись об обучении модели (включая метрики)."""
    return get_record(model_name)


@app.post("/interpret")
def interpret(req: InterpretRequest):
    """Интерпретация результатов обучения модели."""
    return interpret_results(get_record(req.model_name))


@app.delete("/models/{model_name}")
def delete_model(model_name: str):
    """Удалить обученную модель."""
    model_path = Path("models") / f"{model_name}.pkl"
    if model_path.is_file():
        os.remove(model_path)

    conn = sqlite3.connect("models.db")
    conn.execute("DELETE FROM training_results WHERE model_name = ?", (model_name,))
    conn.commit()
    conn.close()
    return {"message": f"Модель '{model_name}' удалена"}
