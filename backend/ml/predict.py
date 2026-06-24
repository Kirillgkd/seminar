from pathlib import Path
import pickle
import pandas as pd


def predict_with_model(model_name: str, input_data: dict) -> dict:
    """Загрузить обученную модель и сделать предсказание."""
    model_path = Path("models") / f"{model_name}.pkl"

    if not model_path.is_file():
        raise FileNotFoundError(f"Модель '{model_name}' не найдена: {model_path}")

    with open(model_path, "rb") as f:
        saved = pickle.load(f)

    model = saved["model"]
    features = saved["features"]
    scaler = saved.get("scaler")
    fill_values = saved.get("fill_values", {})

    df = pd.DataFrame([input_data])

    # Такое же заполнение пропусков, как при обучении
    for col, value in fill_values.items():
        if col in df.columns:
            df[col] = df[col].fillna(value)

    df = pd.get_dummies(df, drop_first=True)

    # Выравниваем колонки под признаки обучения
    for col in features:
        if col not in df.columns:
            df[col] = 0
    df = df[features]

    # Применяем масштабирование, если оно было при обучении
    if scaler is not None:
        df = pd.DataFrame(scaler.transform(df), columns=features)

    prediction = model.predict(df)
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df).tolist()

    return {
        "prediction": prediction.tolist(),
        "probabilities": proba,
        "task_type": saved.get("task_type"),
        "target": saved.get("target"),
    }
