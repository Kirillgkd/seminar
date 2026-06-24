import os
import json
import requests
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("📊 Обученные модели")

st.button("🔄 Обновить")


def parse_metrics(value):
    if isinstance(value, dict):
        return value
    try:
        return json.loads(value or "{}")
    except (TypeError, ValueError):
        return {}


try:
    r = requests.get(f"{API_URL}/models/trained")
except requests.exceptions.ConnectionError:
    st.error("Не удалось подключиться к API. Убедитесь, что бэкенд запущен.")
    st.stop()

if r.status_code != 200:
    st.error(f"Ошибка API: {r.text}")
    st.stop()

results = r.json()["results"]
if not results:
    st.info("Пока нет обученных моделей. Перейдите на страницу 'Обучение'.")
    st.stop()

# Таблица с извлечёнными метриками
rows = []
for res in results:
    m = parse_metrics(res.get("metrics"))
    rows.append({
        "id": res["id"],
        "model_name": res["model_name"],
        "task_type": res.get("task_type"),
        "model_type": res["model_type"],
        "filename": res.get("filename"),
        "target": res.get("target_column"),
        "train_size": res.get("train_size"),
        "accuracy": m.get("accuracy"),
        "f1_score": m.get("f1_score"),
        "r2": m.get("r2"),
        "rmse": m.get("rmse"),
        "mae": m.get("mae"),
        "created_at": res.get("created_at"),
    })

df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True)

st.markdown("---")
st.subheader("Сравнение моделей")

clf = df[df["task_type"] == "classification"].dropna(subset=["accuracy"])
reg = df[df["task_type"] == "regression"].dropna(subset=["r2"])

c1, c2 = st.columns(2)
with c1:
    if not clf.empty:
        fig = px.bar(clf, x="model_name", y="accuracy", color="model_type",
                     title="Accuracy (классификация)", range_y=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
with c2:
    if not reg.empty:
        fig = px.bar(reg, x="model_name", y="r2", color="model_type",
                     title="R² (регрессия)")
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("Детали модели")
selected = st.selectbox("Выберите модель", df["model_name"].tolist())

if selected:
    dr = requests.get(f"{API_URL}/models/{selected}/details")
    if dr.status_code == 200:
        detail = dr.json()
        metrics = detail.get("metrics", {})

        # Карточки метрик
        cols = st.columns(3)
        if detail.get("task_type") == "classification":
            cols[0].metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
            cols[1].metric("F1-score", f"{metrics.get('f1_score', 0):.4f}")
        else:
            cols[0].metric("R²", f"{metrics.get('r2', 0):.4f}")
            cols[1].metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
            cols[2].metric("MAE", f"{metrics.get('mae', 0):.4f}")

        g1, g2 = st.columns(2)

        # Важность признаков
        with g1:
            fi = metrics.get("feature_importance", {})
            if fi:
                fi_df = pd.DataFrame(
                    sorted(fi.items(), key=lambda x: x[1], reverse=True)[:15],
                    columns=["feature", "importance"],
                )
                fig = px.bar(fi_df, x="importance", y="feature", orientation="h",
                             title="Важность признаков")
                fig.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)

        # Матрица ошибок
        with g2:
            cm = metrics.get("confusion_matrix")
            labels = metrics.get("labels")
            if cm and labels:
                fig = go.Figure(data=go.Heatmap(
                    z=cm, x=labels, y=labels, colorscale="Blues",
                    text=cm, texttemplate="%{text}",
                ))
                fig.update_layout(title="Матрица ошибок",
                                  xaxis_title="Предсказано", yaxis_title="Факт")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.error(f"Ошибка: {dr.text}")

st.markdown("---")
st.subheader("Удаление модели")
model_to_delete = st.selectbox("Модель для удаления", [""] + df["model_name"].tolist())
if st.button("🗑️ Удалить"):
    if model_to_delete:
        dr = requests.delete(f"{API_URL}/models/{model_to_delete}")
        if dr.status_code == 200:
            st.success(dr.json()["message"])
            st.rerun()
        else:
            st.error(f"Ошибка: {dr.text}")
