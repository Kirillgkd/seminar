"""Интерпретация результатов обучения модели через LLM.

Приоритет провайдеров:
  1. Anthropic (Claude)   — если задан ANTHROPIC_API_KEY
  2. OpenAI-совместимый     — если задан OPENAI_API_KEY
  3. Резерв по правилам    — работает офлайн
"""
import os
import json


def _build_prompt(record: dict) -> str:
    metrics = record.get("metrics", {})
    metrics_clean = {k: v for k, v in metrics.items() if k != "confusion_matrix"}
    return (
        "Ты — ML-эксперт. Кратко (на русском) интерпретируй результаты обучения модели "
        "для пользователя без глубоких знаний ML. Объясни качество модели, на что обратить "
        "внимание, и дай 1-2 совета по улучшению.\n\n"
        f"Тип задачи: {record.get('task_type')}\n"
        f"Модель: {record.get('model_type')}\n"
        f"Датасет: {record.get('filename')}, целевая: {record.get('target_column')}\n"
        f"Гиперпараметры: {record.get('params')}\n"
        f"Метрики: {json.dumps(metrics_clean, ensure_ascii=False)}\n"
    )


def _rule_based(record: dict) -> str:
    """Офлайн-интерпретация по порогам метрик."""
    task = record.get("task_type")
    metrics = record.get("metrics", {})
    lines = []

    if task == "classification":
        acc = metrics.get("accuracy", 0)
        f1 = metrics.get("f1_score", 0)
        lines.append(f"Модель классификации **{record.get('model_type')}** обучена.")
        lines.append(f"Accuracy = {acc:.3f}, F1 = {f1:.3f}.")
        if acc >= 0.9:
            lines.append("Качество отличное — модель уверенно разделяет классы.")
        elif acc >= 0.75:
            lines.append("Качество хорошее, но есть пространство для улучшения.")
        else:
            lines.append("Качество низкое — стоит поработать с признаками или моделью.")
    else:
        r2 = metrics.get("r2", 0)
        rmse = metrics.get("rmse", 0)
        lines.append(f"Модель регрессии **{record.get('model_type')}** обучена.")
        lines.append(f"R2 = {r2:.3f}, RMSE = {rmse:.3f}.")
        if r2 >= 0.8:
            lines.append("Модель хорошо объясняет вариацию целевой переменной.")
        elif r2 >= 0.5:
            lines.append("Модель объясняет данные умеренно.")
        else:
            lines.append("Низкий R2 — модель плохо описывает данные.")

    # Самые важные признаки
    fi = metrics.get("feature_importance", {})
    if fi:
        top = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:3]
        feats = ", ".join(f"`{name}`" for name, _ in top)
        lines.append(f"Наиболее важные признаки: {feats}.")

    lines.append(
        "**Советы:** попробуйте другие модели, настройте гиперпараметры, "
        "добавьте больше данных или включите масштабирование признаков."
    )
    return "\n\n".join(lines)


def _anthropic(prompt: str, api_key: str) -> str:
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)
    model = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    resp = client.messages.create(
        model=model,
        max_tokens=600,
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def _openai(prompt: str, api_key: str) -> str:
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content


def interpret_results(record: dict) -> dict:
    """Return an LLM (or rule-based) interpretation of a training record."""
    prompt = _build_prompt(record)

    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if anthropic_key:
        try:
            return {"source": "anthropic", "interpretation": _anthropic(prompt, anthropic_key)}
        except Exception as e:
            return {
                "source": "rule_based_fallback",
                "interpretation": _rule_based(record),
                "error": f"Anthropic error: {e}",
            }

    if openai_key:
        try:
            return {"source": "openai", "interpretation": _openai(prompt, openai_key)}
        except Exception as e:
            return {
                "source": "rule_based_fallback",
                "interpretation": _rule_based(record),
                "error": f"OpenAI error: {e}",
            }

    return {"source": "rule_based", "interpretation": _rule_based(record)}
