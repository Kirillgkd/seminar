# ML Service — Семинарский проект

Проект на основе учебного [mds-2025_edu/2-Min_ML_Service](https://github.com/KayumovRu/mds-2025_edu/tree/master/2-Min_ML_Service).

## Что добавлено по сравнению с учебным проектом

- **Классификация и регрессия**: выбор типа задачи, своя метрика под каждую
- **Выбор модели**: Logistic/Linear/Ridge Regression, Random Forest, Gradient Boosting
- **Выбор целевой переменной**: можно указать любой столбец как target
- **Препроцессинг**: масштабирование признаков (StandardScaler), заполнение пропусков (mean/median/мода), one-hot кодирование
- **Настройка гиперпараметров**: для каждой модели свои параметры на фронтенде
- **Метрики и графики** (plotly): feature importance, confusion matrix, сравнение моделей
- **AI-интерпретация результатов**: эндпоинт `/interpret` — объяснение качества модели через LLM (OpenAI-совместимый API) с офлайн-fallback на правила
- **Эндпоинт предсказания**: `/predict` — одиночное и пакетное (CSV)
- **Расширенная БД**: тип задачи, тип модели, датасет, целевая, train_size, параметры (JSON), метрики (JSON)
- **Удаление модели**: эндпоинт и UI

## AI / LLM

Интерпретация результатов (`/interpret`) выбирает провайдера по приоритету:
1. **Anthropic (Claude)** — если задан `ANTHROPIC_API_KEY`
2. **OpenAI-совместимый** — если задан `OPENAI_API_KEY`
3. **Офлайн** — встроенная интерпретация по правилам (без ключа), всегда доступна

Ключи задаются через переменные окружения. Скопируйте `.env.example` в `.env` и впишите свой ключ:


Файл `.env` добавлен в `.gitignore` и не попадёт в репозиторий.

## Стек

- **Backend**: FastAPI
- **Frontend**: Streamlit
- **ML**: scikit-learn (LogisticRegression, RandomForestClassifier, GradientBoostingClassifier)
- **БД**: SQLite
- **Контейнеризация**: Docker + docker-compose

## Структура проекта

```
├── backend/
│   ├── api.py              # FastAPI приложение
│   └── ml/
│       ├── train.py        # Обучение (классификация + регрессия, препроцессинг)
│       ├── predict.py      # Предсказание
│       └── interpret.py    # AI/LLM интерпретация результатов
├── frontend/
│   ├── Main.py             # Главная страница Streamlit
│   └── pages/
│       ├── 1_Data.py               # Загрузка и просмотр данных
│       ├── 2_Train.py              # Обучение модели
│       ├── 3_Models.py             # Модели + графики метрик
│       ├── 4_Predict.py            # Предсказание
│       └── 5_AI_Interpretation.py  # AI-интерпретация
├── utils/
│   └── init_db.py          # Инициализация БД
├── data/                   # Папка для датасетов
├── docker-compose.yml
├── Dockerfile.backend
├── Dockerfile.frontend
├── requirements.txt
└── README.md
```

## Запуск (локально)

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Инициализация БД

```bash
python utils/init_db.py
```

### 3. Запуск бэкенда

```bash
uvicorn backend.api:app --reload
```

API будет доступен по адресу: http://localhost:8000/docs

### 4. Запуск фронтенда

```bash
streamlit run frontend/Main.py
```

Фронтенд будет доступен по адресу: http://localhost:8501

## Запуск через Docker

```bash
docker-compose up --build
```

- Frontend: http://localhost:8501
- Backend API: http://localhost:8000/docs

## API эндпоинты

| Метод  | URL               | Описание                          |
|--------|-------------------|-----------------------------------|
| GET    | `/`                       | Проверка работы API                       |
| GET    | `/models/available`       | Доступные модели по задачам               |
| GET    | `/models/trained`         | Список обученных моделей                  |
| GET    | `/models/{name}/details`  | Детали модели + метрики                   |
| GET    | `/data/files`             | Список файлов данных                      |
| GET    | `/data/columns`           | Столбцы CSV (для выбора target)           |
| POST   | `/train`                  | Запуск обучения модели                    |
| POST   | `/predict`                | Предсказание                              |
| POST   | `/interpret`              | AI-интерпретация результатов              |
| DELETE | `/models/{name}`          | Удаление модели                           |

## Пример использования

1. Загрузите CSV файл на странице "Данные" (последний столбец — целевая переменная)
2. Перейдите на "Обучение", выберите модель и параметры
3. Нажмите "Обучить модель"
4. На странице "Модели" проверьте результат
5. На "Предсказание" используйте обученную модель
