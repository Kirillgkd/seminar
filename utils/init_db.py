import sqlite3

conn = sqlite3.connect("models.db")
conn.execute("""
CREATE TABLE IF NOT EXISTS training_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    model_type TEXT NOT NULL,
    task_type TEXT NOT NULL DEFAULT 'classification',
    filename TEXT,
    target_column TEXT,
    train_size REAL,
    params TEXT,
    metrics TEXT,
    model_path TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()
conn.close()
print("Database initialized successfully.")
