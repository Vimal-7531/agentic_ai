import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

database = BASE_DIR / "data" / "telecom_ops.db"
schema_file = BASE_DIR / "sql" / "01_schema.sql"
seed_file = BASE_DIR / "sql" / "02_seed_data.sql"

database.parent.mkdir(parents=True, exist_ok=True)

connection = sqlite3.connect(database)

try:
    with open(schema_file, "r", encoding="utf-8") as file:
        connection.executescript(file.read())

    with open(seed_file, "r", encoding="utf-8") as file:
        connection.executescript(file.read())

    connection.commit()
    print(f"Database created successfully: {database}")

finally:
    connection.close()