import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "db.db"


@contextmanager
def _connect(db_path=DB_PATH):
    """Open the project database and enforce SQLite foreign keys."""
    conn = sqlite3.connect(str(Path(db_path).resolve()), timeout=30)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_db(db_path=DB_PATH):
    """Create the database and its tables if they do not already exist."""
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS patient (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_name TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS patient_case (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                serial_number TEXT UNIQUE,
                doctor_name TEXT,
                audiogram TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES patient(id)
            )
            """
        )


def upsert_record(
    serial_number, patient_name, doctor_name, audiogram=None, db_path=DB_PATH
):
    init_db(db_path)
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT id FROM patient WHERE patient_name = ? ORDER BY id LIMIT 1",
            (patient_name,),
        ).fetchone()

        if row:
            patient_id = row[0]
        else:
            cursor = conn.execute(
                "INSERT INTO patient (patient_name) VALUES (?)", (patient_name,)
            )
            patient_id = cursor.lastrowid

        conn.execute(
            """
            INSERT INTO patient_case (
                patient_id, serial_number, doctor_name, audiogram
            )
            VALUES (?, ?, ?, ?)
            ON CONFLICT(serial_number) DO UPDATE SET
                patient_id = excluded.patient_id,
                doctor_name = excluded.doctor_name,
                audiogram = excluded.audiogram,
                updated_at = CURRENT_TIMESTAMP
            """,
            (patient_id, serial_number, doctor_name, audiogram),
        )


def get_database_snapshot(db_path=DB_PATH):
    """Return display-ready patient and case tables for the web UI."""
    init_db(db_path)
    with _connect(db_path) as conn:
        patients = pd.read_sql_query(
            """
            SELECT id AS '病患 ID', patient_name AS '病患姓名', created_at AS '建立時間'
            FROM patient
            ORDER BY id DESC
            """,
            conn,
        )
        cases = pd.read_sql_query(
            """
            SELECT
                pc.id AS '病例 ID',
                pc.serial_number AS '序號',
                p.patient_name AS '病患姓名',
                pc.doctor_name AS '醫師姓名',
                pc.audiogram AS '聽力圖位置',
                pc.created_at AS '建立時間',
                pc.updated_at AS '更新時間'
            FROM patient_case AS pc
            JOIN patient AS p ON p.id = pc.patient_id
            ORDER BY pc.updated_at DESC, pc.id DESC
            """,
            conn,
        )
    status = (
        f"資料庫位置：{Path(db_path).resolve()}\n\n"
        f"病患：{len(patients)} 筆｜病例：{len(cases)} 筆"
    )
    return patients, cases, status


def clear_tables(db_path=DB_PATH):
    init_db(db_path)
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM patient_case")
        conn.execute("DELETE FROM patient")
        conn.execute("DELETE FROM sqlite_sequence WHERE name='patient_case'")
        conn.execute("DELETE FROM sqlite_sequence WHERE name='patient'")


if __name__ == "__main__":
    init_db()
    print(f"Database is ready: {DB_PATH}")
