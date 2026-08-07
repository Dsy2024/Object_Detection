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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS hearing_result (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_case_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                frequency_hz INTEGER NOT NULL,
                db_value REAL NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(patient_case_id, symbol, frequency_hz),
                FOREIGN KEY (patient_case_id) REFERENCES patient_case(id)
                    ON DELETE CASCADE
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


def save_hearing_results(serial_number, results_df, db_path=DB_PATH):
    """Replace one case's structured PTA results with the latest detection."""
    if results_df is None or results_df.empty or serial_number == "N/A":
        return 0

    init_db(db_path)
    with _connect(db_path) as conn:
        case = conn.execute(
            "SELECT id FROM patient_case WHERE serial_number = ?", (serial_number,)
        ).fetchone()
        if not case:
            return 0

        case_id = case[0]
        rows = []
        for _, result in results_df.iterrows():
            symbol = str(result.get("cls", "")).strip()
            if not symbol:
                continue
            for frequency, value in result.items():
                if frequency == "cls" or value == "" or pd.isna(value):
                    continue
                try:
                    rows.append((case_id, symbol, int(frequency), float(value)))
                except (TypeError, ValueError):
                    continue

        if not rows:
            return 0

        conn.execute("DELETE FROM hearing_result WHERE patient_case_id = ?", (case_id,))
        conn.executemany(
            """
            INSERT INTO hearing_result (
                patient_case_id, symbol, frequency_hz, db_value
            ) VALUES (?, ?, ?, ?)
            """,
            rows,
        )
    return len(rows)


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
        hearing_results = pd.read_sql_query(
            """
            SELECT
                hr.id AS '結果 ID',
                pc.serial_number AS '病例序號',
                p.patient_name AS '病患姓名',
                hr.symbol AS '符號',
                hr.frequency_hz AS '頻率 (Hz)',
                hr.db_value AS '聽力閾值 (dB)',
                hr.created_at AS '建立時間'
            FROM hearing_result AS hr
            JOIN patient_case AS pc ON pc.id = hr.patient_case_id
            JOIN patient AS p ON p.id = pc.patient_id
            ORDER BY pc.id DESC, hr.symbol, hr.frequency_hz
            """,
            conn,
        )
    status = (
        f"資料庫位置：{Path(db_path).resolve()}\n\n"
        f"病患：{len(patients)} 筆｜病例：{len(cases)} 筆｜"
        f"聽力結果：{len(hearing_results)} 筆"
    )
    return patients, cases, hearing_results, status


def clear_tables(db_path=DB_PATH):
    init_db(db_path)
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM hearing_result")
        conn.execute("DELETE FROM patient_case")
        conn.execute("DELETE FROM patient")
        conn.execute("DELETE FROM sqlite_sequence WHERE name='hearing_result'")
        conn.execute("DELETE FROM sqlite_sequence WHERE name='patient_case'")
        conn.execute("DELETE FROM sqlite_sequence WHERE name='patient'")


if __name__ == "__main__":
    init_db()
    print(f"Database is ready: {DB_PATH}")
