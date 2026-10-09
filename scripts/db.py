import json
import sqlite3
import pandas as pd
from pathlib import Path


DB_PATH = "db.db"


def init_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS patient (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT not NULL,
            gender TEXT,
            birthdate TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS patient_case (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            serial_number TEXT UNIQUE,
            doctor_name TEXT,
            test_date TEXT,
            audiogram TEXT,
            audiogram_df TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (patient_id) REFERENCES patient(id)
        )
    """)

    ensure_column(cur, "patient", "gender", "TEXT")
    ensure_column(cur, "patient", "birthdate", "TEXT")
    ensure_column(cur, "patient_case", "test_date", "TEXT")
    ensure_column(cur, "patient_case", "audiogram_df", "TEXT")

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_patient_identity
        ON patient(patient_name, birthdate, gender)
    """)

    conn.commit()
    conn.close()


def ensure_column(cur, table_name, column_name, column_type):
    cur.execute(f"PRAGMA table_info({table_name})")
    columns = {row[1] for row in cur.fetchall()}
    if column_name not in columns:
        cur.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")


def serialize_dataframe(df):
    if df is None:
        return None

    if hasattr(df, "to_json"):
        return df.to_json(orient="split", force_ascii=False)

    return json.dumps(df, ensure_ascii=False, default=str)


def upsert_record(
    serial_number,
    patient_name,
    doctor_name,
    gender=None,
    birthdate=None,
    test_date=None,
    audiogram=None,
    audiogram_df=None,
    db_path=DB_PATH,
):
    init_db(db_path)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # A patient is the same person only when all identity fields match.
    cur.execute("""
        SELECT id FROM patient
        WHERE patient_name = ?
          AND birthdate IS ?
          AND gender IS ?
    """, (patient_name, birthdate, gender))
    row = cur.fetchone()
    patient_id = 0

    if row:
        patient_id = row[0]
    else:
        # insert new patient
        cur.execute("""
            INSERT INTO patient (patient_name, gender, birthdate)
            VALUES (?, ?, ?)
        """, (patient_name, gender, birthdate))
        patient_id = cur.lastrowid
    
    cur.execute("""
        UPDATE patient
        SET
            gender = COALESCE(?, gender),
            birthdate = COALESCE(?, birthdate)
        WHERE id = ?
    """, (gender, birthdate, patient_id))

    audiogram = str(audiogram) if audiogram is not None else None
    audiogram_df = serialize_dataframe(audiogram_df)
    # insert patient case
    cur.execute("""
        INSERT INTO patient_case (
            patient_id,
            serial_number,
            doctor_name,
            test_date,
            audiogram,
            audiogram_df
        )
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(serial_number) DO UPDATE SET
            patient_id = excluded.patient_id,
            doctor_name = excluded.doctor_name,
            test_date = COALESCE(excluded.test_date, patient_case.test_date),
            audiogram = excluded.audiogram,
            audiogram_df = COALESCE(excluded.audiogram_df, patient_case.audiogram_df),
            updated_at = CURRENT_TIMESTAMP;
    """, (patient_id, serial_number, doctor_name, test_date, audiogram, audiogram_df))

    conn.commit()
    conn.close()


def clear_tables(db_path="db.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("DELETE FROM patient_case")
    cur.execute("DELETE FROM patient")

    # reset AUTOINCREMENT id
    cur.execute("DELETE FROM sqlite_sequence WHERE name='patient_case'")
    cur.execute("DELETE FROM sqlite_sequence WHERE name='patient'")

    conn.commit()
    conn.close()


def search_patient_cases(
    patient_name="",
    gender="",
    birthdate="",
    db_path=DB_PATH,
):
    conn = sqlite3.connect(db_path)

    query = """
        SELECT
            pc.id AS case_id,
            p.id AS patient_id,
            p.patient_name,
            p.gender,
            p.birthdate,
            pc.serial_number,
            pc.doctor_name,
            pc.test_date
        FROM patient p
        LEFT JOIN patient_case pc
            ON p.id = pc.patient_id
        WHERE 1=1
    """

    params = []

    patient_name = patient_name.strip()
    gender = gender.strip() if gender else ""
    birthdate = birthdate.strip()

    if patient_name:
        query += " AND p.patient_name LIKE ?"
        params.append(f"%{patient_name}%")

    if gender:
        query += " AND p.gender = ?"
        params.append(gender)

    if birthdate:
        query += " AND p.birthdate = ?"
        params.append(birthdate)

    query += """
        ORDER BY
            p.patient_name,
            pc.test_date DESC
    """

    df = pd.read_sql_query(
        query,
        conn,
        params=params,
    )

    conn.close()

    return df


def load_audiogram_df(case_id, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT audiogram_df
        FROM patient_case
        WHERE id = ?
        """,
        (int(case_id),),
    )

    row = cur.fetchone()
    conn.close()

    if not row or not row[0]:
        return pd.DataFrame()

    obj = json.loads(row[0])

    df = pd.DataFrame(
        obj["data"],
        columns=obj["columns"],
        index=obj["index"],
    )

    return df


def load_audiogram_path(case_id, db_path=DB_PATH):
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT audiogram
            FROM patient_case
            WHERE id = ?
            """,
            (int(case_id),),
        ).fetchone()

    if not row or not row[0]:
        return None

    path = Path(row[0])

    if not path.is_file():
        return None

    return str(path.resolve())


if __name__ == "__main__":
    init_db()
    clear_tables()
