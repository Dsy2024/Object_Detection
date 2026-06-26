import sqlite3


DB_PATH = "db.db"


def init_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS patient (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT not NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS patient_case (
            serial_number TEXT PRIMARY KEY,
            patient_id INTEGER,
            doctor_name TEXT not NULL,
            audiogram TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (patient_id) REFERENCES patient(id)
        )
    """)

    conn.commit()
    conn.close()


def upsert_record(serial_number, patient_name, doctor_name, audiogram=None, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # find patient by name
    cur.execute("""
        SELECT id FROM patient WHERE patient_name = ?
    """, (patient_name,))
    row = cur.fetchone()
    patient_id = 0

    if row:
        patient_id = row[0]
    else:
        # insert new patient
        cur.execute("""
            INSERT INTO patient (patient_name)
            VALUES (?)
        """, (patient_name,))
        patient_id = cur.lastrowid

    # insert patient case
    cur.execute("""
        INSERT INTO patient_case (
            serial_number,
            patient_id,
            doctor_name,
            audiogram
        )
        VALUES (?, ?, ?, ?)
        ON CONFLICT(serial_number) DO UPDATE SET
            patient_id = excluded.patient_id,
            doctor_name = excluded.doctor_name,
            audiogram = excluded.audiogram,
            updated_at = CURRENT_TIMESTAMP;
    """, (serial_number, patient_id, doctor_name, audiogram))

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


if __name__ == "__main__":
    init_db()