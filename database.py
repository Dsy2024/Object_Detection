import sqlite3

def init_db(db_path="patient_info.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS detected_patient_info (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_filename TEXT NOT NULL,
            medical_record_number TEXT,
            patient_name TEXT,
            mrn_confidence REAL,
            name_confidence REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()

def insert_detected_info(db_path, image_filename, mrn, patient_name, mrn_conf=None, name_conf=None):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO detected_patient_info (
            image_filename,
            medical_record_number,
            patient_name,
            mrn_confidence,
            name_confidence
        )
        VALUES (?, ?, ?, ?, ?)
    """, (image_filename, mrn, patient_name, mrn_conf, name_conf))

    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()