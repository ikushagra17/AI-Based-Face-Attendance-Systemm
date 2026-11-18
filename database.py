"""
database.py
------------
Handles all SQLite database operations for the AI-based Face Attendance System:
 - Student registration records
 - Daily attendance logs
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "attendance.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Create tables if they don't already exist."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS students (
            student_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            enrollment_no TEXT UNIQUE NOT NULL,
            name          TEXT NOT NULL,
            branch        TEXT,
            created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            attendance_id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id    INTEGER NOT NULL,
            date          TEXT NOT NULL,
            time          TEXT NOT NULL,
            status        TEXT DEFAULT 'Present',
            FOREIGN KEY (student_id) REFERENCES students(student_id),
            UNIQUE(student_id, date)
        )
    """)

    conn.commit()
    conn.close()


def add_student(enrollment_no, name, branch=""):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO students (enrollment_no, name, branch) VALUES (?, ?, ?)",
        (enrollment_no, name, branch),
    )
    conn.commit()
    student_id = cur.lastrowid
    conn.close()
    return student_id


def get_student_by_enrollment(enrollment_no):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM students WHERE enrollment_no = ?", (enrollment_no,))
    row = cur.fetchone()
    conn.close()
    return row


def get_all_students():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM students ORDER BY name")
    rows = cur.fetchall()
    conn.close()
    return rows


def mark_attendance(student_id, date_str, time_str):
    """Insert an attendance row; ignore if already marked for that date."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO attendance (student_id, date, time, status) VALUES (?, ?, ?, 'Present')",
            (student_id, date_str, time_str),
        )
        conn.commit()
        marked = True
    except sqlite3.IntegrityError:
        # Already marked present today
        marked = False
    conn.close()
    return marked


def get_attendance_for_date(date_str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT s.enrollment_no, s.name, a.time, a.status
        FROM attendance a
        JOIN students s ON s.student_id = a.student_id
        WHERE a.date = ?
        ORDER BY a.time
    """, (date_str,))
    rows = cur.fetchall()
    conn.close()
    return rows


def get_attendance_summary(enrollment_no):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT a.date, a.time, a.status
        FROM attendance a
        JOIN students s ON s.student_id = a.student_id
        WHERE s.enrollment_no = ?
        ORDER BY a.date DESC
    """, (enrollment_no,))
    rows = cur.fetchall()
    conn.close()
    return rows
