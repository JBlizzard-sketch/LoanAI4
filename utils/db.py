import sqlite3, os, time, json
from pathlib import Path

DB_PATH = os.environ.get("LOANIQ_DB", "data/loaniq.sqlite")
Path("data").mkdir(exist_ok=True)

SCHEMA = {
    "users": """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password_hash TEXT,
            role TEXT CHECK(role in ('client','admin')) NOT NULL DEFAULT 'client',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """,
    "datasets": """
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            owner TEXT,
            name TEXT,
            meta TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """,
    "models": """
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            family TEXT,
            version INTEGER,
            metrics TEXT,
            path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            deployed INTEGER DEFAULT 0
        );
    """,
    "audit": """
        CREATE TABLE IF NOT EXISTS audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            action TEXT,
            detail TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """
}

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn

def init():
    conn = get_conn()
    cur = conn.cursor()
    for ddl in SCHEMA.values():
        cur.execute(ddl)
    conn.commit()
    conn.close()

def record_audit(username, action, detail=""):
    conn = get_conn(); cur = conn.cursor()
    cur.execute("INSERT INTO audit(username, action, detail) VALUES(?,?,?)", (username, action, detail))
    conn.commit(); conn.close()

def upsert_user(username, password_hash, role="client"):
    conn = get_conn(); cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO users(username,password_hash,role) VALUES(?,?,?)",
                (username, password_hash, role))
    conn.commit(); conn.close()

def get_user(username):
    conn = get_conn(); cur = conn.cursor()
    cur.execute("SELECT id, username, password_hash, role FROM users WHERE username=?", (username,))
    row = cur.fetchone()
    conn.close()
    return row

def list_models():
    conn = get_conn(); cur = conn.cursor()
    cur.execute("SELECT family, version, metrics, path, deployed, created_at FROM models ORDER BY family, version DESC")
    rows = cur.fetchall(); conn.close()
    return rows

def insert_model(family, version, metrics_dict, path, deployed=0):
    conn = get_conn(); cur = conn.cursor()
    cur.execute("INSERT INTO models(family, version, metrics, path, deployed) VALUES(?,?,?,?,?)",
                (family, version, json.dumps(metrics_dict), path, deployed))
    conn.commit(); conn.close()

def mark_deployed(family, version):
    conn = get_conn(); cur = conn.cursor()
    cur.execute("UPDATE models SET deployed=0 WHERE family=?", (family,))
    cur.execute("UPDATE models SET deployed=1 WHERE family=? AND version=?", (family, version))
    conn.commit(); conn.close()