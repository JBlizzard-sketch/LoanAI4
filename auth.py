import hashlib
from utils import db

def hash_pw(p: str) -> str:
    return hashlib.sha256(("pepper::" + p).encode()).hexdigest()

def ensure_admin():
    db.init()
    admin_user = db.get_user("admin")
    if not admin_user:
        db.upsert_user("admin", hash_pw("Shady868"), role="admin")

def register(username: str, password: str):
    if not username or not password:
        return False, "Username and password required", None
    if db.get_user(username):
        return False, "User already exists", None
    db.upsert_user(username, hash_pw(password), role="client")
    db.record_audit(username, "register", "new client")
    # Auto-login after successful registration
    user_row = db.get_user(username)
    if user_row:
        uid, uname, pwhash, role = user_row
        user_data = {"id": uid, "username": uname, "role": role}
        return True, "Registered and logged in", user_data
    return True, "Registered", None

def login(username: str, password: str):
    row = db.get_user(username)
    if not row:
        return False, "User not found", None
    uid, uname, pwhash, role = row
    if pwhash == hash_pw(password):
        db.record_audit(username, "login", role)
        return True, "OK", {"id": uid, "username": uname, "role": role}
    return False, "Invalid credentials", None