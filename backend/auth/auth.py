"""
Minimal JWT auth (optional feature). Uses a local SQLite file so there's
no extra service to run. Each authenticated user gets their own
collection-name prefix, isolating their indexed repos in the vector DB.
"""
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

SECRET_KEY = os.environ.get("JWT_SECRET", "change-me-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 1 day

DB_PATH = os.environ.get("AUTH_DB_PATH", "/data/users.db")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)


def _get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            hashed_password TEXT NOT NULL
        )
    """)
    return conn


def create_user(username: str, password: str) -> None:
    conn = _get_conn()
    existing = conn.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone()
    if existing:
        raise ValueError("Username already exists")
    hashed = pwd_context.hash(password)
    conn.execute("INSERT INTO users (username, hashed_password) VALUES (?, ?)", (username, hashed))
    conn.commit()
    conn.close()


def authenticate_user(username: str, password: str) -> bool:
    conn = _get_conn()
    row = conn.execute("SELECT hashed_password FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    if not row:
        return False
    return pwd_context.verify(password, row[0])


def create_access_token(username: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": username, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: Optional[str] = Depends(oauth2_scheme)) -> str:
    """
    Returns the username if a valid token is provided.
    If AUTH_REQUIRED=false (default for easy local demos), missing tokens
    fall back to a shared 'guest' namespace instead of raising 401.
    """
    auth_required = os.environ.get("AUTH_REQUIRED", "false").lower() == "true"

    if token is None:
        if auth_required:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
        return "guest"

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
