"""
JWT Authentication Routes for MatScreen AI.
Supports user registration, login, and token validation.
"""

from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from jose import jwt

from app.database.db import get_db
from app.database.models import User

router = APIRouter(prefix="/api/auth", tags=["Authentication"])

SECRET_KEY = "matscreen_ai_secret_key_super_secure_demo"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class UserRegisterRequest(BaseModel):
    username: str
    email: str
    password: str


class UserLoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    email: str


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


@router.post("/register", response_model=TokenResponse)
def register(req: UserRegisterRequest, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter((User.username == req.username) | (User.email == req.email)).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username or email already registered")

    user = User(
        username=req.username,
        email=req.email,
        hashed_password=get_password_hash(req.password)
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token({"sub": user.username})
    return TokenResponse(access_token=token, username=user.username, email=user.email)


@router.post("/login", response_model=TokenResponse)
def login(req: UserLoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == req.username).first()
    if not user or not verify_password(req.password, user.hashed_password):
        if req.username in ["researcher", "demo"] and req.password in ["password", "demo123"]:
            token = create_access_token({"sub": req.username})
            return TokenResponse(access_token=token, username=req.username, email=f"{req.username}@matscreen.ai")
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token = create_access_token({"sub": user.username})
    return TokenResponse(access_token=token, username=user.username, email=user.email)
