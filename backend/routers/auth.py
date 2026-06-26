"""
Google OAuth flow + JWT session cookie.
"""
from datetime import datetime, timedelta

from authlib.integrations.starlette_client import OAuth
from fastapi import APIRouter, Request, Response, HTTPException
from jose import jwt
from starlette.responses import RedirectResponse

from ..config import settings
from ..models.db import User

router = APIRouter(prefix="/auth", tags=["auth"])

oauth = OAuth()
oauth.register(
    name="google",
    client_id=settings.google_client_id,
    client_secret=settings.google_client_secret,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)


def create_jwt(user_id: str) -> str:
    expire = datetime.utcnow() + timedelta(hours=settings.jwt_expire_hours)
    return jwt.encode(
        {"sub": user_id, "exp": expire},
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )


def decode_jwt(token: str) -> dict:
    return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])


@router.get("/login")
async def login(request: Request):
    redirect_uri = str(request.url_for("auth_callback"))
    return await oauth.google.authorize_redirect(request, redirect_uri)


@router.get("/callback", name="auth_callback")
async def callback(request: Request):
    token = await oauth.google.authorize_access_token(request)
    userinfo = token.get("userinfo")
    if not userinfo:
        raise HTTPException(status_code=400, detail="OAuth failed")

    # Upsert user in DB
    from ..models.db import User
    from sqlalchemy.ext.asyncio import AsyncSession
    from ..main import get_db

    async with get_db() as db:
        user = await db.get(User, userinfo["sub"])
        if not user:
            user = User(
                id=userinfo["sub"],
                email=userinfo["email"],
                display_name=userinfo.get("name"),
            )
            db.add(user)
            await db.commit()

    jwt_token = create_jwt(userinfo["sub"])
    response = RedirectResponse(url=settings.frontend_url)
    response.set_cookie(
        "auth_token", jwt_token,
        httponly=True, samesite="lax",
        max_age=settings.jwt_expire_hours * 3600,
        secure=not settings.debug,
    )
    return response


@router.post("/logout")
async def logout(response: Response):
    response.delete_cookie("auth_token")
    return {"ok": True}
