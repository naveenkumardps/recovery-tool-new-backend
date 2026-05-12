import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from recovery_api.supabase_http import (
    SbAnonDep,
    SbServiceDep,
    auth_password_grant,
    auth_refresh_session,
    auth_signup,
    ensure_default_subscription,
)

log = logging.getLogger("recovery_api.auth")

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginBody(BaseModel):
    email: str = Field(..., min_length=3)
    password: str = Field(..., min_length=8)


class RegisterBody(BaseModel):
    email: str = Field(..., min_length=3)
    password: str = Field(..., min_length=8)
    display_name: str | None = Field(default=None, max_length=200)


class TokenOut(BaseModel):
    access_token: str
    refresh_token: str | None = None
    token_type: str = "bearer"


def _normalize_email(email: str) -> str:
    return email.strip().lower()


@router.post("/register", response_model=TokenOut)
async def register(body: RegisterBody, sb: SbAnonDep, sb_service: SbServiceDep):
    email = _normalize_email(body.email)
    if not email:
        raise HTTPException(status_code=400, detail="invalid_email")
    label = (body.display_name or "").strip()
    display_name = label or (email.split("@")[0] if "@" in email else email)
    out = await auth_signup(sb, email=email, password=body.password, display_name=display_name)
    token = out.get("access_token")
    if not token:
        # When email-confirmation is enabled, signup can succeed without an
        # immediate session. Surface a clear error so the client can prompt.
        raise HTTPException(status_code=400, detail="signup_requires_confirmation")

    # Attach the user to the Free plan by default. Best-effort: any failure here
    # must not break signup itself — the GET /api/me/subscriptions backstop will
    # heal the account on first dashboard load.
    user_payload = out.get("user") or {}
    user_id_raw = user_payload.get("id") if isinstance(user_payload, dict) else None
    if user_id_raw:
        try:
            await ensure_default_subscription(sb_service, UUID(str(user_id_raw)))
        except Exception:  # noqa: BLE001
            log.exception("register: failed to attach default free plan user=%s", user_id_raw)

    return TokenOut(access_token=token, refresh_token=out.get("refresh_token"))


@router.post("/login", response_model=TokenOut)
async def login(body: LoginBody, sb: SbAnonDep):
    email = _normalize_email(body.email)
    out = await auth_password_grant(sb, email=email, password=body.password)
    token = out.get("access_token")
    if not token:
        raise HTTPException(status_code=401, detail="invalid_credentials")
    return TokenOut(access_token=token, refresh_token=out.get("refresh_token"))


class RefreshBody(BaseModel):
    refresh_token: str = Field(..., min_length=10)


@router.post("/refresh", response_model=TokenOut)
async def refresh(body: RefreshBody, sb: SbAnonDep):
    """
    Exchange a Supabase refresh token for a new access token.
    This is used by the desktop app to keep sessions alive during long scans.
    """
    out = await auth_refresh_session(sb, refresh_token=body.refresh_token)
    token = out.get("access_token")
    if not token:
        raise HTTPException(status_code=401, detail="invalid_refresh_token")
    return TokenOut(access_token=token, refresh_token=out.get("refresh_token"))
