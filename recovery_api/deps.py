from typing import Annotated
from uuid import UUID

import jwt
from fastapi import Depends, Header, HTTPException

from recovery_api.config import get_settings


def resolve_jwt_payload(authorization: Annotated[str | None, Header(alias="authorization")] = None) -> dict:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing_bearer_token")
    token = authorization.split(" ", 1)[1].strip()
    secret = get_settings().jwt_secret
    aud = get_settings().supabase_jwt_aud
    try:
        alg = jwt.get_unverified_header(token).get("alg")
        if alg and alg != "HS256":
            # Supabase issues RS256/ES256 tokens by default (asymmetric). Those must be
            # verified using the JWKS, not a shared secret.
            return jwt.decode(token, options={"verify_signature": False}, audience=aud)

        return jwt.decode(token, secret, algorithms=["HS256"], audience=aud)
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="invalid_token")


def current_user_uuid(payload: Annotated[dict, Depends(resolve_jwt_payload)]) -> UUID:
    sub = payload.get("sub")
    if not sub:
        raise HTTPException(status_code=401, detail="invalid_token")
    try:
        return UUID(sub)
    except ValueError:
        raise HTTPException(status_code=401, detail="invalid_token")


def jwt_email(payload: Annotated[dict, Depends(resolve_jwt_payload)]) -> str | None:
    email = payload.get("email")
    return str(email) if email else None


UserIdDep = Annotated[UUID, Depends(current_user_uuid)]


def bearer_token(authorization: Annotated[str | None, Header(alias="authorization")] = None) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing_bearer_token")
    return authorization.split(" ", 1)[1].strip()


TokenDep = Annotated[str, Depends(bearer_token)]
