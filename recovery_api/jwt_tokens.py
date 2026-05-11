from datetime import datetime, timedelta, timezone
from uuid import UUID

import jwt

from recovery_api.config import get_settings

JWT_AUDIENCE = "recovery-api"
ACCESS_TTL = timedelta(days=7)


def create_access_token(*, user_id: UUID, email: str) -> str:
    settings = get_settings()
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "email": email,
        "aud": JWT_AUDIENCE,
        "iat": now,
        "exp": now + ACCESS_TTL,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")
