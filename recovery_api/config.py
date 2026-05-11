from functools import lru_cache
from pathlib import Path
from typing import Any
import logging
from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

log = logging.getLogger("recovery_api.config")


def _env_files() -> tuple[str, ...]:
    """
    Load `.env` reliably even if uvicorn is started from another directory.
    We support both `backend/.env` and `backend/recovery_api/.env` (common in this repo).
    """
    here = Path(__file__).resolve()
    pkg_dir = here.parent  # backend/recovery_api
    backend_dir = pkg_dir.parent  # backend/

    candidates = [
        Path.cwd() / ".env",
        backend_dir / ".env",
        pkg_dir / ".env",
        backend_dir.parent / ".env",
    ]
    return tuple(str(p) for p in candidates if p.exists())


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_env_files() or ".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    # Supabase project configuration (used for Auth + PostgREST)
    supabase_url: str = Field(default="", validation_alias=AliasChoices("SUPABASE_URL", "supabase_url"))
    supabase_anon_key: str = Field(default="", validation_alias=AliasChoices("SUPABASE_ANON_KEY", "supabase_anon_key"))
    supabase_jwt_aud: str = Field(default="authenticated", validation_alias=AliasChoices("SUPABASE_JWT_AUD", "supabase_jwt_aud"))

    # Optional legacy DB URL (no longer required when using Supabase PostgREST)
    database_url: str = Field(default="postgresql://postgres:postgres@127.0.0.1:5432/postgres")
    # Used to verify Supabase-issued access tokens (HS256).
    # Prefer the Supabase project secret; keep JWT_SECRET as a legacy alias.
    jwt_secret: str = Field(
        default="replace-with-supabase-jwt-secret",
        validation_alias=AliasChoices("SUPABASE_JWT_SECRET", "JWT_SECRET"),
    )
    require_subscription: bool = Field(False, validation_alias="REQUIRE_SUBSCRIPTION")
    # Supabase service role key (server-to-server). Required for webhooks/background tasks
    # that must bypass RLS (e.g. Razorpay webhook updating subscription status).
    supabase_service_role_key: str = Field(
        default="",
        validation_alias=AliasChoices("SUPABASE_SERVICE_ROLE_KEY", "supabase_service_role_key"),
    )

    # Razorpay integration (recurring subscriptions)
    razorpay_key_id: str = Field(
        default="",
        validation_alias=AliasChoices("RAZORPAY_KEY_ID", "razorpay_key_id"),
    )
    razorpay_key_secret: str = Field(
        default="",
        validation_alias=AliasChoices("RAZORPAY_KEY_SECRET", "razorpay_key_secret"),
    )
    razorpay_webhook_secret: str = Field(
        default="",
        validation_alias=AliasChoices("RAZORPAY_WEBHOOK_SECRET", "razorpay_webhook_secret"),
    )
    public_app_url: str = Field(
        default="http://localhost:8080",
        validation_alias=AliasChoices("PUBLIC_APP_URL", "public_app_url"),
    )
    allowed_origins: str = (
        "http://127.0.0.1:5173,http://127.0.0.1:5174,"
        "http://localhost:5173,http://localhost:5174,"
        "http://127.0.0.1:8080,http://localhost:8080,"
        "http://127.0.0.1:3000,http://localhost:3000,"
        "https://tauri.localhost,http://tauri.localhost"
    )


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    try:
        # High-signal startup diagnostics (do not log secrets).
        envs = _env_files()
        log.warning(
            "settings_loaded env_files=%s has_service_role_key=%s has_webhook_secret=%s",
            list(envs) if envs else [],
            bool((s.supabase_service_role_key or "").strip()),
            bool((s.razorpay_webhook_secret or "").strip()),
        )
    except Exception:
        # Never fail app startup due to logging.
        pass
    return s


@field_validator("jwt_secret", mode="before")
@classmethod
def _strip_wrapping_quotes(cls, v: Any) -> Any:
    if isinstance(v, str):
        s = v.strip()
        if (len(s) >= 2) and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
            return s[1:-1]
    return v


def cors_origins_list(settings: Settings) -> list[str]:
    return [o.strip() for o in settings.allowed_origins.split(",") if o.strip()]
