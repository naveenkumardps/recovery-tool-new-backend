from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Annotated, Any
from uuid import UUID

from fastapi import Depends, HTTPException
from supabase import AsyncClient, acreate_client

from recovery_api.config import get_settings
from recovery_api.deps import TokenDep

log = logging.getLogger("recovery_api.supabase")


async def _build_client() -> AsyncClient:
    s = get_settings()
    if not s.supabase_url or not s.supabase_anon_key:
        raise HTTPException(status_code=500, detail="missing_supabase_config")
    return await acreate_client(s.supabase_url, s.supabase_anon_key)


async def _build_service_client() -> AsyncClient:
    """
    Service-role Supabase client for server-to-server operations.
    This bypasses RLS and must never be exposed to the browser.
    """
    s = get_settings()
    if not s.supabase_url or not s.supabase_service_role_key:
        raise HTTPException(status_code=500, detail="missing_supabase_service_role_config")
    return await acreate_client(s.supabase_url, s.supabase_service_role_key)


async def get_sb_anon() -> AsyncClient:
    """
    Anon-keyed Supabase client for unauthenticated flows (login / signup).
    """
    return await _build_client()


async def get_sb_user(token: TokenDep) -> AsyncClient:
    """
    Supabase client whose PostgREST requests carry the caller's JWT,
    so RLS policies are evaluated as that user.
    """
    client = await _build_client()
    client.postgrest.auth(token)
    return client


async def get_sb_service() -> AsyncClient:
    return await _build_service_client()


SbAnonDep = Annotated[AsyncClient, Depends(get_sb_anon)]
SbDep = Annotated[AsyncClient, Depends(get_sb_user)]
SbServiceDep = Annotated[AsyncClient, Depends(get_sb_service)]


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------


async def auth_password_grant(
    sb: AsyncClient, email: str, password: str
) -> dict[str, Any]:
    try:
        res = await sb.auth.sign_in_with_password({"email": email, "password": password})
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=401, detail="invalid_credentials") from exc
    if not res or not res.session or not res.session.access_token:
        raise HTTPException(status_code=401, detail="invalid_credentials")
    return {
        "access_token": res.session.access_token,
        "refresh_token": res.session.refresh_token,
        "user": res.user.model_dump() if res.user else None,
    }


async def auth_signup(
    sb: AsyncClient, email: str, password: str, display_name: str | None
) -> dict[str, Any]:
    payload: dict[str, Any] = {"email": email, "password": password}
    if display_name:
        payload["options"] = {"data": {"name": display_name}}
    try:
        res = await sb.auth.sign_up(payload)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="signup_failed") from exc
    token: str | None = None
    refresh_token: str | None = None
    if res and res.session:
        token = res.session.access_token
        refresh_token = res.session.refresh_token
    return {
        "access_token": token,
        "refresh_token": refresh_token,
        "user": res.user.model_dump() if res and res.user else None,
    }


async def auth_refresh_session(sb: AsyncClient, refresh_token: str) -> dict[str, Any]:
    try:
        res = await sb.auth.refresh_session(refresh_token)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=401, detail="invalid_refresh_token") from exc
    if not res or not res.session or not res.session.access_token:
        raise HTTPException(status_code=401, detail="invalid_refresh_token")
    return {
        "access_token": res.session.access_token,
        "refresh_token": res.session.refresh_token,
        "user": res.user.model_dump() if res.user else None,
    }


# ---------------------------------------------------------------------------
# PostgREST helpers (thin wrappers over supabase-py's table() builder)
# ---------------------------------------------------------------------------


def _unwrap(res: Any) -> list[dict[str, Any]]:
    data = getattr(res, "data", None)
    if data is None:
        return []
    return list(data)


async def count_rows(
    sb: AsyncClient,
    table: str,
    *,
    eq: dict[str, Any] | None = None,
    gte: dict[str, Any] | None = None,
    lte: dict[str, Any] | None = None,
) -> int:
    q = sb.table(table).select("id", count="exact", head=True)
    for k, v in (eq or {}).items():
        q = q.eq(k, v)
    for k, v in (gte or {}).items():
        q = q.gte(k, v)
    for k, v in (lte or {}).items():
        q = q.lte(k, v)
    try:
        res = await q.execute()
    except Exception as exc:  # noqa: BLE001
        log.exception("supabase_count_failed table=%s", table)
        raise HTTPException(status_code=502, detail=f"supabase_count_failed: {exc}") from exc
    return int(getattr(res, "count", 0) or 0)


async def safe_execute(query: Any, *, error_detail: str = "supabase_query_failed") -> list[dict[str, Any]]:
    try:
        res = await query.execute()
    except Exception as exc:  # noqa: BLE001
        log.exception("supabase_query_failed detail=%s", error_detail)
        raise HTTPException(status_code=502, detail=f"{error_detail}: {exc}") from exc
    return _unwrap(res)


async def ensure_default_subscription(sb: AsyncClient, user_id: UUID) -> str | None:
    """
    Make sure every user has an active subscription by attaching them to the
    cheapest free (price 0) active plan when they have none.

    Returns the plan_id of the resulting (existing or just-created) active
    subscription, or None when no free plan exists in `subscription_plans`.

    Designed to be idempotent and safe to call lazily from any authenticated
    endpoint — it inserts only when there is no active subscription on file.
    """
    existing = await safe_execute(
        sb.table("subscriptions")
        .select("id,plan_id")
        .eq("user_id", str(user_id))
        .eq("status", "active")
        .limit(1)
    )
    if existing:
        pid = existing[0].get("plan_id")
        return str(pid) if pid is not None else None

    free_plans = await safe_execute(
        sb.table("subscription_plans")
        .select("id,price_inr_paise")
        .eq("active", True)
        .eq("price_inr_paise", 0)
        .order("created_at")
        .limit(1)
    )
    if not free_plans:
        return None

    plan_id = str(free_plans[0].get("id") or "").strip()
    if not plan_id:
        return None

    now = datetime.now(timezone.utc)
    row: dict[str, Any] = {
        "user_id": str(user_id),
        "plan_id": plan_id,
        "status": "active",
        "current_period_start": now.isoformat(),
        "cancel_at_period_end": False,
    }
    try:
        await safe_execute(
            sb.table("subscriptions").insert(row),
            error_detail="supabase_insert_failed",
        )
    except HTTPException:
        # Race condition (another request just inserted one) is fine — confirm and
        # return whatever's now active.
        again = await safe_execute(
            sb.table("subscriptions")
            .select("plan_id")
            .eq("user_id", str(user_id))
            .eq("status", "active")
            .limit(1)
        )
        if again:
            pid = again[0].get("plan_id")
            return str(pid) if pid is not None else None
        raise
    return plan_id
