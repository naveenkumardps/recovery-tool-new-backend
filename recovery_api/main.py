from datetime import datetime, timedelta, timezone
from typing import Annotated, Any
from uuid import UUID

import hashlib
import hmac
import json
import logging

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from recovery_api.config import cors_origins_list, get_settings
from recovery_api.deps import UserIdDep, jwt_email
from recovery_api.razorpay_client import (
    create_order as rz_create_order,
    create_plan as rz_create_plan,
    create_subscription as rz_create_subscription,
    fetch_order as rz_fetch_order,
    fetch_payment as rz_fetch_payment,
    verify_order_payment_signature as rz_verify_order_payment_signature,
)
from recovery_api.routers.auth import router as auth_router
from recovery_api.supabase_http import SbDep, SbServiceDep, count_rows, safe_execute

settings = get_settings()
log = logging.getLogger("recovery_api")

app = FastAPI(title="Recovery API", version="0.4.0")

_cors_regex = (settings.cors_origin_regex or "").strip()
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins_list(settings),
    allow_origin_regex=_cors_regex or None,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)


# ---------------------------------------------------------------------------
# Small helpers (Supabase only — no SQLAlchemy anywhere)
# ---------------------------------------------------------------------------


async def _is_admin(sb: Any, user_id: UUID) -> bool:
    rows = await safe_execute(
        sb.table("admin_users").select("user_id").eq("user_id", str(user_id)).limit(1)
    )
    return bool(rows)


async def _require_admin(sb: Any, user_id: UUID) -> None:
    if not await _is_admin(sb, user_id):
        raise HTTPException(status_code=403, detail="admin_only")


def _iso(dt: Any) -> str | None:
    if dt is None:
        return None
    if isinstance(dt, datetime):
        return dt.isoformat()
    return str(dt)


def _serialize_plan_row(r: dict[str, Any]) -> dict[str, Any]:
    feat = r.get("features")
    feat_obj: dict[str, Any] = feat if isinstance(feat, dict) else {}
    return {
        "id": r["id"],
        "name": r["name"],
        "price_inr_paise": int(r["price_inr_paise"]),
        "billing_period": r["billing_period"],
        "max_devices": int(r["max_devices"]),
        "monthly_recovery_quota_mb": int(r["monthly_recovery_quota_mb"]),
        "features": feat_obj,
        "active": bool(r["active"]),
        "created_at": _iso(r.get("created_at")),
    }


def _features_modules(feat: Any) -> dict[str, bool]:
    if not isinstance(feat, dict):
        return {}
    raw = feat.get("modules")
    if not isinstance(raw, dict):
        return {}
    return {str(k): bool(v) for k, v in raw.items()}


async def _active_subscription_plan_id(sb: Any, user_id: UUID) -> str | None:
    subs = await safe_execute(
        sb.table("subscriptions")
        .select("plan_id,created_at")
        .eq("user_id", str(user_id))
        .eq("status", "active")
        .order("created_at", desc=True)
        .limit(1)
    )
    if not subs:
        return None
    pid = subs[0].get("plan_id")
    return str(pid) if pid is not None else None


# ---------------------------------------------------------------------------
# Health / me
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True}


class MeOut(BaseModel):
    user_id: UUID
    email: str | None
    display_name: str | None
    is_admin: bool


@app.get("/api/me", response_model=MeOut)
async def get_me(
    sb: SbDep,
    user_id: UserIdDep,
    email_claim: Annotated[str | None, Depends(jwt_email)],
) -> MeOut:
    rows = await safe_execute(
        sb.table("users").select("email,display_name").eq("id", str(user_id)).limit(1)
    )
    row = rows[0] if rows else {}
    db_email = row.get("email")
    return MeOut(
        user_id=user_id,
        email=db_email if db_email else email_claim,
        display_name=row.get("display_name"),
        is_admin=await _is_admin(sb, user_id),
    )


# ---------------------------------------------------------------------------
# Subscription gate (active sub × plan quota × month-to-date usage)
# ---------------------------------------------------------------------------


@app.get("/api/me/subscription-gate")
async def subscription_gate(
    sb: SbDep,
    user_id: UserIdDep,
    module: Annotated[str | None, Query(description="module_catalog.id to also enforce")] = None,
) -> dict[str, Any]:
    """
    Combined gate for any user-initiated recovery action.

    - When `REQUIRE_SUBSCRIPTION` is false on the server, always allow (still echoes
      `module_enabled` so the desktop can reflect entitlement in the UI).
    - When a `module` is provided, it is enforced against the active plan's
      `features.modules` map (or the catalog id is rejected if unknown). Admins bypass.
    - Otherwise: active subscription + monthly quota are enforced as before.
    """
    admin = await _is_admin(sb, user_id)
    plan_id = await _active_subscription_plan_id(sb, user_id)

    # Resolve module entitlement up-front so it's reported even when subscriptions are off.
    module_enabled: bool | None = None
    if module:
        if admin:
            module_enabled = True
        elif plan_id:
            plans_feat = await safe_execute(
                sb.table("subscription_plans")
                .select("features")
                .eq("id", plan_id)
                .limit(1)
            )
            if plans_feat:
                module_enabled = _features_modules(plans_feat[0].get("features")).get(
                    module, False
                )
            else:
                module_enabled = False
        else:
            module_enabled = False

    if not settings.require_subscription and not module:
        return {
            "ok": True,
            "plan_id": plan_id,
            "quota_mb": None,
            "used_mb_this_month": 0,
            "module_enabled": module_enabled,
        }

    if module and module_enabled is False and not admin:
        return {
            "ok": False,
            "plan_id": plan_id,
            "module_enabled": False,
            "reason": (
                f"The '{module}' module is not enabled on your current plan. "
                "Ask an admin to enable it or upgrade your plan."
            ),
        }

    if settings.require_subscription and not admin:
        if not plan_id:
            return {
                "ok": False,
                "plan_id": None,
                "module_enabled": module_enabled,
                "reason": "No active subscription. Open the billing portal and activate a plan.",
            }

        plans = await safe_execute(
            sb.table("subscription_plans")
            .select("monthly_recovery_quota_mb")
            .eq("id", plan_id)
            .limit(1)
        )
        quota = int((plans[0]["monthly_recovery_quota_mb"] or 0) if plans else 0)

        now = datetime.now(timezone.utc)
        month_start = datetime(now.year, now.month, 1, tzinfo=timezone.utc).isoformat()
        logs = await safe_execute(
            sb.table("usage_logs")
            .select("quantity")
            .eq("user_id", str(user_id))
            .gte("created_at", month_start)
        )
        used = float(sum(float(l.get("quantity") or 0) for l in logs))

        if quota > 0 and used >= quota:
            return {
                "ok": False,
                "plan_id": plan_id,
                "module_enabled": module_enabled,
                "quota_mb": quota,
                "used_mb_this_month": used,
                "reason": (
                    f"Monthly recovery quota reached ({used:.1f} / {quota} MB). "
                    "Upgrade or wait for the next billing period."
                ),
            }

        return {
            "ok": True,
            "plan_id": plan_id,
            "quota_mb": quota,
            "used_mb_this_month": used,
            "module_enabled": module_enabled,
        }

    return {
        "ok": True,
        "plan_id": plan_id,
        "quota_mb": None,
        "used_mb_this_month": 0,
        "module_enabled": module_enabled,
    }


# ---------------------------------------------------------------------------
# Plans (public, active list + admin CRUD)
# ---------------------------------------------------------------------------


PLAN_COLS = (
    "id,name,price_inr_paise,billing_period,max_devices,"
    "monthly_recovery_quota_mb,features,active,created_at"
)


@app.get("/api/plans/active")
async def list_active_plans(sb: SbDep, user_id: UserIdDep):
    _ = user_id
    rows = await safe_execute(
        sb.table("subscription_plans")
        .select(PLAN_COLS)
        .eq("active", True)
        .order("price_inr_paise")
    )
    return [_serialize_plan_row(r) for r in rows]


@app.get("/api/admin/plans")
async def admin_list_plans(sb: SbDep, user_id: UserIdDep):
    await _require_admin(sb, user_id)
    rows = await safe_execute(
        sb.table("subscription_plans").select(PLAN_COLS).order("created_at", desc=True)
    )
    return [_serialize_plan_row(r) for r in rows]


class PlanUpsertBody(BaseModel):
    id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    price_inr_paise: int = Field(ge=0)
    billing_period: str = Field(pattern="^(monthly|yearly|lifetime)$")
    max_devices: int = Field(ge=1)
    monthly_recovery_quota_mb: int = Field(ge=0)
    features: dict[str, Any] = Field(default_factory=dict)
    active: bool = True


@app.put("/api/admin/plans", status_code=204)
async def admin_upsert_plan(body: PlanUpsertBody, sb: SbDep, user_id: UserIdDep):
    await _require_admin(sb, user_id)
    await safe_execute(
        sb.table("subscription_plans").upsert(body.model_dump(), on_conflict="id"),
        error_detail="supabase_upsert_failed",
    )

    # Auto-provision Razorpay Plan for monthly/yearly plans so the web portal can create subscriptions.
    # Stored in `subscription_plans.features.razorpay_plan_id`.
    if settings.razorpay_key_id and settings.razorpay_key_secret:
        try:
            log.info("admin_upsert_plan: ensuring razorpay_plan_id for plan=%s", body.id)
            rows = await safe_execute(
                sb.table("subscription_plans")
                .select("id,name,price_inr_paise,billing_period,features,active")
                .eq("id", body.id)
                .limit(1)
            )
            if rows:
                r = rows[0]
                if bool(r.get("active")) and r.get("billing_period") in ("monthly", "yearly"):
                    feat = r.get("features")
                    feat_obj: dict[str, Any] = dict(feat) if isinstance(feat, dict) else {}

                    existing_id = feat_obj.get("razorpay_plan_id")
                    existing_amount = feat_obj.get("razorpay_plan_amount_paise")
                    existing_period = feat_obj.get("razorpay_plan_period")

                    need_new = not (isinstance(existing_id, str) and existing_id.strip())
                    if (
                        not need_new
                        and existing_amount is not None
                        and int(existing_amount) != int(r.get("price_inr_paise") or 0)
                    ):
                        # Razorpay plans are effectively immutable for amount/period; create a new one.
                        need_new = True
                    if not need_new and existing_period and str(existing_period) != str(r.get("billing_period")):
                        need_new = True

                    if need_new:
                        period = "monthly" if r.get("billing_period") == "monthly" else "yearly"
                        interval = 1
                        amount = int(r.get("price_inr_paise") or 0)
                        log.info(
                            "razorpay_plan: creating plan for %s amount=%s period=%s",
                            body.id,
                            amount,
                            period,
                        )
                        rz_plan = await rz_create_plan(
                            {
                                "period": period,
                                "interval": interval,
                                "item": {
                                    "name": str(r.get("name") or r.get("id")),
                                    "amount": amount,
                                    "currency": "INR",
                                    "description": f"RecoverVault plan {r.get('id')}",
                                },
                                "notes": {"plan_id": str(r.get("id"))},
                            }
                        )
                        rz_plan_id = str(rz_plan.get("id") or "")
                        if rz_plan_id:
                            log.info("razorpay_plan: created plan_id=%s for plan=%s", rz_plan_id, body.id)
                            feat_obj["razorpay_plan_id"] = rz_plan_id
                            feat_obj["razorpay_plan_amount_paise"] = amount
                            feat_obj["razorpay_plan_period"] = period
                            feat_obj["razorpay_plan_interval"] = interval
                            await safe_execute(
                                sb.table("subscription_plans")
                                .update({"features": feat_obj})
                                .eq("id", body.id),
                                error_detail="supabase_update_failed",
                            )
        except Exception:
            # If Razorpay provisioning fails, keep the plan saved; admin can retry later.
            log.exception("razorpay_plan: provisioning failed for plan=%s", body.id)


@app.delete("/api/admin/plans/{plan_id}", status_code=204)
async def admin_delete_plan(plan_id: str, sb: SbDep, user_id: UserIdDep):
    await _require_admin(sb, user_id)
    try:
        await safe_execute(
            sb.table("subscription_plans").delete().eq("id", plan_id),
            error_detail="supabase_delete_failed",
        )
    except HTTPException as e:
        # Foreign-key violations bubble up as 502 from safe_execute.
        if e.status_code == 502:
            raise HTTPException(status_code=409, detail="plan_referenced_by_subscriptions") from None
        raise


# ---------------------------------------------------------------------------
# Subscriptions / licenses / usage
# ---------------------------------------------------------------------------


class CreateSubscriptionBody(BaseModel):
    plan_id: str = Field(..., min_length=1)


@app.post("/api/me/subscriptions", status_code=201)
async def create_subscription(
    body: CreateSubscriptionBody, sb: SbDep, user_id: UserIdDep
):
    plans = await safe_execute(
        sb.table("subscription_plans")
        .select("billing_period,active")
        .eq("id", body.plan_id)
        .eq("active", True)
        .limit(1)
    )
    if not plans:
        raise HTTPException(status_code=400, detail="unknown_or_inactive_plan")

    bp = plans[0].get("billing_period")
    now = datetime.now(timezone.utc)
    if bp == "monthly":
        end_iso: str | None = (now + timedelta(days=31)).isoformat()
    elif bp == "yearly":
        end_iso = (now + timedelta(days=365)).isoformat()
    else:
        end_iso = None

    row: dict[str, Any] = {
        "user_id": str(user_id),
        "plan_id": body.plan_id,
        "status": "active",
        "current_period_start": now.isoformat(),
        "cancel_at_period_end": False,
    }
    if end_iso:
        row["current_period_end"] = end_iso

    await safe_execute(
        sb.table("subscriptions").insert(row),
        error_detail="supabase_insert_failed",
    )
    return {"ok": True}


class CreateRazorpaySubscriptionBody(BaseModel):
    plan_id: str = Field(..., min_length=1)


class CreateRazorpaySubscriptionOut(BaseModel):
    ok: bool = True
    plan_id: str
    razorpay_subscription_id: str
    status: str


class CreateRazorpayCheckoutOut(BaseModel):
    ok: bool = True
    plan_id: str
    mode: str  # subscription|order|free
    razorpay_subscription_id: str | None = None
    razorpay_order_id: str | None = None
    amount_paise: int | None = None
    currency: str | None = None


@app.post("/api/me/razorpay/subscription", response_model=CreateRazorpayCheckoutOut)
async def create_razorpay_subscription(  # kept path for compatibility
    body: CreateRazorpaySubscriptionBody, sb: SbDep, user_id: UserIdDep
):
    # Require Razorpay config in server env.
    if not settings.razorpay_key_id or not settings.razorpay_key_secret:
        raise HTTPException(status_code=500, detail="razorpay_not_configured")

    plans = await safe_execute(
        sb.table("subscription_plans")
        .select("id,billing_period,active,features,price_inr_paise")
        .eq("id", body.plan_id)
        .eq("active", True)
        .limit(1)
    )
    if not plans:
        raise HTTPException(status_code=400, detail="unknown_or_inactive_plan")

    plan = plans[0]
    bp = str(plan.get("billing_period") or "")
    if bp not in ("monthly", "yearly"):
        raise HTTPException(status_code=400, detail="plan_not_subscribable")

    feat = plan.get("features")
    feat_obj: dict[str, Any] = feat if isinstance(feat, dict) else {}
    razorpay_plan_id = feat_obj.get("razorpay_plan_id")
    if not razorpay_plan_id or not isinstance(razorpay_plan_id, str):
        # Some existing rows may not have been provisioned via the admin upsert path.
        # Create a Razorpay plan on-demand and persist it into features so future
        # checkouts are stable.
        try:
            period = "monthly" if bp == "monthly" else "yearly"
            interval = 1
            amount = int(plan.get("price_inr_paise") or 0)
            rz_plan = await rz_create_plan(
                {
                    "period": period,
                    "interval": interval,
                    "item": {
                        "name": str(plan.get("id") or body.plan_id),
                        "amount": amount,
                        "currency": "INR",
                        "description": f"RecoverVault plan {body.plan_id}",
                    },
                    "notes": {"plan_id": body.plan_id},
                }
            )
            rz_plan_id = str(rz_plan.get("id") or "").strip()
            if not rz_plan_id:
                raise HTTPException(status_code=502, detail="razorpay_plan_create_failed")
            feat_obj["razorpay_plan_id"] = rz_plan_id
            feat_obj["razorpay_plan_amount_paise"] = amount
            feat_obj["razorpay_plan_period"] = period
            feat_obj["razorpay_plan_interval"] = interval
            await safe_execute(
                sb.table("subscription_plans")
                .update({"features": feat_obj})
                .eq("id", body.plan_id),
                error_detail="supabase_update_failed",
            )
            razorpay_plan_id = rz_plan_id
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("razorpay_plan: on-demand provisioning failed plan=%s", body.plan_id)
            raise HTTPException(status_code=502, detail="razorpay_plan_create_failed") from exc

    # Test-mode fallback: Razorpay subscriptions often fail in test accounts due to mandates.
    if settings.razorpay_key_id.startswith("rzp_test_"):
        amount = int(plan.get("price_inr_paise") or 0)
        if amount <= 0:
            # Free plan / 0-priced plan: activate locally without payment.
            now = datetime.now(timezone.utc)
            row: dict[str, Any] = {
                "user_id": str(user_id),
                "plan_id": body.plan_id,
                "status": "active",
                "current_period_start": now.isoformat(),
                "cancel_at_period_end": False,
            }
            await safe_execute(
                sb.table("subscriptions").insert(row),
                error_detail="supabase_insert_failed",
            )
            return CreateRazorpayCheckoutOut(plan_id=body.plan_id, mode="free", amount_paise=0, currency="INR")
        try:
            rz_order = await rz_create_order(
                {
                    "amount": amount,
                    "currency": "INR",
                    "receipt": f"rv_{body.plan_id}_{str(user_id)[:8]}_{int(datetime.now(timezone.utc).timestamp())}",
                    "payment_capture": 1,
                    "notes": {"user_id": str(user_id), "plan_id": body.plan_id},
                }
            )
        except Exception as exc:  # noqa: BLE001
            status = getattr(exc, "status_code", None)
            err = getattr(exc, "error", None)
            detail: dict[str, Any] = {"code": "razorpay_order_create_failed"}
            if status is not None:
                detail["razorpay_status_code"] = status
            if err is not None:
                detail["razorpay_error"] = err
            raise HTTPException(status_code=502, detail=detail) from exc
        order_id = str(rz_order.get("id") or "")
        if not order_id:
            raise HTTPException(status_code=502, detail="razorpay_order_create_failed")

        # Create a local pending subscription row so the UI can show "processing" immediately.
        # Schema (0001_init) has no `metadata` on subscriptions — do not insert unknown columns.
        # Webhook `payment.captured` / `order.paid` marks this row active via order `notes`.
        pending: dict[str, Any] = {
            "user_id": str(user_id),
            "plan_id": body.plan_id,
            "status": "past_due",
            "cancel_at_period_end": False,
            "razorpay_order_id": order_id,
        }
        await safe_execute(
            sb.table("subscriptions").insert(pending),
            error_detail="supabase_insert_failed",
        )

        return CreateRazorpayCheckoutOut(
            plan_id=body.plan_id,
            mode="order",
            razorpay_order_id=order_id,
            amount_paise=amount,
            currency="INR",
        )

    # Live-mode: create the Razorpay subscription.
    total_count = 12 if bp == "monthly" else 1
    payload = {
        "plan_id": razorpay_plan_id,
        "total_count": total_count,
        "customer_notify": 1,
        "notes": {"user_id": str(user_id), "plan_id": body.plan_id},
    }
    try:
        rz_sub = await rz_create_subscription(payload)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail="razorpay_create_failed") from exc

    rz_id = str(rz_sub.get("id") or "")
    if not rz_id:
        raise HTTPException(status_code=502, detail="razorpay_create_failed")

    # Store locally as pending until webhook activation.
    row: dict[str, Any] = {
        "user_id": str(user_id),
        "plan_id": body.plan_id,
        "status": "past_due",
        "cancel_at_period_end": False,
        "razorpay_subscription_id": rz_id,
    }
    await safe_execute(
        sb.table("subscriptions").insert(row),
        error_detail="supabase_insert_failed",
    )

    return CreateRazorpayCheckoutOut(
        plan_id=body.plan_id,
        mode="subscription",
        razorpay_subscription_id=rz_id,
        amount_paise=int(plan.get("price_inr_paise") or 0),
        currency="INR",
    )


async def _finalize_subscription_after_order_paid(
    sb: Any,
    *,
    user_id: UUID,
    plan_id: str,
    razorpay_order_id: str | None,
    razorpay_payment_id: str | None,
) -> None:
    plans = await safe_execute(
        sb.table("subscription_plans")
        .select("billing_period")
        .eq("id", plan_id)
        .limit(1),
        error_detail="supabase_query_failed",
    )
    if not plans:
        raise HTTPException(status_code=400, detail="unknown_plan")

    bp = str(plans[0].get("billing_period") or "")
    now = datetime.now(timezone.utc)
    end_iso: str | None
    if bp == "monthly":
        end_iso = (now + timedelta(days=31)).isoformat()
    elif bp == "yearly":
        end_iso = (now + timedelta(days=365)).isoformat()
    else:
        end_iso = None

    updates: dict[str, Any] = {
        "status": "active",
        "current_period_start": now.isoformat(),
        "cancel_at_period_end": False,
    }
    if end_iso:
        updates["current_period_end"] = end_iso
    if razorpay_payment_id:
        updates["razorpay_payment_id"] = razorpay_payment_id.strip()
    if razorpay_order_id:
        updates["razorpay_order_id"] = razorpay_order_id.strip()

    existing = await safe_execute(
        sb.table("subscriptions")
        .select("id")
        .eq("user_id", str(user_id))
        .eq("plan_id", plan_id)
        .order("created_at", desc=True)
        .limit(1),
        error_detail="supabase_query_failed",
    )

    if existing:
        await safe_execute(
            sb.table("subscriptions")
            .update(updates)
            .eq("id", str(existing[0]["id"])),
            error_detail="supabase_update_failed",
        )
        return

    row: dict[str, Any] = {
        **updates,
        "user_id": str(user_id),
        "plan_id": plan_id,
    }
    await safe_execute(
        sb.table("subscriptions").insert(row),
        error_detail="supabase_insert_failed",
    )


class ConfirmRazorpayOrderBody(BaseModel):
    plan_id: str = Field(..., min_length=1)
    razorpay_order_id: str = Field(..., min_length=1)
    razorpay_payment_id: str = Field(..., min_length=1)
    razorpay_signature: str = Field(..., min_length=1)


@app.post("/api/me/razorpay/order/confirm")
async def confirm_razorpay_order(
    body: ConfirmRazorpayOrderBody, sb: SbDep, user_id: UserIdDep
) -> dict[str, Any]:
    if not settings.razorpay_key_id or not settings.razorpay_key_secret:
        raise HTTPException(status_code=500, detail="razorpay_not_configured")

    oid = body.razorpay_order_id.strip()
    pid = body.razorpay_payment_id.strip()
    sig = body.razorpay_signature.strip()
    if not rz_verify_order_payment_signature(oid, pid, sig):
        raise HTTPException(status_code=400, detail="invalid_payment_signature")

    rz_order = await rz_fetch_order(oid)
    rz_pay = await rz_fetch_payment(pid)

    if str(rz_pay.get("order_id") or "") != oid:
        raise HTTPException(status_code=400, detail="payment_order_mismatch")

    pay_status = str(rz_pay.get("status") or "").lower()
    if pay_status not in ("captured", "authorized"):
        raise HTTPException(status_code=400, detail="payment_not_captured")

    ord_status = str(rz_order.get("status") or "").lower()
    if ord_status not in ("paid", "attempted"):
        if pay_status != "captured":
            raise HTTPException(status_code=400, detail="order_not_paid")

    notes_raw = rz_order.get("notes")
    notes: dict[str, str] = {}
    if isinstance(notes_raw, dict):
        notes = {
            str(k): str(v) if v is not None else ""
            for k, v in notes_raw.items()
        }

    if (notes.get("user_id") or "").strip() != str(user_id):
        raise HTTPException(status_code=403, detail="payment_user_mismatch")
    if (notes.get("plan_id") or "").strip() != body.plan_id.strip():
        raise HTTPException(status_code=400, detail="payment_plan_mismatch")

    plan_rows = await safe_execute(
        sb.table("subscription_plans")
        .select("price_inr_paise")
        .eq("id", body.plan_id)
        .eq("active", True)
        .limit(1),
        error_detail="supabase_query_failed",
    )
    if not plan_rows:
        raise HTTPException(status_code=400, detail="unknown_or_inactive_plan")
    expected = int(plan_rows[0].get("price_inr_paise") or 0)
    got_amount = int(rz_order.get("amount") or 0)
    if expected > 0 and got_amount != expected:
        raise HTTPException(status_code=400, detail="order_amount_mismatch")

    await _finalize_subscription_after_order_paid(
        sb,
        user_id=user_id,
        plan_id=body.plan_id.strip(),
        razorpay_order_id=oid,
        razorpay_payment_id=pid,
    )
    return {"ok": True}


class SyncRazorpayOrderBody(BaseModel):
    plan_id: str = Field(..., min_length=1)


@app.post("/api/me/razorpay/order/sync")
async def sync_razorpay_order_after_payment(
    body: SyncRazorpayOrderBody, sb: SbDep, user_id: UserIdDep
) -> dict[str, Any]:
    """
    Fallback when webhooks lag: activate a pending subscription if the Razorpay order is paid.
    Requires `subscriptions.razorpay_order_id` (see migration `0010_subscriptions_razorpay_order_id.sql`).
    """
    if not settings.razorpay_key_id or not settings.razorpay_key_secret:
        raise HTTPException(status_code=500, detail="razorpay_not_configured")

    pending = await safe_execute(
        sb.table("subscriptions")
        .select("id,razorpay_order_id,plan_id,status")
        .eq("user_id", str(user_id))
        .eq("plan_id", body.plan_id.strip())
        .eq("status", "past_due")
        .order("created_at", desc=True)
        .limit(1),
        error_detail="supabase_query_failed",
    )
    if not pending:
        return {"ok": False, "reason": "no_pending_subscription"}
    oid_raw = pending[0].get("razorpay_order_id")
    if not oid_raw or not str(oid_raw).strip():
        return {"ok": False, "reason": "missing_razorpay_order_id"}

    oid = str(oid_raw).strip()
    rz_order = await rz_fetch_order(oid)
    ord_status = str(rz_order.get("status") or "").lower()
    if ord_status != "paid":
        return {"ok": False, "reason": "order_not_paid", "order_status": ord_status}

    notes_raw = rz_order.get("notes")
    notes: dict[str, str] = {}
    if isinstance(notes_raw, dict):
        notes = {
            str(k): str(v) if v is not None else ""
            for k, v in notes_raw.items()
        }
    if (notes.get("user_id") or "").strip() != str(user_id):
        return {"ok": False, "reason": "payment_user_mismatch"}
    if (notes.get("plan_id") or "").strip() != body.plan_id.strip():
        return {"ok": False, "reason": "payment_plan_mismatch"}

    await _finalize_subscription_after_order_paid(
        sb,
        user_id=user_id,
        plan_id=body.plan_id.strip(),
        razorpay_order_id=oid,
        razorpay_payment_id=None,
    )
    return {"ok": True}


def _rz_verify_signature(*, secret: str, body: bytes, signature: str) -> bool:
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    try:
        return hmac.compare_digest(digest, signature)
    except Exception:  # noqa: BLE001
        return False


@app.post("/api/webhooks/razorpay")
async def razorpay_webhook(
    request: Request,
    sb: SbServiceDep,
    x_razorpay_signature: Annotated[str | None, Header(alias="x-razorpay-signature")] = None,
):
    secret = settings.razorpay_webhook_secret.strip()
    if not secret:
        raise HTTPException(status_code=500, detail="razorpay_webhook_not_configured")

    raw = await request.body()
    sig = (x_razorpay_signature or "").strip()
    if not sig or not _rz_verify_signature(secret=secret, body=raw, signature=sig):
        raise HTTPException(status_code=401, detail="invalid_signature")

    try:
        # Razorpay payloads are JSON; tolerate UTF-8 BOM if present.
        payload = json.loads(raw.decode("utf-8-sig"))
    except Exception:  # noqa: BLE001
        # Signature already verified; don't return 400 (Razorpay will retry forever).
        # Log enough context to debug without dumping entire payload.
        ct = request.headers.get("content-type")
        log.exception(
            "razorpay_webhook: invalid_json content_type=%s raw_len=%s raw_prefix=%r",
            ct,
            len(raw),
            raw[:200],
        )
        return {"ok": True, "stored": False, "reason": "invalid_json"}

    if not isinstance(payload, dict):
        log.error("razorpay_webhook: payload_not_object type=%s", type(payload).__name__)
        return {"ok": True, "stored": False, "reason": "payload_not_object"}

    event_id = str(payload.get("id") or "")
    event_type = str(payload.get("event") or "")
    if not event_id or not event_type:
        log.error(
            "razorpay_webhook: invalid_event raw_len=%s keys=%s",
            len(raw),
            sorted(payload.keys()),
        )
        return {"ok": True, "stored": False, "reason": "invalid_event"}

    def _merge_notes_from_entities() -> dict[str, str]:
        """Collect `notes` from nested Razorpay entities (order inherits into payment when present)."""
        p = payload.get("payload", {})
        if not isinstance(p, dict):
            return {}
        merged: dict[str, str] = {}
        for key in ("subscription", "payment", "order", "invoice"):
            block = p.get(key)
            if not isinstance(block, dict):
                continue
            ent = block.get("entity", {})
            if not isinstance(ent, dict):
                continue
            notes = ent.get("notes")
            if not isinstance(notes, dict):
                continue
            for nk, nv in notes.items():
                if nv is None:
                    continue
                merged[str(nk)] = str(nv)
        return merged

    async def _activate_from_notes(*, notes: dict[str, str], payment_id: str | None = None) -> bool:
        uid = (notes.get("user_id") or "").strip()
        pid = (notes.get("plan_id") or "").strip()
        if not uid or not pid:
            return False

        plans = await safe_execute(
            sb.table("subscription_plans").select("billing_period").eq("id", pid).limit(1),
            error_detail="supabase_query_failed",
        )
        bp = str(plans[0].get("billing_period") or "") if plans else ""

        now = datetime.now(timezone.utc)
        end_iso: str | None
        if bp == "monthly":
            end_iso = (now + timedelta(days=31)).isoformat()
        elif bp == "yearly":
            end_iso = (now + timedelta(days=365)).isoformat()
        else:
            end_iso = None

        row: dict[str, Any] = {
            "user_id": uid,
            "plan_id": pid,
            "status": "active",
            "current_period_start": now.isoformat(),
            "cancel_at_period_end": False,
            "razorpay_last_event_id": event_id,
        }
        if payment_id:
            row["razorpay_payment_id"] = payment_id
        if end_iso:
            row["current_period_end"] = end_iso

        # If we have a pending row from order checkout, update it; else create new.
        existing = await safe_execute(
            sb.table("subscriptions")
            .select("id,status,created_at")
            .eq("user_id", uid)
            .eq("plan_id", pid)
            .order("created_at", desc=True)
            .limit(1),
            error_detail="supabase_query_failed",
        )
        if existing:
            await safe_execute(
                sb.table("subscriptions").update(row).eq("id", str(existing[0]["id"])),
                error_detail="supabase_update_failed",
            )
        else:
            await safe_execute(
                sb.table("subscriptions").insert(row),
                error_detail="supabase_insert_failed",
            )
        return True

    # For order-based test payments we won't have subscription entities. Activate via notes.
    if event_type in ("payment.captured", "order.paid", "payment.authorized"):
        ent = (
            payload.get("payload", {})
            if isinstance(payload.get("payload", {}), dict)
            else {}
        )
        pay_ent = (
            ent.get("payment", {}).get("entity", {})
            if isinstance(ent.get("payment", {}), dict)
            else {}
        )
        payment_id = str(pay_ent.get("id") or "").strip() if isinstance(pay_ent, dict) else ""
        try:
            notes = _merge_notes_from_entities()
            activated = await _activate_from_notes(notes=notes, payment_id=payment_id or None)
            if not activated:
                log.warning(
                    "razorpay_webhook: payment/order activation skipped notes=%s event=%s payload_keys=%s",
                    notes,
                    event_type,
                    list((payload.get("payload") or {}).keys())
                    if isinstance(payload.get("payload"), dict)
                    else None,
                )
            return {"ok": True, "activated": activated, "event": event_type}
        except HTTPException:
            log.exception("razorpay_webhook: failed to activate from payment/order notes event_id=%s", event_id)
            return {"ok": True, "activated": False}

    # Idempotency: if we already stored this event id, ignore.
    # NOTE: This relies on Supabase being reachable and the table existing.
    # If Supabase is down/misconfigured we still return 200 so Razorpay doesn't
    # keep retrying forever; the event can be reconciled from Razorpay later.
    try:
        existing = await safe_execute(
            sb.table("razorpay_webhook_events").select("id").eq("id", event_id).limit(1),
            error_detail="supabase_query_failed",
        )
        if existing:
            return {"ok": True, "duplicate": True}
    except HTTPException:
        log.exception("razorpay_webhook: idempotency check failed event_id=%s", event_id)
        return {"ok": True, "stored": False, "reason": "supabase_unavailable"}

    entity = (
        payload.get("payload", {})
        .get("subscription", {})
        .get("entity", {})
    )
    rz_sub_id = str(entity.get("id") or "")

    sub_row = None
    if rz_sub_id:
        try:
            rows = await safe_execute(
                sb.table("subscriptions")
                .select("id,user_id,plan_id,status,current_period_start,current_period_end")
                .eq("razorpay_subscription_id", rz_sub_id)
                .order("created_at", desc=True)
                .limit(1),
                error_detail="supabase_query_failed",
            )
            sub_row = rows[0] if rows else None
        except HTTPException:
            log.exception(
                "razorpay_webhook: failed to lookup subscription rz_sub_id=%s event_id=%s",
                rz_sub_id,
                event_id,
            )
            return {"ok": True, "stored": False, "reason": "supabase_unavailable"}

    sub_uuid = str(sub_row["id"]) if sub_row else None

    # Always store the event for audit, even if we can't map it.
    try:
        await safe_execute(
            sb.table("razorpay_webhook_events").insert(
                {
                    "id": event_id,
                    "subscription_id": sub_uuid,
                    "event_type": event_type,
                    "payload": payload,
                }
            ),
            error_detail="supabase_insert_failed",
        )
    except HTTPException:
        log.exception("razorpay_webhook: failed to store event audit event_id=%s", event_id)
        return {"ok": True, "stored": False, "reason": "supabase_unavailable"}

    if not sub_row:
        return {"ok": True, "stored": True, "mapped": False}

    # Map Razorpay event -> local subscription status.
    now = datetime.now(timezone.utc)
    updates: dict[str, Any] = {"razorpay_last_event_id": event_id}

    if event_type in ("subscription.activated", "subscription.charged"):
        updates["status"] = "active"
        updates["current_period_start"] = _iso(now)
        # Razorpay provides end_at (unix) in subscription entity sometimes.
        end_at = entity.get("end_at")
        if isinstance(end_at, (int, float)) and end_at > 0:
            dt = datetime.fromtimestamp(float(end_at), tz=timezone.utc)
            updates["current_period_end"] = _iso(dt)
    elif event_type in ("subscription.cancelled",):
        updates["status"] = "cancelled"
    elif event_type in ("subscription.halted", "subscription.paused"):
        updates["status"] = "past_due"

    try:
        await safe_execute(
            sb.table("subscriptions").update(updates).eq("id", sub_uuid),
            error_detail="supabase_update_failed",
        )
    except HTTPException:
        log.exception(
            "razorpay_webhook: failed to update local subscription sub_id=%s event_id=%s",
            sub_uuid,
            event_id,
        )
        return {"ok": True, "stored": True, "updated": False}

    return {"ok": True}


@app.get("/api/me/subscriptions")
async def my_subscriptions(sb: SbDep, user_id: UserIdDep):
    return await safe_execute(
        sb.table("subscriptions")
        .select("id,plan_id,status,current_period_start,current_period_end,created_at")
        .eq("user_id", str(user_id))
        .order("created_at", desc=True)
        .limit(50)
    )


@app.get("/api/me/licenses")
async def my_licenses(sb: SbDep, user_id: UserIdDep):
    return await safe_execute(
        sb.table("licenses")
        .select("id,status,expires_at,issued_at")
        .eq("user_id", str(user_id))
        .order("issued_at", desc=True)
        .limit(50)
    )


@app.get("/api/me/usage/summary-month")
async def usage_summary_month(sb: SbDep, user_id: UserIdDep):
    now = datetime.now(timezone.utc)
    month_start = datetime(now.year, now.month, 1, tzinfo=timezone.utc).isoformat()
    logs = await safe_execute(
        sb.table("usage_logs")
        .select("quantity")
        .eq("user_id", str(user_id))
        .gte("created_at", month_start)
    )
    return {"used_mb": float(sum(float(l.get("quantity") or 0) for l in logs))}


@app.get("/api/me/usage")
async def my_usage(
    sb: SbDep,
    user_id: UserIdDep,
    limit: Annotated[int, Query(ge=1, le=500)] = 200,
):
    return await safe_execute(
        sb.table("usage_logs")
        .select("id,event_type,quantity,metadata,created_at")
        .eq("user_id", str(user_id))
        .order("created_at", desc=True)
        .limit(limit)
    )


class UsageIn(BaseModel):
    event_type: str = Field(..., min_length=1)
    quantity: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@app.post("/api/me/usage", status_code=201)
async def ingest_usage(body: UsageIn, sb: SbDep, user_id: UserIdDep):
    await safe_execute(
        sb.table("usage_logs").insert(
            {
                "user_id": str(user_id),
                "event_type": body.event_type,
                "quantity": body.quantity,
                "metadata": body.metadata,
            }
        ),
        error_detail="supabase_insert_failed",
    )
    return {"ok": True}


# ---------------------------------------------------------------------------
# Billing profile (per user, upsert on user_id)
# ---------------------------------------------------------------------------


class BillingProfile(BaseModel):
    company: str | None = None
    billing_email: str | None = None
    address_line1: str | None = None
    address_line2: str | None = None
    city: str | None = None
    postal_code: str | None = None
    state: str | None = None
    country: str | None = None
    gstin: str | None = None
    legal_name: str | None = None
    auto_renew: bool = True


_BILLING_FIELDS = (
    "company,billing_email,address_line1,address_line2,city,"
    "postal_code,state,country,gstin,legal_name,auto_renew"
)


@app.get("/api/me/billing", response_model=BillingProfile)
async def get_billing(sb: SbDep, user_id: UserIdDep) -> BillingProfile:
    rows = await safe_execute(
        sb.table("billing_profiles")
        .select(_BILLING_FIELDS)
        .eq("user_id", str(user_id))
        .limit(1)
    )
    if not rows:
        return BillingProfile()
    return BillingProfile(**rows[0])


@app.put("/api/me/billing", response_model=BillingProfile)
async def update_billing(
    body: BillingProfile, sb: SbDep, user_id: UserIdDep
) -> BillingProfile:
    payload = {"user_id": str(user_id), **body.model_dump()}
    await safe_execute(
        sb.table("billing_profiles").upsert(payload, on_conflict="user_id"),
        error_detail="supabase_upsert_failed",
    )
    return body


# ---------------------------------------------------------------------------
# User-facing transactions (subscriptions joined with their plan)
# ---------------------------------------------------------------------------


@app.get("/api/me/transactions")
async def my_transactions(sb: SbDep, user_id: UserIdDep):
    subs = await safe_execute(
        sb.table("subscriptions")
        .select("id,plan_id,status,created_at,current_period_start,current_period_end")
        .eq("user_id", str(user_id))
        .order("created_at", desc=True)
        .limit(200)
    )
    if not subs:
        return []

    plan_ids = sorted({s["plan_id"] for s in subs if s.get("plan_id")})
    plans_by_id: dict[str, dict[str, Any]] = {}
    if plan_ids:
        plan_rows = await safe_execute(
            sb.table("subscription_plans")
            .select("id,name,price_inr_paise,billing_period")
            .in_("id", plan_ids)
        )
        plans_by_id = {p["id"]: p for p in plan_rows}

    out: list[dict[str, Any]] = []
    for s in subs:
        p = plans_by_id.get(s.get("plan_id") or "", {})
        out.append(
            {
                "id": str(s["id"]),
                "plan_id": s.get("plan_id"),
                "plan_name": p.get("name") or s.get("plan_id"),
                "status": s.get("status"),
                "amount_inr_paise": int(p.get("price_inr_paise") or 0),
                "billing_period": p.get("billing_period"),
                "method": "Razorpay",
                "created_at": _iso(s.get("created_at")),
                "current_period_start": _iso(s.get("current_period_start")),
                "current_period_end": _iso(s.get("current_period_end")),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Module catalog (drives admin matrix + plan edit dialog)
# ---------------------------------------------------------------------------


@app.get("/api/modules")
async def list_modules(sb: SbDep, user_id: UserIdDep):
    _ = user_id
    return await safe_execute(
        sb.table("module_catalog")
        .select("id,name,description,sort_order")
        .order("sort_order")
        .order("name")
    )


class MeModuleRow(BaseModel):
    id: str
    name: str
    description: str
    sort_order: int
    enabled: bool


class MeModulesOut(BaseModel):
    """Catalog rows plus whether the caller's plan (or admin) unlocks each module."""

    plan_id: str | None
    modules: list[MeModuleRow]
    enabled_module_ids: list[str]


@app.get("/api/me/modules", response_model=MeModulesOut)
async def my_modules(sb: SbDep, user_id: UserIdDep) -> MeModulesOut:
    catalog = await safe_execute(
        sb.table("module_catalog")
        .select("id,name,description,sort_order")
        .order("sort_order")
        .order("name")
    )
    admin = await _is_admin(sb, user_id)
    plan_id = await _active_subscription_plan_id(sb, user_id)
    flags: dict[str, bool] = {}
    if admin:
        flags = {str(r["id"]): True for r in catalog}
    elif plan_id:
        plans = await safe_execute(
            sb.table("subscription_plans")
            .select("features")
            .eq("id", plan_id)
            .limit(1)
        )
        if plans:
            flags = _features_modules(plans[0].get("features"))

    rows: list[MeModuleRow] = []
    enabled_ids: list[str] = []
    for r in catalog:
        mid = str(r["id"])
        en = bool(flags.get(mid, False))
        rows.append(
            MeModuleRow(
                id=mid,
                name=str(r.get("name") or mid),
                description=str(r.get("description") or ""),
                sort_order=int(r.get("sort_order") or 0),
                enabled=en,
            )
        )
        if en:
            enabled_ids.append(mid)

    return MeModulesOut(plan_id=plan_id, modules=rows, enabled_module_ids=enabled_ids)


# ---------------------------------------------------------------------------
# Admin: overview / users / transactions / revenue
# ---------------------------------------------------------------------------


@app.get("/api/admin/overview")
async def admin_overview(sb: SbDep, user_id: UserIdDep):
    await _require_admin(sb, user_id)

    now = datetime.now(timezone.utc)
    iso_7d = (now - timedelta(days=7)).isoformat()
    iso_30d = (now - timedelta(days=30)).isoformat()

    users_total = await count_rows(sb, "users")
    users_active_7d = await count_rows(sb, "users", gte={"last_seen_at": iso_7d})
    users_new_7d = await count_rows(sb, "users", gte={"created_at": iso_7d})
    active_subs = await count_rows(sb, "subscriptions", eq={"status": "active"})
    scans_30d = await count_rows(
        sb,
        "usage_logs",
        eq={"event_type": "scan_complete"},
        gte={"created_at": iso_30d},
    )

    # MRR / ARR — fetch active subscriptions and join with plans in Python.
    sub_rows = await safe_execute(
        sb.table("subscriptions").select("plan_id").eq("status", "active")
    )
    plan_ids = sorted({r["plan_id"] for r in sub_rows if r.get("plan_id")})
    mrr_paise = 0
    arr_yearly_paise = 0
    if plan_ids:
        plans = await safe_execute(
            sb.table("subscription_plans")
            .select("id,price_inr_paise,billing_period")
            .in_("id", plan_ids)
        )
        plan_by_id = {p["id"]: p for p in plans}
        for r in sub_rows:
            p = plan_by_id.get(r.get("plan_id") or "")
            if not p:
                continue
            price = int(p.get("price_inr_paise") or 0)
            if p.get("billing_period") == "monthly":
                mrr_paise += price
            elif p.get("billing_period") == "yearly":
                arr_yearly_paise += price

    # Recovered MB — sum quantity over the last 30 days.
    usage_rows = await safe_execute(
        sb.table("usage_logs").select("quantity").gte("created_at", iso_30d)
    )
    recovered_mb_30d = float(sum(float(u.get("quantity") or 0) for u in usage_rows))

    mrr = mrr_paise / 100.0
    arr = (mrr * 12.0) + (arr_yearly_paise / 100.0)
    return {
        "users_total": users_total,
        "users_active_7d": users_active_7d,
        "users_new_7d": users_new_7d,
        "active_subscriptions": active_subs,
        "mrr_inr": mrr,
        "arr_inr": arr,
        "recovered_mb_30d": recovered_mb_30d,
        "scans_30d": scans_30d,
    }


@app.get("/api/admin/users")
async def admin_users(
    sb: SbDep,
    user_id: UserIdDep,
    q: Annotated[str | None, Query(max_length=200)] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    await _require_admin(sb, user_id)

    base = sb.table("users").select("id,email,display_name,created_at,last_seen_at")
    count_q = sb.table("users").select("id", count="exact", head=True)
    if q:
        # PostgREST `or=()` filter: match on email or display_name.
        like = f"*{q}*"
        base = base.or_(f"email.ilike.{like},display_name.ilike.{like}")
        count_q = count_q.or_(f"email.ilike.{like},display_name.ilike.{like}")

    users = await safe_execute(
        base.order("created_at", desc=True).range(offset, offset + limit - 1)
    )

    try:
        total_res = await count_q.execute()
        total = int(getattr(total_res, "count", 0) or 0)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail="supabase_count_failed") from exc

    user_ids = [u["id"] for u in users]
    admins: set[str] = set()
    active_by_user: dict[str, list[str]] = {}
    if user_ids:
        admin_rows = await safe_execute(
            sb.table("admin_users").select("user_id").in_("user_id", user_ids)
        )
        admins = {str(r["user_id"]) for r in admin_rows}

        sub_rows = await safe_execute(
            sb.table("subscriptions")
            .select("user_id,plan_id,created_at,status")
            .in_("user_id", user_ids)
            .eq("status", "active")
            .order("created_at", desc=True)
        )
        for s in sub_rows:
            active_by_user.setdefault(str(s["user_id"]), []).append(s.get("plan_id"))

    items = []
    for u in users:
        uid = str(u["id"])
        plans = active_by_user.get(uid, [])
        items.append(
            {
                "id": uid,
                "email": u.get("email"),
                "display_name": u.get("display_name"),
                "created_at": _iso(u.get("created_at")),
                "last_seen_at": _iso(u.get("last_seen_at")),
                "is_admin": uid in admins,
                "active_plan_id": plans[0] if plans else None,
                "active_subscriptions": len(plans),
                "status": "Active",
            }
        )
    return {"items": items, "total": total, "limit": limit, "offset": offset}


@app.get("/api/admin/transactions")
async def admin_transactions(
    sb: SbDep,
    user_id: UserIdDep,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
):
    await _require_admin(sb, user_id)

    subs = await safe_execute(
        sb.table("subscriptions")
        .select("id,user_id,plan_id,status,created_at")
        .order("created_at", desc=True)
        .limit(limit)
    )
    if not subs:
        return []

    plan_ids = sorted({s["plan_id"] for s in subs if s.get("plan_id")})
    user_ids = sorted({s["user_id"] for s in subs if s.get("user_id")})

    plans_by_id: dict[str, dict[str, Any]] = {}
    if plan_ids:
        plan_rows = await safe_execute(
            sb.table("subscription_plans")
            .select("id,name,price_inr_paise,billing_period")
            .in_("id", plan_ids)
        )
        plans_by_id = {p["id"]: p for p in plan_rows}

    users_by_id: dict[str, dict[str, Any]] = {}
    if user_ids:
        user_rows = await safe_execute(
            sb.table("users").select("id,email").in_("id", user_ids)
        )
        users_by_id = {str(u["id"]): u for u in user_rows}

    out: list[dict[str, Any]] = []
    for s in subs:
        p = plans_by_id.get(s.get("plan_id") or "", {})
        u = users_by_id.get(str(s.get("user_id") or ""), {})
        out.append(
            {
                "id": str(s["id"]),
                "user_id": str(s["user_id"]),
                "user_email": u.get("email"),
                "plan_id": s.get("plan_id"),
                "plan_name": p.get("name") or s.get("plan_id"),
                "status": s.get("status"),
                "amount_inr_paise": int(p.get("price_inr_paise") or 0),
                "billing_period": p.get("billing_period"),
                "created_at": _iso(s.get("created_at")),
            }
        )
    return out


@app.get("/api/admin/revenue/monthly")
async def admin_revenue_monthly(sb: SbDep, user_id: UserIdDep):
    await _require_admin(sb, user_id)

    cutoff = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()
    subs = await safe_execute(
        sb.table("subscriptions").select("plan_id,created_at").gte("created_at", cutoff)
    )
    if not subs:
        return []

    plan_ids = sorted({s["plan_id"] for s in subs if s.get("plan_id")})
    plans_by_id: dict[str, dict[str, Any]] = {}
    if plan_ids:
        plan_rows = await safe_execute(
            sb.table("subscription_plans").select("id,price_inr_paise").in_("id", plan_ids)
        )
        plans_by_id = {p["id"]: p for p in plan_rows}

    buckets: dict[str, dict[str, Any]] = {}
    for s in subs:
        created = s.get("created_at")
        if not created:
            continue
        try:
            dt = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
        except ValueError:
            continue
        key = f"{dt.year:04d}-{dt.month:02d}-01"
        b = buckets.setdefault(key, {"month": key, "revenue_paise": 0, "new_subs": 0})
        b["new_subs"] += 1
        p = plans_by_id.get(s.get("plan_id") or "")
        if p:
            b["revenue_paise"] += int(p.get("price_inr_paise") or 0)

    return [
        {
            "month": b["month"],
            "revenue_inr": float(b["revenue_paise"]) / 100.0,
            "new_subscriptions": int(b["new_subs"]),
        }
        for b in sorted(buckets.values(), key=lambda x: x["month"])
    ]


# ---------------------------------------------------------------------------
# Admin: plan ↔ module matrix (stored inside subscription_plans.features)
# ---------------------------------------------------------------------------


class PlanModulesBody(BaseModel):
    modules: dict[str, bool] = Field(default_factory=dict)


class PlanModulesOut(BaseModel):
    ok: bool = True
    plan_id: str
    modules: dict[str, bool]


@app.put("/api/admin/plans/{plan_id}/modules", response_model=PlanModulesOut)
async def admin_set_plan_modules(
    plan_id: str, body: PlanModulesBody, sb: SbDep, user_id: UserIdDep
):
    await _require_admin(sb, user_id)
    rows = await safe_execute(
        sb.table("subscription_plans").select("features").eq("id", plan_id).limit(1)
    )
    if not rows:
        raise HTTPException(status_code=404, detail="unknown_plan")

    feat = rows[0].get("features")
    feat_obj: dict[str, Any] = dict(feat) if isinstance(feat, dict) else {}
    feat_obj["modules"] = {k: bool(v) for k, v in body.modules.items()}

    await safe_execute(
        sb.table("subscription_plans").update({"features": feat_obj}).eq("id", plan_id),
        error_detail="supabase_update_failed",
    )

    return PlanModulesOut(plan_id=plan_id, modules=feat_obj["modules"])
