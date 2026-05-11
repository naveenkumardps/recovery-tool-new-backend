from __future__ import annotations

import hashlib
import hmac
import anyio
import razorpay
import logging

from recovery_api.config import get_settings

log = logging.getLogger("recovery_api.razorpay")

def _client() -> razorpay.Client:
    s = get_settings()
    if not s.razorpay_key_id or not s.razorpay_key_secret:
        raise RuntimeError("razorpay_not_configured")
    c = razorpay.Client(auth=(s.razorpay_key_id, s.razorpay_key_secret))
    return c


async def create_subscription(payload: dict) -> dict:
    """
    Create a Razorpay subscription (sync SDK wrapped for async FastAPI).
    """
    def _do() -> dict:
        log.info("razorpay: create_subscription payload_keys=%s", sorted(payload.keys()))
        return _client().subscription.create(payload)  # type: ignore[no-any-return]

    return await anyio.to_thread.run_sync(_do)


async def create_plan(payload: dict) -> dict:
    """
    Create a Razorpay plan (sync SDK wrapped for async FastAPI).
    Razorpay plan is used by subscriptions (`plan_id`).
    """

    def _do() -> dict:
        log.info("razorpay: create_plan period=%s interval=%s", payload.get("period"), payload.get("interval"))
        return _client().plan.create(payload)  # type: ignore[no-any-return]

    return await anyio.to_thread.run_sync(_do)


def verify_order_payment_signature(order_id: str, payment_id: str, signature: str) -> bool:
    """Razorpay standard: HMAC_SHA256(secret, order_id + '|' + payment_id)."""
    s = get_settings()
    secret = (s.razorpay_key_secret or "").strip()
    if not secret or not signature:
        return False
    message = f"{order_id}|{payment_id}".encode("utf-8")
    digest = hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()
    try:
        return hmac.compare_digest(digest, signature.strip())
    except Exception:  # noqa: BLE001
        return False


async def fetch_order(order_id: str) -> dict:
    def _do() -> dict:
        return _client().order.fetch(order_id)  # type: ignore[no-any-return]

    return await anyio.to_thread.run_sync(_do)


async def fetch_payment(payment_id: str) -> dict:
    def _do() -> dict:
        return _client().payment.fetch(payment_id)  # type: ignore[no-any-return]

    return await anyio.to_thread.run_sync(_do)


async def create_order(payload: dict) -> dict:
    """
    Create a Razorpay order (one-time payment).
    Used as a test-mode fallback when subscription mandates are not available.
    """

    def _do() -> dict:
        log.info("razorpay: create_order amount=%s currency=%s", payload.get("amount"), payload.get("currency"))
        try:
            return _client().order.create(payload)  # type: ignore[no-any-return]
        except Exception as exc:  # noqa: BLE001
            # Razorpay SDK exceptions often carry structured fields (`status_code`, `error`).
            status = getattr(exc, "status_code", None)
            err = getattr(exc, "error", None)
            log.exception(
                "razorpay: create_order failed status=%s error=%s payload_keys=%s",
                status,
                err,
                sorted(payload.keys()),
            )
            raise

    return await anyio.to_thread.run_sync(_do)

