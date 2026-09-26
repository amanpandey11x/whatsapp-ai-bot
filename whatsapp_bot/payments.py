"""
Razorpay integration — pure gateway mechanics only (create a link, verify a
webhook signature). Deciding *whether* to send a link, at what amount, and
what to say lives in services.maybe_send_payment_cta; recording the outcome
lives in tasks.process_payment_webhook. Kept separate from services.py the
same way flow_crypto.py is kept separate from flow_screens.py — one file per
external protocol this project speaks.

Uses plain `requests` against Razorpay's REST API rather than pulling in
their SDK — this project already talks to the Meta Graph API the same way
(see services._post_whatsapp_payload), so this matches the existing style
and adds no new dependency.
"""

import hashlib
import hmac
import logging
from datetime import timedelta

import requests
from django.conf import settings
from django.utils import timezone

logger = logging.getLogger(__name__)

RAZORPAY_API_BASE = "https://api.razorpay.com/v1"


def razorpay_configured():
    """False in any environment where RAZORPAY_KEY_ID/SECRET aren't set —
    every caller treats that as 'skip payments silently', not an error, so
    dev/staging without keys keeps working exactly as it does today."""
    return bool(settings.RAZORPAY_KEY_ID and settings.RAZORPAY_KEY_SECRET)


def _auth():
    return (settings.RAZORPAY_KEY_ID, settings.RAZORPAY_KEY_SECRET)


def create_payment_link(order):
    """
    Creates a Razorpay Payment Link for `order` (already saved, with
    contact/amount/is_advance set) and stores the returned id/url back onto
    it. Returns True on success. Never raises into the caller — a gateway
    failure here should degrade to "no payment link sent this turn", not
    break the rest of the conversation reply.
    """
    if not razorpay_configured():
        logger.warning("Razorpay not configured — skipping payment link for order %s", order.id)
        return False

    expiry_hours = settings.PAYMENT_LINK_EXPIRY_HOURS
    payload = {
        "amount": int(order.amount * 100),  # Razorpay wants paise, not rupees
        "currency": "INR",
        "accept_partial": False,
        "description": (
            f"Booking advance — order #{order.id}" if order.is_advance
            else f"Payment — order #{order.id}"
        ),
        "reference_id": f"order-{order.id}",
        "customer": {"contact": order.contact.phone_number},
        # WhatsApp is what actually delivers this link to the customer —
        # don't let Razorpay also fire its own SMS/email for the same link.
        "notify": {"sms": False, "email": False},
        "expire_by": int((timezone.now() + timedelta(hours=expiry_hours)).timestamp()),
        "notes": {"contact_id": str(order.contact_id), "order_id": str(order.id)},
    }

    try:
        response = requests.post(
            f"{RAZORPAY_API_BASE}/payment_links", json=payload, auth=_auth(), timeout=10,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.error("Failed to create Razorpay payment link for order %s: %s", order.id, exc)
        return False

    data = response.json()
    order.payment_link_id = data["id"]
    order.payment_link_url = data["short_url"]
    order.save(update_fields=["payment_link_id", "payment_link_url"])
    return True


def cancel_payment_link(order):
    """
    Cancels `order`'s Razorpay Payment Link on the gateway side — called by
    services.invalidate_stale_orders whenever the cart it was quoted against
    no longer reflects what's actually selected (see tasks.py's edit_cart
    handling), so the old link genuinely can't be paid anymore, not just
    marked stale in our own DB while still live on Razorpay's.

    Same never-raise-into-the-caller contract as create_payment_link — a
    gateway failure here degrades to "the link stays live a bit longer than
    intended", not a broken cart-edit reply. The caller still marks the
    order STATUS_EXPIRED locally regardless of whether this call succeeds,
    so maybe_send_payment_cta's exact-snapshot reuse check can never pick a
    stale order back up either way.
    """
    if not order.payment_link_id:
        return False
    try:
        response = requests.post(
            f"{RAZORPAY_API_BASE}/payment_links/{order.payment_link_id}/cancel",
            auth=_auth(), timeout=10,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.error("Failed to cancel Razorpay payment link for order %s: %s", order.id, exc)
        return False
    return True


def verify_webhook_signature(request_body: bytes, signature_header: str) -> bool:
    """
    Verifies Razorpay's X-Razorpay-Signature header (HMAC-SHA256 of the raw
    request body, keyed with the webhook secret configured on Razorpay's
    dashboard — a separate secret from the API key/secret pair used above)
    against what we'd compute ourselves. hmac.compare_digest is a
    constant-time comparison — a plain `==` here would leak timing
    information an attacker could use to guess the signature byte by byte.

    Returns False (never raises) for anything malformed — an unverifiable
    webhook should be rejected the same way as a wrong one, not crash the
    handler.
    """
    if not settings.RAZORPAY_WEBHOOK_SECRET or not signature_header:
        return False
    expected = hmac.new(
        settings.RAZORPAY_WEBHOOK_SECRET.encode(), request_body, hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature_header)
