import json
import logging

from django.conf import settings
from django.db import IntegrityError
from django.http import HttpResponse, JsonResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from . import payments
from .flow_crypto import FlowDecryptionError, decrypt_request, encrypt_response
from .flow_screens import handle_flow_action
from .models import Contact, Message
from .tasks import process_incoming_message, process_payment_webhook

logger = logging.getLogger(__name__)


def _extract_message_text(message):
    """
    Pulls the effective "text" out of a webhook message event — either a
    typed message body, or the stable `id` of a tapped interactive reply
    (button or list). Using `id` rather than `title` means a button tap goes
    straight through the exact same resolve_*() matching as if the user had
    typed that id — e.g. the language buttons use id="1"/"2", so no separate
    interactive-specific handling is needed anywhere downstream.

    A submitted WhatsApp Flow (see services.send_whatsapp_flow) arrives as
    its own interactive sub-type, "nfm_reply", carrying the whole submission
    as a JSON string rather than a single id — that string is passed through
    as-is; services.parse_itr_flow_reply is what actually reads it.
    """
    if message.get("type") == "interactive":
        interactive = message.get("interactive", {})
        if interactive.get("type") == "nfm_reply":
            return interactive.get("nfm_reply", {}).get("response_json", "")
        reply = interactive.get("button_reply") or interactive.get("list_reply") or {}
        return reply.get("id", "")
    return message.get("text", {}).get("body", "")


@csrf_exempt
@require_http_methods(["GET", "POST"])
def whatsapp_webhook(request):
    """
    GET: Meta's verification handshake (run once when you register the webhook URL).
    POST: Actual incoming message events.
    """
    if request.method == "GET":
        verify_token = request.GET.get("hub.verify_token")
        challenge = request.GET.get("hub.challenge")
        if verify_token == settings.WHATSAPP_VERIFY_TOKEN:
            return HttpResponse(challenge)
        return HttpResponse("Invalid verify token", status=403)

    # POST — incoming message event
    received_at = timezone.now()
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        logger.warning("Received non-JSON payload on webhook")
        return JsonResponse({"status": "ignored"})

    logger.info("Webhook payload: %s", data)

    try:
        entry = data["entry"][0]["changes"][0]["value"]
        message = entry["messages"][0]

        phone_number = message["from"]
        text = _extract_message_text(message)
        wa_message_id = message["id"]
        # WhatsApp's own profile name for this number — greeting-only (see
        # services._greeting_prefix), never the verified real name collected
        # later via the ITR Flow. Absent contacts array (or profile) falls
        # back to None, same as a first-time-ever payload shape surprise.
        contacts = entry.get("contacts") or [{}]
        whatsapp_display_name = contacts[0].get("profile", {}).get("name")

        # Meta retries a webhook delivery on anything but a fast 2xx (and can
        # occasionally redeliver for other transient reasons) — without this
        # check, a retry silently re-runs the whole turn: a second real
        # Gemini API call (classify_and_respond), a second reply sent to the
        # customer, and for an edit_cart/payment turn, a second cart mutation
        # or Order. whatsapp_message_id is stable per Meta message, so a
        # duplicate delivery is detected here before any of that work starts
        # — cheap (one indexed lookup) next to what it prevents. Still 200s
        # a duplicate so Meta doesn't keep retrying it forever.
        if Message.objects.filter(direction="in", whatsapp_message_id=wa_message_id).exists():
            logger.info("Duplicate webhook delivery for wa_message_id=%s — skipping", wa_message_id)
            return JsonResponse({"status": "duplicate"})

        contact, _ = Contact.objects.get_or_create(phone_number=phone_number)
        # Opportunistic resync, not one-time capture: a contact can rename
        # themselves on WhatsApp at any point, so this just keeps drifting
        # toward whatever the last-seen payload said rather than treating
        # the first value as gospel.
        if whatsapp_display_name and contact.whatsapp_display_name != whatsapp_display_name:
            contact.whatsapp_display_name = whatsapp_display_name
            contact.save(update_fields=["whatsapp_display_name"])
        try:
            Message.objects.create(
                contact=contact,
                direction="in",
                body=text,
                whatsapp_message_id=wa_message_id,
            )
        except IntegrityError:
            # The exists() check above is a plain SELECT — two genuinely
            # concurrent retried deliveries for the same wa_message_id can
            # both pass it before either INSERT lands. The DB-level
            # unique_incoming_wa_message_id constraint (see Message.Meta)
            # is the real guarantee; this just turns the race's loser into
            # the same "duplicate, already handled" response instead of a
            # 500 and a second process_incoming_message.delay() below.
            logger.info(
                "Duplicate webhook delivery for wa_message_id=%s (race on insert) — skipping",
                wa_message_id,
            )
            return JsonResponse({"status": "duplicate"})

        # Hand off to Celery so the webhook responds immediately.
        process_incoming_message.delay(contact.id, text, received_at.isoformat(), wa_message_id)

    except (KeyError, IndexError):
        # Not a message event — could be a delivery/read status update. Safe to ignore.
        logger.info("Webhook event had no message payload (likely a status update).")

    return JsonResponse({"status": "received"})


@csrf_exempt
@require_http_methods(["POST"])
def whatsapp_flow_endpoint(request):
    """
    The live data-exchange endpoint for a WhatsApp Flow with dynamic
    screens (services list, cart total, etc.) — every request/response body
    here is end-to-end encrypted per Meta's protocol (see flow_crypto.py),
    completely separate from the plain-webhook JSON whatsapp_webhook()
    handles above. Meta calls this directly (not via a Message the way
    chat replies work), so nothing here touches Celery — it must respond
    synchronously within Meta's timeout.

    A 421 tells Meta the payload couldn't be decrypted, which makes it
    re-fetch our public key and retry — the correct response for any
    decryption failure, not just a "wrong key" one (see
    FlowDecryptionError's use in flow_crypto.decrypt_request).
    """
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        logger.warning("Flow endpoint received non-JSON payload")
        return HttpResponse(status=400)

    try:
        payload, aes_key, iv = decrypt_request(body)
    except FlowDecryptionError as exc:
        logger.warning("Flow endpoint: request decryption failed: %s", exc)
        return HttpResponse(status=421)

    if payload.get("action") == "ping":
        response_data = {"data": {"status": "active"}}
    else:
        response_data = handle_flow_action(payload)

    encrypted_body = encrypt_response(response_data, aes_key, iv)
    return HttpResponse(encrypted_body, content_type="text/plain")


@csrf_exempt
@require_http_methods(["POST"])
def razorpay_webhook(request):
    """
    Razorpay's webhook — payment_link.paid / payment_link.expired /
    payment.failed, configured on the Razorpay dashboard against this URL
    with a webhook secret (separate from the API key/secret pair; see
    payments.verify_webhook_signature). Every event is verified, then
    handed to Celery — this view itself does no DB writes, so a slow/locked
    DB can never cause it to miss the 200 Razorpay expects and trigger an
    unnecessary retry.

    Signature check happens on the *raw* body before any JSON parsing —
    parsing first and re-serializing to verify would risk key-ordering or
    whitespace differences breaking the HMAC compare for reasons that have
    nothing to do with whether the payload is genuine.
    """
    signature = request.headers.get("X-Razorpay-Signature", "")
    if not payments.verify_webhook_signature(request.body, signature):
        logger.warning("Razorpay webhook signature verification failed")
        return HttpResponse(status=400)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        logger.warning("Razorpay webhook: non-JSON payload despite valid signature")
        return HttpResponse(status=400)

    event_type = data.get("event", "")
    payload = data.get("payload", {})
    payment_link_id = payload.get("payment_link", {}).get("entity", {}).get("id", "")
    payment_entity = payload.get("payment", {}).get("entity", {})
    gateway_payment_id = payment_entity.get("id", "")

    if not payment_link_id:
        # Not an event shape we act on (Razorpay sends many event types on
        # the same endpoint if you subscribe broadly) — 200 it so Razorpay
        # doesn't keep retrying something we were never going to handle.
        logger.info("Razorpay webhook: event=%s carried no payment_link id, ignoring", event_type)
        return JsonResponse({"status": "ignored"})

    process_payment_webhook.delay(event_type, payment_link_id, gateway_payment_id, payment_entity)
    return JsonResponse({"status": "received"})
