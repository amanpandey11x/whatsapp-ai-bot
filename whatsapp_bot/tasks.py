import logging
import time

from celery import shared_task
from django.conf import settings
from django.db import transaction
from django.utils import timezone
from django.utils.dateparse import parse_datetime

from . import payments
from .agents import run_agent_graph
from .intake import pending_nudge, run_guided_intake
from .models import Contact, FlowIntentClassification, Message, Order, ServiceCategory, ServicePricing
from .services import (
    ADDON_LIST_DONE_ID,
    ADDONS_FLOW_SCREEN_ID,
    CALLBACK_CTA_BODY,
    CALLBACK_CTA_BUTTON_OPTIONS,
    CALLBACK_SLOT_BODY,
    CALLBACK_SLOT_BUTTON_OPTIONS,
    BOOK_CALLBACK_ID,
    CONFIRM_SELECTION_ID,
    CONFIRMATION_LIST_BUTTON_LABEL,
    CONFIRMATION_LIST_OPTIONS,
    CSAT_BUTTON_OPTIONS,
    CSAT_PROMPT_BODY,
    EDIT_SELECTION_ID,
    FLOW_LAUNCH_CONFIRM_ID,
    FLOW_LAUNCH_EDIT_ID,
    HELP_KEYWORD,
    ITR_TYPE_FLOW_SCREEN_ID,
    KEEP_CHATTING_ID,
    LANGUAGE_BUTTON_OPTIONS,
    LANGUAGE_HOW_CAN_I_HELP,
    LANGUAGE_NAMES,
    MENU_KEYWORD,
    ORDER_STATUS_RESEND_PREFIX,
    ORDER_STATUS_SELECT_PREFIX,
    QUOTE_CANCEL_ID,
    QUOTE_CONFIRM_ID,
    QUOTE_CONFIRMATION_LIST_BUTTON_LABEL,
    QUOTE_CONFIRMATION_LIST_OPTIONS,
    RESTART_KEYWORD,
    STATUS_KEYWORD,
    SUMMARY_EVERY_N_MESSAGES,
    add_service_selection,
    addon_flow_enabled,
    auto_include_bundled_children,
    build_addon_list_body,
    build_addon_list_options,
    build_addons_screen_flow_payload,
    build_callback_confirmation,
    build_cart_nudge_message,
    build_cart_nudge_template_params,
    build_category_list_body,
    build_category_list_options,
    build_category_menu,
    build_csat_thank_you,
    build_dropoff_nudge_message,
    build_flattened_menu_body,
    build_flattened_menu_sections,
    build_order_status_reply,
    build_single_order_status_reply,
    build_service_flow_trigger_body,
    build_welcome_buttons_body,
    build_full_cutover_welcome_message,
    build_hot_lead_nudge_message,
    build_hot_lead_nudge_template_params,
    build_welcome_message,
    build_help_reply,
    build_itr_type_flow_payload,
    build_pricing_reply,
    build_quantity_reply,
    build_referral_message,
    build_resume_step_nudge,
    build_selection_complete_reply,
    build_service_flow_completion_reply,
    build_subservice_list_body,
    build_subservice_list_options,
    build_subservice_menu,
    build_topic_confirmation,
    build_topic_list_options,
    build_topic_question,
    build_warm_lead_nudge_message,
    build_warm_lead_nudge_template_params,
    classify_and_respond,
    clear_review_reached,
    flattened_menu_fits,
    is_addon_list_done,
    is_service_selected,
    itr_wizard_parents,
    mark_review_reached,
    maybe_send_payment_cta,
    notify_human_alert,
    optional_children,
    parse_itr_flow_reply,
    parse_service_flow_reply,
    record_last_quote,
    remove_service_selection,
    resend_or_recreate_payment_link,
    resolve_callback_choice,
    resolve_callback_slot,
    resolve_category_choice,
    resolve_confirmation_choice,
    resolve_csat_choice,
    resolve_flattened_service_choice,
    resolve_flow_launch_choice,
    match_keyword_command,
    resolve_language_choice,
    resolve_language_switch,
    resolve_order_status_selection,
    resolve_quantity,
    resolve_quote_confirmation,
    resolve_service_choice,
    resolve_topic_choice,
    send_service_flow,
    send_typing_indicator,
    send_whatsapp_buttons,
    send_whatsapp_flow,
    send_whatsapp_list,
    send_whatsapp_list_sections,
    send_whatsapp_message,
    send_whatsapp_template,
    service_flow_enabled,
    summarize_conversation,
    update_lead_score,
)

# How long after a completed booking to send the referral ask — long enough
# not to pile onto the CSAT/confirmation messages from the same turn, short
# enough that the booking is still fresh. See send_referral_ask.
REFERRAL_ASK_DELAY_SECONDS = 60 * 60  # 1 hour

# Re-send the referral ask at most this often per contact, in case they
# complete another booking soon after — see send_referral_ask.
REFERRAL_ASK_COOLDOWN_DAYS = 30

# Abandoned-cart nudge thresholds — see send_abandoned_cart_nudges.
CART_NUDGE_AFTER_HOURS = (2, 24)

# Drop-off nudge schedule — see send_dropoff_nudges. A separate, additional
# set of checkpoints alongside CART_NUDGE_AFTER_HOURS above (not a
# replacement), each gated to a different lead_tier audience: the earliest
# and latest touches (30m, 7d) are reserved for hot leads only — a cold
# drop-off at the welcome menu is low-intent noise, not worth a same-day
# nudge — while the two middle checkpoints (20h, 48h) widen to hot+warm.
# (stage_label, hours_after, sent_at_field, allowed lead tiers)
DROPOFF_NUDGE_STAGES = (
    ("30m", 0.5, "dropoff_nudge_30m_sent_at", {Contact.LEAD_TIER_HOT}),
    ("20h", 20, "dropoff_nudge_20h_sent_at", {Contact.LEAD_TIER_HOT, Contact.LEAD_TIER_WARM}),
    ("48h", 48, "dropoff_nudge_48h_sent_at", {Contact.LEAD_TIER_HOT, Contact.LEAD_TIER_WARM}),
    ("7d", 24 * 7, "dropoff_nudge_7d_sent_at", {Contact.LEAD_TIER_HOT}),
)

# In-conversation resume nudge — see the free-chat branch of
# process_incoming_message below and Contact.free_chat_turns_since_welcomed.
# Deliberately a message-count threshold, not a time-based one like the
# nudges above (send_abandoned_cart_nudges/send_dropoff_nudges, both Celery
# Beat periodic tasks) — this fires live, appended to the bot's own reply,
# for a contact still actively chatting who's simply never engaged the
# service Flow (or opened it and wandered off) rather than one who's gone
# quiet.
RESUME_NUDGE_AFTER_TURNS = 3

# How many recent Message rows to hand Gemini as live conversation context
# on each reply (see process_incoming_message below) — trimmed down from 10
# to shave input tokens off the single largest-volume Gemini call. Separate
# from services.SUMMARY_EVERY_N_MESSAGES/update_contact_summary's own
# 10-message window, which is unaffected by this. Cheap to try, easy to
# revert — bump back to 10 if a longer free-chat thread starts losing
# context faster than the periodic summary can catch up.
LIVE_HISTORY_WINDOW = 8

logger = logging.getLogger(__name__)


def _addon_screen(contact, parent_service):
    """('flow', body, cta, payload) or ('list', body, button_label, options)
    for the add-on picker — reused every time it's (re-)rendered: on first
    entering a wizard parent directly (a specific type already named, e.g.
    "ITR Salary Filing"), after a quantity/topic detour resolves in list
    mode, and when "Edit more" is tapped from the confirmation screen.
    Launches straight at the Flow's ADDONS screen (see
    services.build_addons_screen_flow_payload) whenever it's configured
    (services.addon_flow_enabled), skipping the now-redundant ITR_TYPE
    screen since the type is already known; falls back to the tap-list
    otherwise."""
    if addon_flow_enabled():
        return (
            "flow",
            build_addon_list_body(parent_service, contact),
            "Choose add-ons",
            build_addons_screen_flow_payload(parent_service, contact),
        )
    return (
        "list",
        build_addon_list_body(parent_service, contact),
        "View add-ons",
        build_addon_list_options(parent_service, contact),
    )


def _addon_screen_replies(contact, parent_service):
    """Unpacks _addon_screen's tagged tuple into (reply_text, reply_list,
    reply_flow) ready to assign straight into process_incoming_message's
    locals of the same name. reply_flow is (body, cta, payload, screen_id)."""
    kind, body, label_or_cta, payload = _addon_screen(contact, parent_service)
    if kind == "flow":
        return body, None, (body, label_or_cta, payload, ADDONS_FLOW_SCREEN_ID)
    return body, (body, label_or_cta, payload), None


def _enter_addon_list(contact, parent_service):
    """Puts the contact into the add-on picker state for `parent_service` and
    returns its (reply_text, reply_list, reply_flow) tuple."""
    contact.pending_parent_service_code = parent_service.service_code
    contact.pending_service_code = ""
    contact.conversation_step = Contact.STEP_AWAITING_INCOME_ITEM
    contact.save(update_fields=["pending_parent_service_code", "pending_service_code", "conversation_step"])
    return _addon_screen_replies(contact, parent_service)


def _enter_itr_type_flow(contact):
    """Puts the contact into the add-on-picker state without a specific ITR
    type chosen yet, and launches the Flow at its ITR_TYPE screen — used
    when the customer taps the ITR category itself rather than a specific
    type. Only called once services.addon_flow_enabled() and
    services.itr_wizard_parents() are both non-empty (checked by the
    caller); the type gets resolved once the Flow submits (see
    STEP_AWAITING_INCOME_ITEM's flow-reply handling), same conversation_step
    as every other add-on-picker entry point."""
    contact.pending_category_code = "itr"
    contact.pending_service_code = ""
    contact.pending_parent_service_code = ""
    contact.selected_services = []
    contact.conversation_step = Contact.STEP_AWAITING_INCOME_ITEM
    contact.save(update_fields=[
        "pending_category_code", "pending_service_code", "pending_parent_service_code",
        "selected_services", "conversation_step",
    ])
    body = "ITR Filing — pick your type, then your add-ons, all in one form:"
    return body, None, (body, "Start ITR Filing", build_itr_type_flow_payload(), ITR_TYPE_FLOW_SCREEN_ID)


def _dispatch_service_selection(contact, service):
    """(reply_text, reply_list, reply_flow, trigger_conversion_cta) for a
    top-level service that's just been picked — a wizard parent (enters the
    add-on picker), a quantity/topic clarifier, or a flat-priced service
    (quotes straight away). Shared by both routes into this decision: the
    normal two-step category-then-subservice menu (STEP_SUBSERVICE_SHOWN) and
    the flattened top-level menu (STEP_CATEGORY_SHOWN when
    services.flattened_menu_fits), so picking "ITR Salary Filing" lands in
    exactly the same place regardless of which menu it was tapped from.

    trigger_conversion_cta mirrors what used to be called trigger_csat here
    — true whenever this turn delivered a concrete, quotable outcome. It now
    fires the Book-a-callback/Keep-chatting CTA first; CSAT itself only
    follows once that micro-interaction resolves (see process_incoming_message)."""
    reply_list = None
    reply_flow = None
    trigger_conversion_cta = False

    if service.subservices.filter(is_active=True).exists():
        # A wizard parent (ITR Salary/Business/NRI Filing) — silently bundle
        # its always-included children, then show every optional add-on on
        # one screen (see _enter_addon_list).
        contact.selected_services = []
        contact.save(update_fields=["selected_services"])
        auto_include_bundled_children(contact, service)
        reply_text, reply_list, reply_flow = _enter_addon_list(contact, service)
    elif service.question_type == ServicePricing.QUESTION_QUANTITY:
        contact.pending_category_code = ""
        contact.pending_service_code = service.service_code
        contact.pending_parent_service_code = ""
        contact.conversation_step = Contact.STEP_AWAITING_QUANTITY
        contact.save(update_fields=["pending_category_code", "pending_service_code", "pending_parent_service_code", "conversation_step"])
        reply_text = service.clarifying_question
    elif service.question_type == ServicePricing.QUESTION_TOPIC:
        contact.pending_category_code = ""
        contact.pending_service_code = service.service_code
        contact.conversation_step = Contact.STEP_AWAITING_TOPIC
        contact.save(update_fields=["pending_category_code", "pending_service_code", "conversation_step"])
        reply_text = build_topic_question(service)
        reply_list = (service.clarifying_question, "Choose a topic", build_topic_list_options(service))
    else:
        # SERVICE_SELECTED -> DETAILS_COLLECTED (nothing to collect, flat
        # price) -> SUMMARY -> CONFIRM -> PAYMENT: quote it, but hold the
        # payment link until the customer actually taps Confirm (see
        # STEP_CONFIRMING_QUOTE below) rather than sending it in the same
        # turn as the quote.
        contact.pending_category_code = ""
        contact.pending_service_code = service.service_code
        contact.pending_parent_service_code = ""
        contact.conversation_step = Contact.STEP_CONFIRMING_QUOTE
        contact.save(update_fields=["pending_category_code", "pending_service_code", "pending_parent_service_code", "conversation_step"])
        add_service_selection(contact, service)
        mark_review_reached(contact)
        reply_text = build_pricing_reply(service)
        reply_list = (reply_text, QUOTE_CONFIRMATION_LIST_BUTTON_LABEL, QUOTE_CONFIRMATION_LIST_OPTIONS)

    return reply_text, reply_list, reply_flow, trigger_conversion_cta


def _top_level_menu_replies(prefix=""):
    """(reply_text, reply_list, reply_list_sections) for the very first
    services menu — flattened (every filing option directly, sectioned by
    category — see services.build_flattened_menu_sections) whenever it fits
    WhatsApp's 10-row list cap, which skips the "pick a category, then pick
    within it" round trip entirely; falls back to the existing two-step
    category menu once the catalogue outgrows one list. `prefix` is prepended
    to the body text (e.g. a language-confirmation sentence, or "That menu's
    no longer active.")."""
    if flattened_menu_fits():
        body = prefix + build_flattened_menu_body()
        return body, None, (body, "View services", build_flattened_menu_sections())
    body = prefix + build_category_list_body()
    return prefix + build_category_menu(), (body, "View services", build_category_list_options()), None


def _send_welcome_then_flow(contact):
    """Sends the classic personalized welcome line as its own message,
    immediately followed by the service-Flow trigger — under full cutover
    (service_flow_enabled()) the Flow's own interactive card only ever
    carries build_service_flow_trigger_body()'s short "let's get you
    sorted" line as its body, so without this a contact never actually
    sees a "Welcome to TaxPlan Advisors" greeting, just the flow-open
    card. Called only via _welcome_and_open_flow below.

    Deliberately build_full_cutover_welcome_message, not build_welcome_message
    — by the time this runs, contact.language is already known (asked via
    chat first, by the language gate a few branches below, before the Flow
    is ever opened — see _welcome_and_open_flow), so build_welcome_message's
    "reply 1 or 2 for a language" text would just repeat a question that's
    already been answered."""
    welcome_text = build_full_cutover_welcome_message(contact)
    send_whatsapp_message(contact.phone_number, welcome_text)
    Message.objects.create(contact=contact, direction="out", body=welcome_text)


def _welcome_and_open_flow(contact):
    """Marks a full-cutover contact welcomed and opens the mega Flow —
    shared tail end of "greet this contact for the first time this
    session," called only once contact.language is already known (the
    language gate below asks it via chat first — the Flow itself has no
    LANGUAGE screen of its own to fall back to; see
    flow_screens._entry_screen_response). Returns the Message-log text; the
    caller still sets reply_service_flow=True itself to actually trigger the
    send (see process_incoming_message's dispatch)."""
    contact.conversation_step = Contact.STEP_WELCOMED
    contact.save(update_fields=["conversation_step"])
    _send_welcome_then_flow(contact)
    return build_service_flow_trigger_body(contact)


def _resolve_pending_flow_intent_classification(contact, decision):
    """Marks this contact's most recent still-pending FlowIntentClassification
    row confirmed/rejected — called from the pending_flow_type_candidate
    branch above once the customer answers the Confirm/Edit prompt. Looks up
    "most recent pending" rather than threading an id through
    Contact.pending_flow_type_candidate because a contact only ever has one
    launch confirmation in flight at a time (same single-in-flight
    invariant pending_flow_type_candidate itself already relies on), so
    there's nothing to disambiguate. A silent no-op if somehow none is
    found (e.g. a row created before this feature shipped) — never worth
    failing the customer's actual reply over a missing analytics row."""
    classification = FlowIntentClassification.objects.filter(
        contact=contact, decision=FlowIntentClassification.DECISION_PENDING,
    ).order_by("-created_at").first()
    if classification:
        classification.decision = decision
        classification.resolved_at = timezone.now()
        classification.save(update_fields=["decision", "resolved_at"])


@shared_task(time_limit=30, soft_time_limit=25)
def process_incoming_message(contact_id, text, received_at=None, wa_message_id=None):
    """
    Runs off the request/response cycle so the webhook can reply to Meta instantly.
    Generates an AI response, saves it, and sends it back to the user.

    Logs a per-stage timing breakdown (queue wait / Gemini call / WhatsApp send)
    so latency can be diagnosed from the worker logs instead of guessed at.
    """
    task_start = time.monotonic()

    if received_at:
        queue_wait = (timezone.now() - parse_datetime(received_at)).total_seconds()
        logger.info("[timing] contact=%s queue_wait=%.2fs", contact_id, queue_wait)

    contact = Contact.objects.get(id=contact_id)
    normalized = text.strip().lower()
    # Resolves a likely typo of restart/help/menu/status (e.g. "resatrt",
    # "halp", "meun") to the real keyword too — see
    # services.match_keyword_command. None for anything that isn't a close
    # match, same as before.
    keyword_command = match_keyword_command(normalized)

    # Human-handoff pause feature removed: the bot used to go silent on a
    # contact for services.HUMAN_HANDOFF_PAUSE_HOURS after HELP / a callback
    # booking (via Contact.bot_paused_until), on the assumption a human agent
    # was watching the handoff queue during that window. Per explicit
    # request, the bot now keeps auto-replying even after a handoff is
    # triggered — create_handoff_ticket/notify_human_alert below still fire
    # normally, so staff still gets notified, but the customer is never left
    # hanging on an unstaffed queue. Contact.bot_paused_until itself is kept
    # (unused going forward) rather than migrated away, so historical rows
    # aren't touched.

    # Shows the "typing..." bubble on WhatsApp and marks the message read.
    # Meta caps this at ~25s / clears it automatically once we actually send
    # the reply below, so no separate "stop typing" call is needed.
    if wa_message_id:
        send_typing_indicator(wa_message_id)

    reply_text = None  # set by whichever branch below handles this message;
    # if still None after the scripted branches, falls through to free-chat.
    # reply_buttons / reply_list: set alongside reply_text when it should be
    # sent as an interactive message instead of plain text. reply_buttons is
    # (body_text, options) for <=3 choices (reply buttons); reply_list is
    # (body_text, button_label, options) for menus with more options (a
    # scrollable list, up to 10 rows). Checked in that order at send time —
    # reply_text always still gets saved to the DB either way.
    # reply_list_sections: (body_text, button_label, sections) for the
    # flattened top-level menu (see _top_level_menu_replies) — a list message
    # with more than one titled section. reply_flow: (body_text, flow_cta,
    # flow_action_payload) for the add-on Flow (see _addon_screen_replies) —
    # a native form screen instead of a chat message. Checked alongside
    # reply_buttons/reply_list at send time, in the order declared below.
    reply_buttons = None
    reply_list = None
    reply_list_sections = None
    reply_flow = None
    # Set instead of the above when the mega Flow (see services.send_service_flow)
    # should open — a different send call than reply_flow (the static,
    # inline-data add-on Flow), see the full-cutover branch below.
    reply_service_flow = False
    # Set instead of reply_service_flow when the mega Flow should open
    # deep-linked straight at one routing_model screen (e.g. "ITR_TYPE")
    # rather than SERVICE — the natural-language intent classifier's
    # auto-launch path (see agents.sales_cart_node's launch_flow_candidate
    # branch) and, for any row created before that change, the legacy
    # pending_flow_type_candidate confirm branch below.
    reply_service_flow_target = None
    # The Flow-opening message's own body text, alongside
    # reply_service_flow_target — set only by the auto-launch path above, so
    # Gemini's own conversational reply still reaches the customer (on the
    # Flow message itself, since there's no separate confirm message
    # anymore). None everywhere else, which send_service_flow treats as "use
    # the generic greeting."
    reply_service_flow_body = None
    # Set alongside reply_text whenever this turn delivers a concrete outcome
    # (a price quote, a completed add-on summary, a booking confirmation) —
    # folds that reply_text into the Pay Now CTA-URL message as one merged
    # outbound message when a payment link is eligible (see
    # services.maybe_send_payment_cta), sent in place of the plain-text
    # reply below rather than as a second message after it. No longer fires
    # a Book-a-callback/support ask inline — HELP still hands off to a
    # human (build_help_reply), and a customer can still type "callback"/
    # "book_callback" any time (see resolve_callback_choice below), but
    # nothing currently proactively OFFERS that button anymore since
    # send_hot_lead_nudges (the last thing that did) was removed — see
    # services.CALLBACK_CTA_BODY/CALLBACK_CTA_BUTTON_OPTIONS, now unused.
    trigger_conversion_cta = False
    # Set once a customer explicitly asks for a callback (a picked slot, or
    # the legacy "keep chatting"/"pay" text fallback below) — this is what
    # fires the CSAT follow-up, per services.CSAT_BUTTON_OPTIONS. Since
    # trigger_conversion_cta no longer shows a callback button on every
    # confirm, this now mostly fires from the callback-slot path (a customer
    # who typed HELP, or was nudged later and asked to book one).
    trigger_csat = False
    # Set when a branch below (currently just the order-status "resend
    # payment link" action) already sent its own outbound WhatsApp message
    # directly, same "already handled, skip the generic send below" role as
    # payment_cta_sent further down — see the send-dispatch block.
    special_action_sent = False

    # Mid-conversation language switch (e.g. "reply in English please") — the
    # onboarding gate below only ever sets contact.language once; without
    # this, Gemini can verbally agree to switch while the stored language
    # (and therefore every future system_instruction) stays stale, causing
    # replies to flip-flop between languages independent of what was asked.
    if contact.language:
        switched = resolve_language_switch(text)
        if switched and switched != contact.language:
            contact.language = switched
            contact.save(update_fields=["language"])

    csat_rating = resolve_csat_choice(text)
    if csat_rating:
        # Deterministic, works anytime, independent of conversation_step —
        # a CSAT tap can arrive while the contact is mid-menu or mid-wizard,
        # and shouldn't be swallowed by whatever branch is active there.
        contact.satisfaction_rating = csat_rating
        contact.satisfaction_rating_at = timezone.now()
        contact.save(update_fields=["satisfaction_rating", "satisfaction_rating_at"])
        reply_text = build_csat_thank_you(csat_rating)
        if csat_rating == Contact.SATISFACTION_BAD:
            # A bad rating shouldn't just sit in the DB — flag it for a human
            # to actually look at (see services.notify_human_alert).
            notify_human_alert(contact, "bad_csat")

    elif resolve_callback_choice(text) == BOOK_CALLBACK_ID:
        # Anytime-resolvable, same tier as CSAT above — the callback CTA can
        # be tapped regardless of what conversation_step the contact is
        # otherwise in.
        reply_text = CALLBACK_SLOT_BODY
        reply_buttons = (CALLBACK_SLOT_BODY, CALLBACK_SLOT_BUTTON_OPTIONS)

    elif resolve_callback_choice(text) == KEEP_CHATTING_ID:
        # Id/constant name kept as KEEP_CHATTING_ID (see services.py) — no
        # button offers this anymore (CALLBACK_CTA_BUTTON_OPTIONS dropped
        # "I'll pay now"), this only still fires for a customer who types
        # "pay"/"chat" from muscle memory. Reply acknowledges the payment
        # link already sent rather than inviting free-chat.
        reply_text = "Sounds good — the payment link above is ready whenever you are. Let me know if you need anything else!"
        trigger_csat = True

    elif resolve_callback_slot(text):
        slot = resolve_callback_slot(text)
        contact.callback_slot = slot
        contact.callback_requested_at = timezone.now()
        contact.save(update_fields=["callback_slot", "callback_requested_at"])
        reply_text = build_callback_confirmation(contact, slot)
        notify_human_alert(contact, "callback_requested", detail=f"slot={slot}")
        trigger_csat = True

    elif keyword_command == STATUS_KEYWORD:
        # Same "anytime-resolvable, independent of conversation_step" tier
        # as CSAT/callback above — pure DB read, no LLM involved (see
        # services.build_order_status_reply).
        reply_text, reply_buttons, reply_list = build_order_status_reply(contact)

    elif text.startswith(ORDER_STATUS_SELECT_PREFIX):
        # A tapped row from the multi-order list build_order_status_reply
        # sent above — resolve_order_status_selection scopes the lookup to
        # contact.orders, so an id belonging to another contact can never
        # match here.
        selected_order = resolve_order_status_selection(text, contact)
        if selected_order:
            reply_text, reply_buttons = build_single_order_status_reply(selected_order)
        else:
            reply_text = "Couldn't find that order — reply STATUS to see your orders again."

    elif text.startswith(ORDER_STATUS_RESEND_PREFIX):
        # "Get payment link" tap from build_single_order_status_reply.
        # Re-normalize onto ORDER_STATUS_SELECT_PREFIX so the same ownership
        # -scoped lookup applies — no separate resolver needed for this prefix.
        resend_order = resolve_order_status_selection(
            text.replace(ORDER_STATUS_RESEND_PREFIX, ORDER_STATUS_SELECT_PREFIX, 1), contact,
        )
        if resend_order and resend_order.status in (
            Order.STATUS_CREATED, Order.STATUS_FAILED, Order.STATUS_EXPIRED,
        ) and resend_or_recreate_payment_link(resend_order):
            # The CTA-URL message itself was already sent by
            # resend_or_recreate_payment_link — reply_text here is only
            # logged to the conversation history below, same "logged even
            # though sent a different way" pattern as trigger_conversion_cta's
            # summary text, not sent again as its own plain-text message
            # (see special_action_sent in the send-dispatch block).
            special_action_sent = True
            reply_text = f"Sent a fresh payment link for order #{resend_order.id}."
        else:
            reply_text = "Couldn't send a fresh link — reply HELP and our team will sort it out."

    elif parse_service_flow_reply(text) is not None:
        # The mega Flow's terminal screen (REVIEW/COMING_SOON/ERROR — see
        # flow_screens.py) lands back here as a normal inbound message, same
        # convention as the ITR Flow below. The cart itself was already
        # saved screen-by-screen server-side during the Flow session, so
        # there's nothing left to apply here — just the confirmation reply,
        # deterministic and independent of conversation_step since the
        # contact's chat step never moved while they were inside the Flow.
        flow_reply = parse_service_flow_reply(text)
        reply_text = build_service_flow_completion_reply(flow_reply, contact)
        if flow_reply["kind"] == "error":
            # Not a real completion — don't mark STEP_PRICING_SHOWN (that
            # flag is what suppresses the auto-relaunch below at
            # `service_flow_enabled() and conversation_step !=
            # STEP_PRICING_SHOWN`; setting it here would leave the one
            # customer who just hit a failure unable to get the Flow back
            # without typing "restart" themselves — worst possible customer
            # to drop). Instead reopen it immediately, same turn.
            reply_service_flow = True
        else:
            contact.conversation_step = Contact.STEP_PRICING_SHOWN
            contact.save(update_fields=["conversation_step"])
            # A real completion — the cart is confirmed, so this contact is no
            # longer "sitting on an unconfirmed review" (see
            # services.mark_review_reached, set when the Flow's REVIEW screen
            # was rendered) and a referral ask is worth sending shortly after.
            clear_review_reached(contact)
            _maybe_schedule_referral_ask(contact)
            # selected_services is still the cart just confirmed (cleared
            # only on a later RESTART) — safe to sum here for the link amount.
            record_last_quote(contact)
        trigger_conversion_cta = flow_reply["kind"] == "review"

    elif contact.pending_flow_type_candidate and keyword_command not in (
        RESTART_KEYWORD, MENU_KEYWORD, HELP_KEYWORD,
    ):
        # LEGACY as of the auto-launch change (see agents.sales_cart_node's
        # launch_flow_candidate branch) — nothing sets pending_flow_type_candidate
        # anymore on a new turn, since a launch_flow_candidate classification
        # now opens the deep-linked Flow immediately instead of asking a
        # Confirm/Edit question first. Kept only so a contact who already had
        # this set before that deploy still gets a sane answer instead of
        # falling through to some other branch mid-flight.
        #
        # Anytime-resolvable, same tier as CSAT/callback-choice above — NOT
        # gated on conversation_step (see Contact.pending_flow_type_candidate's
        # help_text for why: a new conversation_step value here would get
        # swallowed by the FULL CUTOVER branch below before ever reaching a
        # handler).
        #
        # keyword_command escape hatch: without it, a customer whose message
        # got mis-classified into the wrong flow_type (or who simply changed
        # their mind) had NO way out short of tapping this exact prompt's own
        # Yes/No buttons — not even RESTART/MENU/HELP worked, since every
        # other reply just re-showed the identical "Want me to open X for you
        # now?" prompt forever (see resolve_flow_launch_choice: anything that
        # isn't a yes/no-ish word falls into the "unparseable" branch below,
        # which deliberately leaves pending_flow_type_candidate untouched).
        # Same carve-out as active_intake_flow_type's branch just below.
        flow_type = contact.pending_flow_type_candidate
        launch_choice = resolve_flow_launch_choice(text)

        if launch_choice == FLOW_LAUNCH_CONFIRM_ID:
            contact.pending_flow_type_candidate = ""
            contact.save(update_fields=["pending_flow_type_candidate"])
            logger.info(
                "[flow-intent] contact=%s flow_type=%s decision=confirmed", contact.id, flow_type,
            )
            _resolve_pending_flow_intent_classification(contact, FlowIntentClassification.DECISION_CONFIRMED)
            reply_text = "Great — opening it for you now."
            reply_service_flow_target = flow_type
            # An explicit re-engagement with the Flow, same reasoning as
            # MENU's reset above — this contact is no longer "still ignoring
            # the pending step."
            if contact.free_chat_turns_since_welcomed or contact.resume_nudge_sent_at:
                contact.free_chat_turns_since_welcomed = 0
                contact.resume_nudge_sent_at = None
                contact.save(update_fields=["free_chat_turns_since_welcomed", "resume_nudge_sent_at"])
        elif launch_choice == FLOW_LAUNCH_EDIT_ID:
            contact.pending_flow_type_candidate = ""
            contact.save(update_fields=["pending_flow_type_candidate"])
            logger.info(
                "[flow-intent] contact=%s flow_type=%s decision=rejected", contact.id, flow_type,
            )
            _resolve_pending_flow_intent_classification(contact, FlowIntentClassification.DECISION_REJECTED)
            reply_text = "No worries — what would you like to do instead?"
        else:
            # Unparseable — but deliberately NOT "re-show the same Confirm/
            # Edit choice forever" (that used to be the behavior here, and
            # it's a trap: a customer who asks a real question, states their
            # name, or makes an unrelated request instead of answering
            # yes/no would just get the identical "Want me to open X for you
            # now?" prompt back on every single turn afterward, since none
            # of those replies ever match resolve_flow_launch_choice either
            # — see the 2026-09-10 stuck-in-TDS-filing trace). Treat an
            # unparseable reply as an implicit decline: resolve/clear the
            # pending candidate now, then leave reply_text=None so this same
            # message falls through to the free-chat branch below and gets
            # answered for real, same "defer to free-chat rather than loop
            # the same prompt" convention as active_intake_flow_type's
            # interruption handling just below.
            contact.pending_flow_type_candidate = ""
            contact.save(update_fields=["pending_flow_type_candidate"])
            logger.info(
                "[flow-intent] contact=%s flow_type=%s decision=rejected (unparseable reply, deferred to free-chat)",
                contact.id, flow_type,
            )
            _resolve_pending_flow_intent_classification(contact, FlowIntentClassification.DECISION_REJECTED)

    elif contact.active_intake_flow_type and keyword_command not in (
        RESTART_KEYWORD, MENU_KEYWORD, HELP_KEYWORD,
    ):
        # LEGACY, same as the pending_flow_type_candidate branch above — a
        # direct_service_request classification now auto-launches the
        # deep-linked Flow directly (agents.sales_cart_node), same as
        # launch_flow_candidate, instead of starting this chat-only guided
        # intake (agents.guided_intake_node is dead code now — see its
        # docstring). Kept only to finish out a wizard already in progress
        # for a contact who has active_intake_flow_type set from before that
        # change; nothing sets it on a fresh turn anymore.
        #
        # Chat-only guided intake (see intake.run_guided_intake). Anytime-
        # resolvable, same tier as pending_flow_type_candidate just above —
        # but this is a wholly separate mechanism from it: the WhatsApp
        # Flow's own launch path is never touched by this.
        reply_text, reply_buttons, reply_list, intake_cta = run_guided_intake(contact, text)
        if intake_cta:
            trigger_conversion_cta = True
        # reply_text stays None for two different reasons, both meaning
        # "fall through to the free-chat branch below for this turn": (1)
        # this message was an interruption — it didn't answer the pending
        # slot, so active_intake_flow_type is deliberately left set and the
        # exact same question is recomputed and re-asked next turn; (2)
        # nothing was actually missing (a stale marker), which
        # run_guided_intake already cleared itself.

    elif keyword_command == HELP_KEYWORD:
        # Deterministic human handoff — works anytime, doesn't depend on Gemini
        # deciding to escalate. Sticky flag: highest-intent signal for lead scoring.
        help_update_fields = []
        if not contact.has_requested_help:
            contact.has_requested_help = True
            help_update_fields.append("has_requested_help")
        if contact.active_intake_flow_type:
            contact.active_intake_flow_type = ""
            help_update_fields.append("active_intake_flow_type")
        contact.save(update_fields=help_update_fields)
        reply_text = build_help_reply()
        notify_human_alert(contact, "help_requested")

    elif keyword_command == RESTART_KEYWORD:
        # Full reset — re-shows build_welcome_message() exactly like a brand new contact,
        # lead score included. Always ends by asking language via the chat
        # list below (contact.language cleared just below, same as any
        # brand-new contact) — never straight back into the Flow, full-
        # cutover or not, since the language gate is unconditional now (see
        # its own docstring further down).
        if contact.pending_flow_type_candidate:
            # Clear the stale flow-intent confirm prompt too — without this,
            # the very next message (which won't match yes/no) would fall
            # straight back into the pending_flow_type_candidate trap above,
            # making RESTART look like it did nothing.
            logger.info(
                "[flow-intent] contact=%s flow_type=%s decision=rejected (restart)",
                contact.id, contact.pending_flow_type_candidate,
            )
            _resolve_pending_flow_intent_classification(contact, FlowIntentClassification.DECISION_REJECTED)
        contact.pending_flow_type_candidate = ""
        contact.language = ""
        contact.conversation_step = ""
        contact.pending_category_code = ""
        contact.pending_service_code = ""
        contact.pending_parent_service_code = ""
        contact.active_intake_flow_type = ""
        contact.selected_services = []
        contact.has_answered_clarifier = False
        contact.has_requested_help = False
        contact.flow_furthest_screen = ""
        contact.lead_score = 0
        contact.lead_tier = Contact.LEAD_TIER_COLD
        contact.predicted_next_action = ""
        contact.satisfaction_rating = ""
        contact.satisfaction_rating_at = None
        # Session/funnel state — not first_time_filer, a stable fact about
        # the contact rather than something this conversation session owns.
        contact.callback_slot = ""
        contact.callback_requested_at = None
        contact.needs_human_review = False
        contact.needs_human_review_reason = ""
        contact.needs_human_review_at = None
        contact.review_reached_at = None
        contact.cart_nudge_2h_sent_at = None
        contact.cart_nudge_24h_sent_at = None
        contact.free_chat_turns_since_welcomed = 0
        contact.resume_nudge_sent_at = None
        # An explicit re-engagement with the bot itself — clears any pending
        # human-handoff pause the same way RESTART clears everything else.
        contact.bot_paused_until = None
        contact.save(update_fields=[
            "pending_flow_type_candidate", "language", "conversation_step", "pending_category_code",
            "pending_service_code", "pending_parent_service_code", "active_intake_flow_type", "selected_services",
            "has_answered_clarifier", "has_requested_help", "flow_furthest_screen", "lead_score", "lead_tier",
            "predicted_next_action", "satisfaction_rating", "satisfaction_rating_at",
            "callback_slot", "callback_requested_at", "needs_human_review", "needs_human_review_reason",
            "needs_human_review_at", "review_reached_at", "cart_nudge_2h_sent_at", "cart_nudge_24h_sent_at",
            "free_chat_turns_since_welcomed", "resume_nudge_sent_at", "bot_paused_until",
        ])
        reply_text = build_welcome_message(contact)
        reply_buttons = (build_welcome_buttons_body(contact), LANGUAGE_BUTTON_OPTIONS)

    elif keyword_command == MENU_KEYWORD and service_flow_enabled() and contact.language:
        # Deliberately lighter than RESTART: reopens the Flow straight at
        # SERVICE (its entry screen — there's no LANGUAGE screen to skip
        # past anymore, see send_service_flow) without touching language,
        # lead_score, satisfaction rating, or history flags. For a returning
        # customer who just wants the menu again, RESTART's full wipe is the
        # wrong tool — this is the "show me the services again" they're
        # actually asking for.
        #
        # `and contact.language` guard: without it, a brand-new contact
        # whose very first-ever message is "menu" would reach this branch
        # before the unconditional language gate further down ever runs —
        # and since the Flow itself no longer has anywhere to ask language,
        # that contact's language would silently never get collected. Failing
        # this condition just falls through to that gate instead, same as
        # any other first message would.
        reply_text = "Sure — here's our menu again:"
        reply_service_flow = True
        # An explicit re-engagement with the Flow — same reset as RESTART's,
        # scoped to just these fields since MENU deliberately leaves
        # everything else (language, lead_score, cart, etc.) untouched. Also
        # clears any pending human-handoff pause, same reasoning as RESTART.
        menu_update_fields = []
        if contact.free_chat_turns_since_welcomed or contact.resume_nudge_sent_at:
            contact.free_chat_turns_since_welcomed = 0
            contact.resume_nudge_sent_at = None
            menu_update_fields += ["free_chat_turns_since_welcomed", "resume_nudge_sent_at"]
        if contact.bot_paused_until:
            contact.bot_paused_until = None
            menu_update_fields.append("bot_paused_until")
        if contact.active_intake_flow_type:
            contact.active_intake_flow_type = ""
            menu_update_fields.append("active_intake_flow_type")
        if contact.pending_flow_type_candidate:
            # Same trap as RESTART's reset above — without clearing this,
            # the next message that isn't yes/no falls straight back into
            # the pending_flow_type_candidate branch and re-shows the old
            # "Want me to open X for you now?" prompt instead of the menu
            # MENU just sent.
            logger.info(
                "[flow-intent] contact=%s flow_type=%s decision=rejected (menu)",
                contact.id, contact.pending_flow_type_candidate,
            )
            _resolve_pending_flow_intent_classification(contact, FlowIntentClassification.DECISION_REJECTED)
            contact.pending_flow_type_candidate = ""
            menu_update_fields.append("pending_flow_type_candidate")
        if menu_update_fields:
            contact.save(update_fields=menu_update_fields)

    elif not contact.language:
        # Ask language via a plain WhatsApp chat list, BEFORE anything else
        # — Flow open or free-chat intent classification alike, legacy
        # wizard or full cutover alike. Blocking: nothing else happens for
        # this contact until they answer it (same as the old legacy-only
        # version of this gate always did) — now unconditional rather than
        # gated off under service_flow_enabled(), because relying on the
        # Flow's own LANGUAGE screen to capture this turned out not to work:
        # a contact who reaches a confirmed flow-intent purely through free
        # chat (agents.sales_cart_node's Confirm/Edit prompt) never
        # necessarily opens the Flow itself, so contact.language could stay
        # blank indefinitely — see services.send_service_flow's docstring on
        # the 2026-09 trace where a confirmed "Yes, start" reopened at
        # LANGUAGE instead of the confirmed service, forcing the customer to
        # repeat themselves. Asking here first means every contact's
        # language is known from their very first real message, not just
        # the ones who happen to tap through the Flow's own LANGUAGE screen.
        chosen = resolve_language_choice(text)
        if chosen:
            contact.language = chosen
            # A language/menu restart must not carry a parent service from an
            # earlier ITR flow into the next selection.
            contact.pending_service_code = ""
            contact.pending_parent_service_code = ""
            contact.selected_services = []
            update_fields = ["language", "pending_service_code", "pending_parent_service_code", "selected_services"]
            if service_flow_enabled():
                contact.save(update_fields=update_fields)
                # Straight into the one-shot welcome + mega-Flow open, same
                # turn — no separate round-trip waiting for another message.
                # The Flow itself never shows its own LANGUAGE screen now
                # (flow_screens._entry_screen_response only falls back to it
                # for a contact somehow still without a language at all).
                reply_text = _welcome_and_open_flow(contact)
                reply_service_flow = True
            else:
                contact.conversation_step = Contact.STEP_CATEGORY_SHOWN
                update_fields.append("conversation_step")
                contact.save(update_fields=update_fields)
                # Language confirmation + the "how can I help" question +
                # services menu, all in one message rather than a separate
                # round-trip — LANGUAGE_HOW_CAN_I_HELP is asked in whichever
                # language was just chosen, not left in English regardless.
                reply_text, reply_list, reply_list_sections = _top_level_menu_replies(
                    f"Great! I'll continue in {LANGUAGE_NAMES[chosen]}. "
                    f"{LANGUAGE_HOW_CAN_I_HELP[chosen]}\n\n"
                )
        else:
            # First message, or they haven't picked a language yet — show the
            # menu instead of letting the model freewheel without that context.
            reply_text = build_welcome_message(contact)
            reply_buttons = (build_welcome_buttons_body(contact), LANGUAGE_BUTTON_OPTIONS)

    elif service_flow_enabled() and (
        not contact.conversation_step
        or contact.conversation_step in (
            # The legacy pre-cutover chat-wizard steps (see the elif chain
            # further below) — only reachable today if a contact got stuck
            # in one of them from before WHATSAPP_SERVICE_FLOW_ID was ever
            # set. Deliberately NOT the broader "conversation_step not in
            # (STEP_WELCOMED, STEP_PRICING_SHOWN)" this used to be — see the
            # 2026-09-11 trace where a contact mid an active mega-Flow cart
            # session (STEP_CONFIRMING_SELECTION, having just added House
            # Property) asked "I have 3 house property, want to know the
            # pricing of gst" and got a brand-new "Welcome to TaxPlan
            # Advisors!" + re-opened Flow instead of an answer — that
            # broader check matched ANY in-progress mega-Flow step
            # (STEP_AWAITING_INCOME_ITEM, STEP_CONFIRMING_SELECTION,
            # STEP_CONFIRMING_QUOTE all instantly qualified, none of them
            # legacy), discarding a real, substantive message from someone
            # already mid-conversation.
            Contact.STEP_LANGUAGE_SELECTED, Contact.STEP_CATEGORY_SHOWN,
            Contact.STEP_SUBSERVICE_SHOWN, Contact.STEP_AWAITING_QUANTITY,
            Contact.STEP_AWAITING_TOPIC,
        )
    ):
        # FULL CUTOVER (see docs/whatsapp_flows/README.md): once
        # WHATSAPP_SERVICE_FLOW_ID is set, the mega Flow replaces the
        # welcome/category/sub-service chat wizard below as the *greeting*
        # — matched here, after the deterministic anytime checks above
        # (CSAT/flow-completion/HELP/RESTART) and the language gate just
        # above (language is always already known by the time this branch
        # runs), for a contact stuck mid a legacy pre-cutover step
        # (STEP_CATEGORY_SHOWN etc. from before WHATSAPP_SERVICE_FLOW_ID was
        # ever set — those branches are dead under cutover, so this sweeps
        # them in too rather than leaving them stuck), or one whose
        # language was somehow already known without ever having gone
        # through the gate above (e.g. imported/legacy data) — anyone who
        # hasn't been sent it yet this session. A brand-new contact or one
        # just reset by RESTART reaches send_service_flow via the language
        # gate directly instead, never this branch, since their language is
        # blank at that point.
        #
        # Deliberately a ONE-SHOT trigger, not "every message until they
        # finish the Flow" — conversation_step is stamped STEP_WELCOMED
        # immediately below specifically so this doesn't match again. A
        # contact who ignores the Flow and just asks a real question instead
        # falls through to free-chat/Gemini like any other message, rather
        # than getting the Flow re-shoved at them every single turn (that
        # was the bug: a contact could never get a normal reply until they
        # completed the Flow). MENU/RESTART above remain the explicit way to
        # reopen it deliberately.
        reply_text = _welcome_and_open_flow(contact)
        reply_service_flow = True

    elif not contact.conversation_step:
        # Language already set but no scripted step recorded — either a
        # returning contact from before this flow existed, or one reset by a
        # schema migration (see 2026-08-22 stuck-contact trace). Re-enter the
        # flow at the top-level menu instead of silently free-chatting forever.
        contact.conversation_step = Contact.STEP_CATEGORY_SHOWN
        contact.save(update_fields=["conversation_step"])
        reply_text, reply_list, reply_list_sections = _top_level_menu_replies()

    elif contact.conversation_step == Contact.STEP_CATEGORY_SHOWN:
        # Flattened menu: when every category's services fit in one 10-row
        # list (services.flattened_menu_fits), the menu shown here was
        # already every actual filing option directly (see
        # _top_level_menu_replies) — so try resolving straight against a
        # top-level service first and, if that hits, skip the "which
        # category?" step entirely via the shared dispatcher. Only when that
        # doesn't match (or the catalogue outgrew the flattened list, so a
        # plain category menu was shown instead) does this fall back to
        # matching a category and showing its sub-service menu.
        service = resolve_flattened_service_choice(text) if flattened_menu_fits() else None
        if service is not None:
            reply_text, reply_list, reply_flow, trigger_conversion_cta = _dispatch_service_selection(contact, service)
        else:
            category = resolve_category_choice(text)
            if category is None:
                # Didn't match anything — don't get them stuck, fall through
                # to free-chat instead of re-showing the menu forever.
                contact.conversation_step = Contact.STEP_PRICING_SHOWN
                contact.save(update_fields=["conversation_step"])
            elif category.category_code == "itr" and addon_flow_enabled() and itr_wizard_parents():
                # ITR specifically skips the sub-service menu entirely once
                # the Flow is configured — type selection AND add-ons both
                # happen inside the Flow's two screens (see
                # _enter_itr_type_flow), rather than a chat list tap into
                # sub-service_shown first.
                reply_text, reply_list, reply_flow = _enter_itr_type_flow(contact)
            else:
                contact.pending_category_code = category.category_code
                # Entering a category always starts at its top-level menu.
                # Without clearing this, a completed Salary Filing flow could
                # make a later Business Filing tap resolve against Salary's
                # child items.
                contact.pending_service_code = ""
                contact.pending_parent_service_code = ""
                contact.selected_services = []
                contact.conversation_step = Contact.STEP_SUBSERVICE_SHOWN
                contact.save(update_fields=["pending_category_code", "pending_service_code", "pending_parent_service_code", "selected_services", "conversation_step"])
                reply_text = build_subservice_menu(category)
                reply_list = (
                    build_subservice_list_body(category), "View options",
                    build_subservice_list_options(category),
                )

    elif contact.conversation_step == Contact.STEP_SUBSERVICE_SHOWN:
        category = ServiceCategory.objects.filter(category_code=contact.pending_category_code).first()
        service = resolve_service_choice(text, category=category) if category else None

        if service is None:
            # Didn't match a sub-service — fall through to free-chat.
            contact.pending_category_code = ""
            contact.pending_service_code = ""
            contact.conversation_step = Contact.STEP_PRICING_SHOWN
            contact.save(update_fields=["pending_category_code", "pending_service_code", "conversation_step"])
        else:
            reply_text, reply_list, reply_flow, trigger_conversion_cta = _dispatch_service_selection(contact, service)

    elif contact.conversation_step == Contact.STEP_AWAITING_INCOME_ITEM:
        # A submit from the ITR Flow (see services.build_itr_type_flow_payload
        # / build_addons_screen_flow_payload) is checked BEFORE looking at
        # contact.pending_parent_service_code — unlike every other flow-reply
        # case, this one can arrive with no parent chosen yet (a fresh entry
        # via the ITR category itself, see _enter_itr_type_flow), so itr_type
        # from the reply itself is the source of truth, not the pending field.
        flow_reply = parse_itr_flow_reply(text)

        if flow_reply is not None:
            parent_service = ServicePricing.objects.filter(
                service_code=flow_reply["itr_type"], is_active=True,
            ).first()
            if not parent_service:
                # Unknown/stale itr_type — fall through to free-chat rather
                # than getting stuck on a dead submission.
                contact.pending_category_code = ""
                contact.pending_service_code = ""
                contact.pending_parent_service_code = ""
                contact.conversation_step = Contact.STEP_PRICING_SHOWN
                contact.save(update_fields=["pending_category_code", "pending_service_code", "pending_parent_service_code", "conversation_step"])
            else:
                # One submit — apply every change (including both quantity
                # fields) in a single backend pass, then land straight on the
                # review screen. No further round trip needed, unlike the
                # tap-list which still detours for a quantity item.
                contact.pending_parent_service_code = parent_service.service_code
                contact.save(update_fields=["pending_parent_service_code"])
                auto_include_bundled_children(contact, parent_service)  # no-op if already applied

                selected_codes = set(flow_reply["addons"])
                for child in optional_children(parent_service):
                    if child.question_type == ServicePricing.QUESTION_QUANTITY:
                        continue  # handled via the two dedicated number fields below
                    wanted = child.service_code in selected_codes
                    already = is_service_selected(contact, child.service_code)
                    if wanted and not already:
                        add_service_selection(contact, child)
                    elif not wanted and already:
                        remove_service_selection(contact, child)

                # Quantity items: the Flow only knows "house property" /
                # "partnership" generically (its two dedicated number
                # fields), not this parent's specific service_code for them
                # — matched by name since every wizard parent names its
                # quantity children identically.
                for child in optional_children(parent_service):
                    if child.question_type != ServicePricing.QUESTION_QUANTITY:
                        continue
                    name = child.display_name.lower()
                    if "house" in name:
                        qty = flow_reply["house_property_qty"]
                    elif "partnership" in name:
                        qty = flow_reply["partnership_qty"]
                    else:
                        qty = None
                    if qty and qty > 0:
                        add_service_selection(contact, child, qty)
                    elif is_service_selected(contact, child.service_code):
                        remove_service_selection(contact, child)

                contact.conversation_step = Contact.STEP_CONFIRMING_SELECTION
                contact.save(update_fields=["conversation_step"])
                mark_review_reached(contact)
                # Snapshot the total shown on this (still unconfirmed) review
                # screen too — not just at final confirmation — so the
                # abandoned-cart nudge task always has the right amount to
                # attach a payment link to, even for a cart never confirmed.
                record_last_quote(contact)
                reply_text = build_selection_complete_reply(contact)
                reply_list = (reply_text, CONFIRMATION_LIST_BUTTON_LABEL, CONFIRMATION_LIST_OPTIONS)

        else:
            parent_service = ServicePricing.objects.filter(
                service_code=contact.pending_parent_service_code, is_active=True,
            ).first()

            if not parent_service:
                # Lost track of the wizard state (or the ITR_TYPE screen
                # hasn't been submitted yet) — fall through to free-chat
                # rather than getting stuck on a dead list.
                contact.pending_service_code = ""
                contact.pending_parent_service_code = ""
                contact.conversation_step = Contact.STEP_PRICING_SHOWN
                contact.save(update_fields=["pending_service_code", "pending_parent_service_code", "conversation_step"])
            elif is_addon_list_done(text):
                contact.conversation_step = Contact.STEP_CONFIRMING_SELECTION
                contact.save(update_fields=["conversation_step"])
                mark_review_reached(contact)
                # Snapshot the total shown on this (still unconfirmed) review
                # screen too — not just at final confirmation — so the
                # abandoned-cart nudge task always has the right amount to
                # attach a payment link to, even for a cart never confirmed.
                record_last_quote(contact)
                reply_text = build_selection_complete_reply(contact)
                reply_list = (reply_text, CONFIRMATION_LIST_BUTTON_LABEL, CONFIRMATION_LIST_OPTIONS)
            else:
                item = resolve_service_choice(text, category=parent_service.category, parent_service=parent_service)
                if item is None:
                    # Smooth fallback: re-show the same picker rather than
                    # dropping to free-chat or silently ignoring an
                    # unrecognised tap/typo.
                    body_text, reply_list, reply_flow = _addon_screen_replies(contact, parent_service)
                    reply_text = "Didn't catch that — pick an add-on below, or tap ✅ Confirm & Done.\n\n" + body_text
                    if reply_list:
                        reply_list = (reply_text, reply_list[1], reply_list[2])
                    if reply_flow:
                        reply_flow = (reply_text, reply_flow[1], reply_flow[2], reply_flow[3])
                elif item.question_type == ServicePricing.QUESTION_QUANTITY and not is_service_selected(contact, item.service_code):
                    contact.pending_service_code = item.service_code
                    contact.conversation_step = Contact.STEP_AWAITING_QUANTITY
                    contact.save(update_fields=["pending_service_code", "conversation_step"])
                    reply_text = item.clarifying_question
                elif item.question_type == ServicePricing.QUESTION_TOPIC and not is_service_selected(contact, item.service_code):
                    contact.pending_service_code = item.service_code
                    contact.conversation_step = Contact.STEP_AWAITING_TOPIC
                    contact.save(update_fields=["pending_service_code", "conversation_step"])
                    reply_text = build_topic_question(item)
                    reply_list = (item.clarifying_question, "Choose a topic", build_topic_list_options(item))
                else:
                    # Toggle: remove it if it's already on the cart (this is
                    # how "Edit more" lets someone take something back out),
                    # else add it straight away (flat-priced items need no
                    # follow-up question, and a quantity/topic item already
                    # on the cart is toggled off rather than re-asked).
                    if is_service_selected(contact, item.service_code):
                        remove_service_selection(contact, item)
                    else:
                        add_service_selection(contact, item)
                    reply_text, reply_list, reply_flow = _addon_screen_replies(contact, parent_service)

    elif contact.conversation_step == Contact.STEP_CONFIRMING_SELECTION:
        parent_service = ServicePricing.objects.filter(
            service_code=contact.pending_parent_service_code, is_active=True,
        ).first()
        choice = resolve_confirmation_choice(text)

        if not parent_service:
            # The add-on list this screen would reopen on Edit no longer
            # resolves (its ServicePricing got deactivated/deleted mid-
            # conversation) — there's nothing left to edit back into, so
            # finalize regardless of choice, but SAY SO explicitly. Past
            # behavior silently treated ANY reply here (including an
            # explicit "Edit" tap) as Confirm with no explanation — a real
            # customer who tapped Edit and got checked out instead deserves
            # to know why, not have it happen invisibly.
            reply_text = (
                "That item's no longer available to edit, so I've gone ahead "
                "with your current selection:\n\n" + build_selection_complete_reply(contact)
            )
            contact.pending_category_code = ""
            contact.pending_service_code = ""
            contact.pending_parent_service_code = ""
            contact.conversation_step = Contact.STEP_PRICING_SHOWN
            contact.save(update_fields=[
                "pending_category_code", "pending_service_code",
                "pending_parent_service_code", "conversation_step",
            ])
            clear_review_reached(contact)
            _maybe_schedule_referral_ask(contact)
            record_last_quote(contact)
            trigger_conversion_cta = True
        elif choice == EDIT_SELECTION_ID:
            reply_text, reply_list, reply_flow = _enter_addon_list(contact, parent_service)
        elif choice == CONFIRM_SELECTION_ID:
            reply_text = build_selection_complete_reply(contact)
            contact.pending_category_code = ""
            contact.pending_service_code = ""
            contact.pending_parent_service_code = ""
            contact.conversation_step = Contact.STEP_PRICING_SHOWN
            contact.save(update_fields=[
                "pending_category_code", "pending_service_code",
                "pending_parent_service_code", "conversation_step",
            ])
            # A real completion — same "no longer sitting on an unconfirmed
            # review" bookkeeping as the mega-Flow completion branch above.
            clear_review_reached(contact)
            _maybe_schedule_referral_ask(contact)
            record_last_quote(contact)
            trigger_conversion_cta = True
        # Unparseable: deliberately reply_text stays None here (conversation_step
        # / pending_parent_service_code left untouched) rather than re-showing
        # this same Confirm/Edit prompt verbatim — falls through to free-chat
        # below, which can resolve a free-text cart edit ("actually also
        # remove X") via classify_and_respond's edit_cart action (see
        # services._build_system_instruction's cart-context note) instead of
        # looping the identical message at someone trying to explain what
        # they want in their own words. A genuinely unrelated reply just gets
        # answered normally, still sitting on this exact screen afterward so
        # a later "confirm"/"edit" still resolves.

    elif contact.conversation_step == Contact.STEP_CONFIRMING_QUOTE:
        # SUMMARY -> CONFIRM -> PAYMENT for a standalone (non-wizard) quote —
        # see the flat/quantity/topic branches above that put the contact
        # here. Mirrors STEP_CONFIRMING_SELECTION just above, minus the
        # "Edit" path (there's no add-on list to go back into for one item).
        choice = resolve_quote_confirmation(text)

        if choice == QUOTE_CONFIRM_ID:
            reply_text = build_selection_complete_reply(contact)
            contact.pending_category_code = ""
            contact.pending_service_code = ""
            contact.pending_parent_service_code = ""
            contact.conversation_step = Contact.STEP_PRICING_SHOWN
            contact.save(update_fields=[
                "pending_category_code", "pending_service_code",
                "pending_parent_service_code", "conversation_step",
            ])
            # A real completion — same "no longer sitting on an unconfirmed
            # review" bookkeeping as STEP_CONFIRMING_SELECTION's confirm.
            clear_review_reached(contact)
            _maybe_schedule_referral_ask(contact)
            record_last_quote(contact)
            trigger_conversion_cta = True
        elif choice == QUOTE_CANCEL_ID:
            # Drop just what THIS quote turn was actually about — not the
            # whole cart. pending_service_code names that one item for every
            # standalone entry point into this screen (the flat/quantity/
            # topic branches all set it right before landing here); a
            # free-text cart edit (tasks.py's edit_cart handling) reaches
            # this same screen too but deliberately clears pending_service_code
            # first, so it falls into the "nothing specific to drop" case
            # below instead of wiping a cart that might hold several other,
            # already-legitimate items the customer built up before this
            # turn — the old blanket `selected_services = []` here used to
            # nuke all of it, correct only back when this screen could never
            # hold more than the one item just quoted.
            pending_code = contact.pending_service_code
            dropped_service = (
                ServicePricing.objects.filter(service_code=pending_code).first() if pending_code else None
            )
            if dropped_service:
                remove_service_selection(contact, dropped_service)
                reply_text = "No worries — dropped that one. Let me know if you'd like to look at something else."
            else:
                reply_text = "No worries — I've left your cart as it was. Let me know what you'd like to change."
            contact.pending_category_code = ""
            contact.pending_service_code = ""
            contact.pending_parent_service_code = ""
            contact.conversation_step = Contact.STEP_PRICING_SHOWN
            contact.save(update_fields=[
                "pending_category_code", "pending_service_code",
                "pending_parent_service_code", "conversation_step",
            ])
            clear_review_reached(contact)
        # Unparseable: deliberately reply_text stays None here, same "fall
        # through to free-chat/edit_cart instead of looping the identical
        # prompt" reasoning as STEP_CONFIRMING_SELECTION just above.

    elif contact.conversation_step == Contact.STEP_AWAITING_QUANTITY:
        service = ServicePricing.objects.filter(service_code=contact.pending_service_code).first()
        quantity = resolve_quantity(text) if service else None
        parent_service = None
        if service and quantity and contact.pending_parent_service_code:
            parent_service = ServicePricing.objects.filter(
                service_code=contact.pending_parent_service_code, is_active=True,
            ).first()

        if service and quantity and parent_service:
            # Mid-wizard quantity item (e.g. House Property) — record it and
            # return to the add-on list.
            add_service_selection(contact, service, quantity)
            contact.has_answered_clarifier = True
            contact.save(update_fields=["has_answered_clarifier"])
            reply_text, reply_list, reply_flow = _enter_addon_list(contact, parent_service)
        elif service and quantity:
            # Standalone quantity-priced service reached directly (not part
            # of the add-on list) — quote the total, but hold the payment
            # link for STEP_CONFIRMING_QUOTE's explicit Confirm tap, same
            # SUMMARY -> CONFIRM -> PAYMENT step as the flat-priced branch.
            contact.pending_parent_service_code = ""
            contact.conversation_step = Contact.STEP_CONFIRMING_QUOTE
            contact.has_answered_clarifier = True
            contact.save(update_fields=["pending_parent_service_code", "conversation_step", "has_answered_clarifier"])
            add_service_selection(contact, service, quantity)
            mark_review_reached(contact)
            reply_text = build_quantity_reply(service, quantity)
            reply_list = (reply_text, QUOTE_CONFIRMATION_LIST_BUTTON_LABEL, QUOTE_CONFIRMATION_LIST_OPTIONS)
        else:
            # Couldn't parse a quantity — fall through to free-chat rather
            # than looping the same question.
            contact.pending_service_code = ""
            contact.pending_parent_service_code = ""
            contact.conversation_step = Contact.STEP_PRICING_SHOWN
            contact.save(update_fields=["pending_service_code", "pending_parent_service_code", "conversation_step"])

    elif contact.conversation_step == Contact.STEP_AWAITING_TOPIC:
        service = ServicePricing.objects.filter(service_code=contact.pending_service_code).first()
        topic = resolve_topic_choice(service, text) if service else None
        parent_service = None
        if service and topic and contact.pending_parent_service_code:
            parent_service = ServicePricing.objects.filter(
                service_code=contact.pending_parent_service_code, is_active=True,
            ).first()

        if service and topic and parent_service:
            # Mid-wizard topic item (none in today's catalogue, kept for
            # symmetry with the quantity case) — record it and return to
            # the add-on list.
            add_service_selection(contact, service)
            contact.has_answered_clarifier = True
            contact.save(update_fields=["has_answered_clarifier"])
            reply_text, reply_list, reply_flow = _enter_addon_list(contact, parent_service)
        elif service and topic:
            # Standalone topic-tagged service reached directly (not part of
            # the add-on list) — same SUMMARY -> CONFIRM -> PAYMENT hold as
            # the flat/quantity branches: quote it, wait for the explicit
            # Confirm tap before the payment link goes out.
            contact.conversation_step = Contact.STEP_CONFIRMING_QUOTE
            contact.has_answered_clarifier = True
            contact.save(update_fields=["conversation_step", "has_answered_clarifier"])
            add_service_selection(contact, service)
            mark_review_reached(contact)
            reply_text = build_topic_confirmation(service, topic)
            reply_list = (reply_text, QUOTE_CONFIRMATION_LIST_BUTTON_LABEL, QUOTE_CONFIRMATION_LIST_OPTIONS)
        else:
            # Couldn't match a topic — fall through to free-chat rather than
            # looping the same question.
            contact.pending_service_code = ""
            contact.conversation_step = Contact.STEP_PRICING_SHOWN
            contact.save(update_fields=["pending_service_code", "conversation_step"])

    # A bare 1-2 digit message that reached here (every scripted branch above
    # falls through to conversation_step=PRICING_SHOWN on no match) almost
    # always means someone tapped a stale menu button/list row from earlier
    # in the chat, or typed a number with no active menu behind it anymore.
    # Free-chat has no real menu to validate against in this state — sending
    # it there produced confusing, sometimes fabricated "invalid option"
    # replies (Gemini improvising a fake menu range instead of admitting it
    # doesn't know). Re-showing the real menu is the honest, helpful move.
    if (
        reply_text is None
        and normalized.isdigit()
        and len(normalized) <= 2
        and contact.conversation_step == Contact.STEP_PRICING_SHOWN
    ):
        contact.conversation_step = Contact.STEP_CATEGORY_SHOWN
        contact.save(update_fields=["conversation_step"])
        reply_text, reply_list, reply_list_sections = _top_level_menu_replies("That menu's no longer active. ")

    if reply_text is None:
        recent_history = contact.messages.order_by("-created_at")[:LIVE_HISTORY_WINDOW]
        t0 = time.monotonic()
        classification = classify_and_respond(contact, text, history=list(recent_history))
        logger.info("[timing] contact=%s ai_call=%.2fs", contact_id, time.monotonic() - t0)

        reply_text = classification["reply"]
        action = classification["action"]
        flow_type = classification["flow_type"]
        # Required, not optional (see the feature's logging requirement) —
        # this is how the classification prompt gets tuned against real
        # misfires later. confirmed/rejected is logged separately, above,
        # once the customer actually answers the Confirm/Edit prompt below.
        logger.info(
            "[flow-intent] contact=%s action=%s flow_type=%s confidence=%s text=%r",
            contact_id, action, flow_type, classification["confidence"], text,
        )

        # Routes to the Sales & Cart / Human Handoff / Order & Status
        # specialist node (or straight through for continue_chat/
        # clarify_first) based on the classification above — see
        # agents.py's StateGraph (Supervisor + specialist nodes). The
        # classify_and_respond call itself stays here, not inside the
        # graph, so this is still one Gemini round-trip per turn, and this
        # is still the exact call site every classify_and_respond test
        # patches.
        agent_result = run_agent_graph(contact, text, classification)
        reply_text = agent_result["reply_text"]
        reply_buttons = agent_result["reply_buttons"]
        reply_list = agent_result["reply_list"]
        reply_service_flow_target = agent_result["reply_service_flow_target"]
        reply_service_flow_body = agent_result["reply_service_flow_body"]
        if agent_result["trigger_conversion_cta"]:
            trigger_conversion_cta = True

        if contact.active_intake_flow_type and agent_result["agent"] != "guided_intake":
            # This turn was an interruption to a guided chat intake still
            # in progress (agents.guided_intake_node left it untouched and
            # deferred to this same free-chat path for a real answer to
            # whatever was actually asked) — the pending question gets
            # folded onto whatever reply this turn already produced, same
            # "append, don't send a second message" pattern as
            # build_resume_step_nudge, rather than silently dropped.
            reply_text = (reply_text or "") + pending_nudge(contact)

        if action in ("continue_chat", "clarify_first") and contact.conversation_step == Contact.STEP_WELCOMED:
            # Only these two actions — launch_flow_candidate and edit_cart
            # are themselves engagement with the pending step (a launch
            # attempt, or actively working an existing cart), not "still
            # ignoring it," so they don't count toward this and (via the
            # resets below) actively clear it.
            contact.free_chat_turns_since_welcomed += 1
            resume_update_fields = ["free_chat_turns_since_welcomed"]
            if (
                contact.free_chat_turns_since_welcomed >= RESUME_NUDGE_AFTER_TURNS
                and not contact.resume_nudge_sent_at
            ):
                reply_text += build_resume_step_nudge(contact)
                contact.resume_nudge_sent_at = timezone.now()
                resume_update_fields.append("resume_nudge_sent_at")
            contact.save(update_fields=resume_update_fields)

    Message.objects.create(contact=contact, direction="out", body=reply_text)

    t1 = time.monotonic()
    # Checkout-confirm outcome (trigger_conversion_cta): fold the summary
    # (reply_text) straight into the Pay Now CTA-URL message instead of
    # sending it as its own plain-text message first, whenever a payment
    # link is actually available — one outbound WhatsApp message instead of
    # two (see services.maybe_send_payment_cta). Every trigger_conversion_cta
    # branch sets plain reply_text only (no buttons/list/flow alongside it),
    # so this guard is just belt-and-braces. Falls through to the plain send
    # below when payment isn't eligible (no amount / Razorpay unset / link
    # creation failed) so the summary still reaches the customer either way.
    # No callback/support message is sent here anymore, on purpose — never
    # unconditionally alongside every confirm. HELP still hands off to a
    # human; nothing else currently proactively offers the callback button
    # (see services.CALLBACK_CTA_BODY/CALLBACK_CTA_BUTTON_OPTIONS, unused
    # since send_hot_lead_nudges was removed) though typing "callback" still
    # works via resolve_callback_choice.
    payment_cta_sent = False
    if trigger_conversion_cta and not (
        reply_buttons or reply_list or reply_list_sections
        or reply_flow or reply_service_flow or reply_service_flow_target
    ):
        payment_cta_sent = maybe_send_payment_cta(contact, summary=reply_text)

    if payment_cta_sent or special_action_sent:
        pass
    elif reply_buttons:
        body_text, options = reply_buttons
        send_whatsapp_buttons(contact.phone_number, body_text, options)
    elif reply_list:
        body_text, button_label, options = reply_list
        send_whatsapp_list(contact.phone_number, body_text, button_label, options)
    elif reply_list_sections:
        body_text, button_label, sections = reply_list_sections
        send_whatsapp_list_sections(contact.phone_number, body_text, button_label, sections)
    elif reply_flow:
        body_text, flow_cta, flow_action_payload, screen_id = reply_flow
        send_whatsapp_flow(contact.phone_number, body_text, flow_cta, flow_action_payload, screen_id=screen_id)
    elif reply_service_flow:
        send_service_flow(contact)
    elif reply_service_flow_target:
        send_service_flow(contact, deep_link_screen=reply_service_flow_target, body_text=reply_service_flow_body)
    else:
        send_whatsapp_message(contact.phone_number, reply_text)
    logger.info("[timing] contact=%s whatsapp_send=%.2fs", contact_id, time.monotonic() - t1)

    # Satisfaction follow-up — fires once the callback-CTA micro-interaction
    # concludes (see trigger_csat above), not on every outcome directly.
    if trigger_csat:
        Message.objects.create(contact=contact, direction="out", body=CSAT_PROMPT_BODY)
        send_whatsapp_buttons(contact.phone_number, CSAT_PROMPT_BODY, CSAT_BUTTON_OPTIONS)

    logger.info("[timing] contact=%s task_total=%.2fs", contact_id, time.monotonic() - task_start)

    # Lead score: pure DB arithmetic, no LLM call, so this runs after every
    # message with negligible cost — after the reply's already sent, so it
    # never adds latency to what the user is waiting on.
    update_lead_score(contact)

    # Fold recent history into the long-term summary (+ next-step prediction)
    # every N messages, off the critical path — doesn't add latency to the
    # reply the user is waiting on.
    total_messages = contact.messages.count()
    if total_messages % SUMMARY_EVERY_N_MESSAGES == 0:
        update_contact_summary.delay(contact_id)


@shared_task(time_limit=30, soft_time_limit=25)
def update_contact_summary(contact_id):
    """
    Refreshes Contact.summary and predicted_next_action from the most recent
    messages, in a single Gemini call — the bot's long-term memory and lead
    intelligence, both kept separate from the raw per-reply history window.
    """
    contact = Contact.objects.get(id=contact_id)
    recent = list(contact.messages.order_by("-created_at")[:SUMMARY_EVERY_N_MESSAGES])
    recent.reverse()  # oldest first

    new_summary, next_action = summarize_conversation(contact, recent)
    update_fields = []
    if new_summary != contact.summary:
        contact.summary = new_summary
        update_fields.append("summary")
    if next_action != contact.predicted_next_action:
        contact.predicted_next_action = next_action
        update_fields.append("predicted_next_action")
    if update_fields:
        contact.save(update_fields=update_fields)
        logger.info("[summary] contact=%s summary/next-action updated: %r", contact_id, next_action)


def _maybe_schedule_referral_ask(contact):
    """Schedules send_referral_ask a while after a booking actually
    completes — checked here too (not just inside the task itself) so a
    contact who re-books within the cooldown window doesn't even get a
    redundant task queued."""
    cutoff = timezone.now() - timezone.timedelta(days=REFERRAL_ASK_COOLDOWN_DAYS)
    if contact.referral_ask_sent_at and contact.referral_ask_sent_at > cutoff:
        return
    send_referral_ask.apply_async((contact.id,), countdown=REFERRAL_ASK_DELAY_SECONDS)


@shared_task(time_limit=30, soft_time_limit=25)
def send_referral_ask(contact_id):
    """Fires ~1h after a completed booking (see _maybe_schedule_referral_ask)
    — "refer a friend, both get 10% off", the cheapest new-lead channel
    available given the templates/Celery infrastructure already in place.
    Re-checks the cooldown itself (not just at schedule time) to survive a
    contact re-booking during the delay."""
    contact = Contact.objects.get(id=contact_id)
    cutoff = timezone.now() - timezone.timedelta(days=REFERRAL_ASK_COOLDOWN_DAYS)
    if contact.referral_ask_sent_at and contact.referral_ask_sent_at > cutoff:
        return
    message = build_referral_message(contact)
    Message.objects.create(contact=contact, direction="out", body=message)
    send_whatsapp_message(contact.phone_number, message)
    contact.referral_ask_sent_at = timezone.now()
    contact.save(update_fields=["referral_ask_sent_at"])


@shared_task(time_limit=60, soft_time_limit=55)
def send_abandoned_cart_nudges():
    """
    Celery Beat periodic task (see myproject/celery.py's beat_schedule) —
    finds contacts sitting on an unconfirmed review/confirm screen (see
    services.mark_review_reached) past the +2h / +24h thresholds who haven't
    already gotten that nudge, and sends it.

    WhatsApp only allows free-form sends within 24h of the customer's last
    inbound message — the +24h nudge, and any +2h nudge for a contact who's
    gone unusually quiet, is always outside that window, so a plain
    send_whatsapp_message there fails outright (error 131047). When
    settings.WHATSAPP_CART_NUDGE_TEMPLATE_NAME is configured, a failed
    free-form send falls back to send_whatsapp_template — the only send type
    Meta allows outside the window — so the nudge still reaches the
    customer. With no template configured, the fallback is skipped and the
    send stays failed, same as before (see docs/whatsapp_flows/README.md to
    set one up).

    Nudge-sent timestamps are only stamped once a send (free-form or
    template) actually succeeds — a still-failed send is left unstamped so
    the next run's cutoff filter picks the contact back up and retries,
    rather than the old behavior of marking a nudge "sent" that never
    reached anyone.
    """
    now = timezone.now()
    template_name = settings.WHATSAPP_CART_NUDGE_TEMPLATE_NAME
    for hours in CART_NUDGE_AFTER_HOURS:
        field = "cart_nudge_2h_sent_at" if hours == 2 else "cart_nudge_24h_sent_at"
        cutoff = now - timezone.timedelta(hours=hours)
        stale_contacts = Contact.objects.filter(
            review_reached_at__isnull=False,
            review_reached_at__lte=cutoff,
            **{f"{field}__isnull": True},
        )
        for contact in stale_contacts:
            message = build_cart_nudge_message(contact, hours)
            result = send_whatsapp_message(contact.phone_number, message)
            sent_body = message

            if result is None and template_name:
                logger.info(
                    "[cart-nudge] contact=%s +%sh free-form send failed (likely outside "
                    "the 24h window) — falling back to template '%s'",
                    contact.id, hours, template_name,
                )
                params = build_cart_nudge_template_params(contact)
                result = send_whatsapp_template(
                    contact.phone_number, template_name,
                    settings.WHATSAPP_CART_NUDGE_TEMPLATE_LANG, params,
                )
                sent_body = f"[template:{template_name}] " + ", ".join(value for _name, value in params)

            if result is None:
                logger.warning(
                    "[cart-nudge] contact=%s +%sh nudge send failed%s — will retry next run",
                    contact.id, hours,
                    "" if template_name else " and no WHATSAPP_CART_NUDGE_TEMPLATE_NAME is "
                    "configured (see docs/whatsapp_flows/README.md)",
                )
                continue

            Message.objects.create(contact=contact, direction="out", body=sent_body)
            # Re-surface the same still-open link (maybe_send_payment_cta
            # reuses rather than re-minting one) — this is what turns a
            # generic "still there?" nudge into an actual recovery
            # mechanism for whoever saw a total and vanished.
            maybe_send_payment_cta(contact)
            setattr(contact, field, now)
            contact.save(update_fields=[field])


@shared_task(time_limit=60, soft_time_limit=55)
def send_dropoff_nudges():
    """
    Celery Beat periodic task — a second, additional checkpoint schedule
    alongside send_abandoned_cart_nudges' 2h/24h (see DROPOFF_NUDGE_STAGES),
    gated per stage by lead_tier rather than firing for every drop-off
    regardless of intent: a cold contact who bounced off the welcome menu is
    low-intent noise and never gets nudged here, only hot/warm ones do, and
    only at the stages configured for their tier.

    NOTE: this runs alongside send_abandoned_cart_nudges rather than
    replacing it (both key off the same review_reached_at signal) — a hot
    lead who abandons a cart can get both schedules' touches (30m/2h/20h/
    24h/48h/7d). If that turns out to be more contact than intended, the fix
    is either to fold CART_NUDGE_AFTER_HOURS into this same stage list, or
    to have one schedule skip contacts the other already nudged this cycle.

    Same "only stamp on real success", free-form-then-template-fallback
    pattern as send_abandoned_cart_nudges — see that task's docstring.
    """
    now = timezone.now()
    template_name = settings.WHATSAPP_CART_NUDGE_TEMPLATE_NAME
    for stage, hours, field, allowed_tiers in DROPOFF_NUDGE_STAGES:
        cutoff = now - timezone.timedelta(hours=hours)
        stale_contacts = Contact.objects.filter(
            review_reached_at__isnull=False,
            review_reached_at__lte=cutoff,
            lead_tier__in=allowed_tiers,
            **{f"{field}__isnull": True},
        )
        for contact in stale_contacts:
            message = build_dropoff_nudge_message(contact, stage)
            result = send_whatsapp_message(contact.phone_number, message)
            sent_body = message

            if result is None and template_name:
                logger.info(
                    "[dropoff-nudge] contact=%s stage=%s free-form send failed (likely "
                    "outside the 24h window) — falling back to template '%s'",
                    contact.id, stage, template_name,
                )
                params = build_cart_nudge_template_params(contact)
                result = send_whatsapp_template(
                    contact.phone_number, template_name,
                    settings.WHATSAPP_CART_NUDGE_TEMPLATE_LANG, params,
                )
                sent_body = f"[template:{template_name}] " + ", ".join(value for _name, value in params)

            if result is None:
                logger.warning(
                    "[dropoff-nudge] contact=%s stage=%s nudge send failed%s — will retry next run",
                    contact.id, stage,
                    "" if template_name else " and no WHATSAPP_CART_NUDGE_TEMPLATE_NAME is "
                    "configured (see docs/whatsapp_flows/README.md)",
                )
                continue

            Message.objects.create(contact=contact, direction="out", body=sent_body)
            maybe_send_payment_cta(contact)
            setattr(contact, field, now)
            contact.save(update_fields=[field])


# Hot-lead nudge schedule — see send_hot_lead_nudges. Three escalating
# touches counted from Contact.hot_lead_flipped_at, same (stage_label,
# hours, field) shape as DROPOFF_NUDGE_STAGES above. Deliberately NOT a
# single 1h touch (the original design) — that sat too close to the 2h cart
# nudge and the 30m drop-off stage for a hot lead who'd also abandoned a
# cart, reading as pile-on. Scoped in the task below to hot leads who never
# reached the review screen at all — a hot lead who did abandon a cart
# already gets DROPOFF_NUDGE_STAGES' hot-only 30m/7d touches instead, so the
# two schedules never double up on the same contact.
HOT_LEAD_NUDGE_STAGES = (
    ("2h", 2, "hot_lead_nudge_2h_sent_at"),
    ("24h", 24, "hot_lead_nudge_24h_sent_at"),
    ("72h", 72, "hot_lead_nudge_72h_sent_at"),
)


@shared_task(time_limit=60, soft_time_limit=55)
def send_hot_lead_nudges():
    """
    Celery Beat periodic task — customer-facing follow-up schedule
    (HOT_LEAD_NUDGE_STAGES) for a contact whose lead_tier just flipped to
    hot purely on engagement (services.update_lead_score stamps
    Contact.hot_lead_flipped_at), who's gone quiet since and never reached
    the review/confirm screen (review_reached_at is None). A hot lead who
    *did* reach review is left to DROPOFF_NUDGE_STAGES' hot-only 30m/7d
    stages instead — see that constant — so this task and that schedule
    never both fire for the same contact. Each stage is independently
    gated/stamped, same "does the earlier stage's due date pass, and has
    this stage's own field not been stamped yet" shape as
    send_dropoff_nudges.

    Does NOT skip on needs_human_review the way send_warm_lead_nudges does —
    a hot flip sets that flag itself (services.notify_human_alert), so gating
    on it being False here would mean this task never fires at all.

    On a successful send, follows up with services.CALLBACK_CTA_BODY /
    CALLBACK_CTA_BUTTON_OPTIONS ("book a callback") as the actual conversion
    action at every stage — a hot lead has shown real intent, so this pushes
    toward a human rather than the warm nudge's "no rush" tone. Only sent
    after a genuine in-window free-form send (an interactive message can't
    be sent outside the 24h window either, so there's nothing to attach it
    to when the template fallback below is what actually reached the
    customer).

    Same free-form-first, template-fallback, only-stamp-on-success pattern as
    the other nudges. All three stages share one approved template
    (WHATSAPP_HOT_LEAD_TEMPLATE_NAME) — see
    services.build_hot_lead_nudge_template_params's docstring for why.
    """
    now = timezone.now()
    template_name = settings.WHATSAPP_HOT_LEAD_TEMPLATE_NAME
    for stage, hours, field in HOT_LEAD_NUDGE_STAGES:
        cutoff = now - timezone.timedelta(hours=hours)
        stale_contacts = Contact.objects.filter(
            lead_tier=Contact.LEAD_TIER_HOT,
            hot_lead_flipped_at__isnull=False,
            hot_lead_flipped_at__lte=cutoff,
            review_reached_at__isnull=True,
            **{f"{field}__isnull": True},
        )
        for contact in stale_contacts:
            message = build_hot_lead_nudge_message(contact, stage)
            result = send_whatsapp_message(contact.phone_number, message)
            sent_body = message
            in_window = result is not None

            if result is None and template_name:
                logger.info(
                    "[hot-lead-nudge] contact=%s stage=%s free-form send failed "
                    "(likely outside the 24h window) — falling back to template '%s'",
                    contact.id, stage, template_name,
                )
                params = build_hot_lead_nudge_template_params(contact)
                result = send_whatsapp_template(
                    contact.phone_number, template_name,
                    settings.WHATSAPP_HOT_LEAD_TEMPLATE_LANG, params,
                )
                sent_body = f"[template:{template_name}] " + ", ".join(value for _name, value in params)

            if result is None:
                logger.warning(
                    "[hot-lead-nudge] contact=%s stage=%s nudge send failed%s — will retry next run",
                    contact.id, stage,
                    "" if template_name else " and no WHATSAPP_HOT_LEAD_TEMPLATE_NAME is "
                    "configured (see docs/whatsapp_flows/README.md)",
                )
                continue

            Message.objects.create(contact=contact, direction="out", body=sent_body)
            if in_window:
                send_whatsapp_buttons(contact.phone_number, CALLBACK_CTA_BODY, CALLBACK_CTA_BUTTON_OPTIONS)
                Message.objects.create(contact=contact, direction="out", body=CALLBACK_CTA_BODY)
            setattr(contact, field, now)
            contact.save(update_fields=[field])


# Warm-lead nudge schedule — see send_warm_lead_nudges. Same (stage_label,
# hours, field) shape as HOT_LEAD_NUDGE_STAGES above, just a much longer
# fuse throughout — warm interest is weaker signal, and a fast/frequent
# follow-up would feel pushy.
WARM_LEAD_NUDGE_STAGES = (
    ("12h", 12, "warm_lead_nudge_12h_sent_at"),
    ("3d", 24 * 3, "warm_lead_nudge_3d_sent_at"),
    ("7d", 24 * 7, "warm_lead_nudge_7d_sent_at"),
)


@shared_task(time_limit=60, soft_time_limit=55)
def send_warm_lead_nudges():
    """
    Celery Beat periodic task — one tier down from send_hot_lead_nudges,
    same overall shape (a WARM_LEAD_NUDGE_STAGES schedule counted from
    Contact.warm_lead_flipped_at, each stage independently gated/stamped),
    deliberately lower-pressure at every stage and spaced much further
    apart (12h/3d/7d vs. 2h/24h/72h).

    Gate direction is the OPPOSITE of send_hot_lead_nudges' needs_human_review
    check: a warm flip never sets needs_human_review itself (only a hot flip
    does — see services.update_lead_score), so this instead skips a contact
    who's currently flagged for staff attention for ANY other reason (a bad
    CSAT, a HELP request, an unresolved hot-lead alert) — piling a bot nudge
    on top of an active human-attention item would be the wrong move here,
    unlike the hot-lead nudge's own dedicated alert.

    No CALLBACK_CTA buttons attached at any stage (unlike the hot-lead
    schedule) — a "book a call" push doesn't match this tier's "no rush"
    tone; see services.build_warm_lead_nudge_message. Same
    free-form-first, template-fallback, only-stamp-on-success pattern as the
    other nudges. All three stages share one approved template
    (WHATSAPP_WARM_LEAD_TEMPLATE_NAME), same reasoning as the hot schedule.
    """
    now = timezone.now()
    template_name = settings.WHATSAPP_WARM_LEAD_TEMPLATE_NAME
    for stage, hours, field in WARM_LEAD_NUDGE_STAGES:
        cutoff = now - timezone.timedelta(hours=hours)
        stale_contacts = Contact.objects.filter(
            lead_tier=Contact.LEAD_TIER_WARM,
            warm_lead_flipped_at__isnull=False,
            warm_lead_flipped_at__lte=cutoff,
            needs_human_review=False,
            **{f"{field}__isnull": True},
        )
        for contact in stale_contacts:
            message = build_warm_lead_nudge_message(contact, stage)
            result = send_whatsapp_message(contact.phone_number, message)
            sent_body = message

            if result is None and template_name:
                logger.info(
                    "[warm-lead-nudge] contact=%s stage=%s free-form send failed "
                    "(likely outside the 24h window) — falling back to template '%s'",
                    contact.id, stage, template_name,
                )
                params = build_warm_lead_nudge_template_params(contact)
                result = send_whatsapp_template(
                    contact.phone_number, template_name,
                    settings.WHATSAPP_WARM_LEAD_TEMPLATE_LANG, params,
                )
                sent_body = f"[template:{template_name}] " + ", ".join(value for _name, value in params)

            if result is None:
                logger.warning(
                    "[warm-lead-nudge] contact=%s stage=%s nudge send failed%s — will retry next run",
                    contact.id, stage,
                    "" if template_name else " and no WHATSAPP_WARM_LEAD_TEMPLATE_NAME is "
                    "configured (see docs/whatsapp_flows/README.md)",
                )
                continue

            Message.objects.create(contact=contact, direction="out", body=sent_body)
            setattr(contact, field, now)
            contact.save(update_fields=[field])


# acks_late + a real retry policy — deliberately different from every task
# above. Those re-run harmlessly if lost (worst case: a reply is late or a
# nudge is skipped this cycle); this one carries a "did the customer's money
# arrive" fact, so silently losing it to a dead worker or a transient DB lock
# is a real business incident, not a shrug. acks_late means the broker only
# drops this job once it actually finishes — a worker crash mid-run puts it
# straight back on the queue instead of discarding it.
@shared_task(
    bind=True, acks_late=True, max_retries=5,
    autoretry_for=(Exception,), retry_backoff=True, retry_backoff_max=300, retry_jitter=True,
    time_limit=30, soft_time_limit=25,
)
def process_payment_webhook(self, event_type, payment_link_id, gateway_payment_id, raw_payload):
    """
    Applies one Razorpay webhook event (see views.razorpay_webhook) to the
    matching Order. Idempotent by design: gateway_payment_id already
    recorded on the order means this exact payment was already applied, so
    a duplicate delivery (Razorpay retries on anything but a 2xx, same as
    Meta's webhook) is a safe no-op rather than a double-send.

    select_for_update + transaction.atomic serializes concurrent deliveries
    for the *same* order (a legitimate retry racing the original) without
    blocking unrelated orders — the row lock is scoped to one Order.id.
    """
    try:
        with transaction.atomic():
            order = Order.objects.select_for_update().get(payment_link_id=payment_link_id)

            if gateway_payment_id and order.gateway_payment_id == gateway_payment_id:
                logger.info("[payment-webhook] event=%s order=%s already applied, skipping", event_type, order.id)
                return

            # services.invalidate_stale_orders (a cart edit after this link
            # went out) already tried to cancel this link on Razorpay's side
            # and marked it expired locally — a payment still landing here
            # means either that cancel call failed, or the customer had
            # already completed checkout in their browser in the race window
            # before it took effect. The money is real either way, so it's
            # still recorded as paid below, just flagged for a human to
            # confirm what was actually charged matches what's now selected.
            was_already_invalidated = (
                order.status == Order.STATUS_EXPIRED and order.failure_reason == "cart_edited"
            )

            if event_type in ("payment_link.paid", "payment.captured"):
                order.status = Order.STATUS_PAID
                order.gateway_payment_id = gateway_payment_id
                order.save(update_fields=["status", "gateway_payment_id"])
            elif event_type in ("payment_link.expired",):
                order.status = Order.STATUS_EXPIRED
                order.save(update_fields=["status"])
            elif event_type in ("payment.failed",):
                order.status = Order.STATUS_FAILED
                order.failure_reason = (raw_payload or {}).get("error_description", "") or "payment_failed"
                order.save(update_fields=["status", "failure_reason"])
            else:
                logger.info("[payment-webhook] ignoring unhandled event type=%s order=%s", event_type, order.id)
                return
    except Order.DoesNotExist:
        # Genuinely unexpected (a webhook for a link we never created, or
        # for one purged some other way) — log loudly rather than silently
        # dropping it, but don't retry forever chasing an order that will
        # never exist.
        logger.error("[payment-webhook] no Order for payment_link_id=%s (event=%s)", payment_link_id, event_type)
        return

    contact = order.contact

    if order.status == Order.STATUS_PAID:
        message = (
            f"Payment received — ₹{order.amount:,.0f}. Thank you! "
            + ("We'll confirm final scope and follow up on the balance shortly."
               if order.is_advance else
               "We're getting started on this right away.")
        )
        Message.objects.create(contact=contact, direction="out", body=message)
        send_whatsapp_message(contact.phone_number, message)
        clear_review_reached(contact)
        _maybe_schedule_referral_ask(contact)
        if was_already_invalidated:
            notify_human_alert(
                contact, "stale_link_paid",
                detail=f"order={order.id} amount={order.amount} — paid after the cart was edited; confirm what was actually charged still matches what's selected now",
            )

    elif order.status == Order.STATUS_FAILED:
        # A failed payment on an otherwise-engaged contact is exactly the
        # kind of thing that should surface for a human to call, not decay
        # silently the way an ordinary abandoned cart does.
        notify_human_alert(contact, "payment_failed", detail=f"order={order.id} reason={order.failure_reason}")
        message = (
            "That payment didn't go through — no charge was made. "
            "Reply HELP and our team can sort out an alternative, or try the link again."
        )
        Message.objects.create(contact=contact, direction="out", body=message)
        send_whatsapp_message(contact.phone_number, message)
