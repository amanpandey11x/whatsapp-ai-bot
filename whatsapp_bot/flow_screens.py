"""
Screen-by-screen logic for the live, multi-category WhatsApp Flow (welcome
through cart review) — the data_exchange half of what views.whatsapp_flow_endpoint
receives once decrypted. Kept separate from flow_crypto.py (pure encryption,
no business logic) and from the ITR-only static Flow in services.py (no
server round trip at all).

Every screen's option list is pulled live from ServiceCategory/ServicePricing
via services.py — the same tables the chat wizard reads — so nothing here
ever invents a price or a catalogue item; a missing/inactive record just
means that option isn't offered, never a hallucinated fallback.

Currently wired: SERVICE -> ITR_TYPE -> ITR_ADDONS -> REVIEW, and
SERVICE -> {GST_FILING, TDS_FILING, BUSINESS_REG, NOTICES, BOOK_EXPERT} ->
REVIEW (all five of those are flat, multi-select catalogues — see
_FLAT_CATEGORY_SCREENS). Any category not in that catalogue routes to
COMING_SOON rather than guessing — see docs/whatsapp_flows/README.md.

SERVICE is the Flow's entry screen — there is no LANGUAGE screen in the
Flow itself. Language is collected once, in chat, before the Flow is ever
opened (tasks.py's unconditional language gate, in process_incoming_message)
— by the time send_service_flow runs, contact.language is always already
set, so the Flow never needs to ask it again. SERVICE also carries the
"First time filing with us?" qualifying question (Contact.first_time_filer)
that used to live on LANGUAGE — conditionally shown (via the `If` block
around it in service_flow.json, driven by `show_first_time_question` in
_service_screen_response below) only while that field is still unset, so a
returning contact reopening the Flow (MENU keyword, an ERROR retry, etc.)
isn't asked it again.
"""

import logging

from . import services
from .models import ServicePricing

logger = logging.getLogger(__name__)

SCREEN_SERVICE = "SERVICE"
SCREEN_ITR_TYPE = "ITR_TYPE"
SCREEN_ITR_ADDONS = "ITR_ADDONS"
SCREEN_GST_FILING = "GST_FILING"
SCREEN_TDS_FILING = "TDS_FILING"
SCREEN_BUSINESS_REG = "BUSINESS_REG"
SCREEN_NOTICES = "NOTICES"
SCREEN_BOOK_EXPERT = "BOOK_EXPERT"
SCREEN_REVIEW = "REVIEW"
SCREEN_COMING_SOON = "COMING_SOON"
SCREEN_ERROR = "ERROR"

# Every flat, top-level, multi-select category (everything except ITR, which
# has its own type-then-addons tree) — one entry drives both the SERVICE
# screen's routing and that screen's own submit handler, so a 5th category
# like this is a single new row here plus a new screen in service_flow.json,
# not scattered special-casing.
_FLAT_CATEGORY_SCREENS = {
    "gst": {"screen": SCREEN_GST_FILING, "options_fn": services.flow_gst_options,
            "data_key": "gst_options", "field": "gst_filings"},
    "tds_filing": {"screen": SCREEN_TDS_FILING, "options_fn": services.flow_tds_filing_options,
                   "data_key": "tds_filing_options", "field": "tds_filing_items"},
    "business_reg": {"screen": SCREEN_BUSINESS_REG, "options_fn": services.flow_business_reg_options,
                      "data_key": "business_reg_options", "field": "business_reg_items"},
    "notices": {"screen": SCREEN_NOTICES, "options_fn": services.flow_notice_options,
                "data_key": "notice_options", "field": "notice_items"},
    "book_expert": {"screen": SCREEN_BOOK_EXPERT, "options_fn": services.flow_book_expert_options,
                     "data_key": "consult_options", "field": "consult_items"},
}
_FLAT_SCREEN_CONFIG_BY_SCREEN = {cfg["screen"]: (code, cfg) for code, cfg in _FLAT_CATEGORY_SCREENS.items()}


def _rows_to_options(rows):
    """(id, title, description) triples -> the {id, title, description} dict
    shape a RadioButtonsGroup/CheckboxGroup data-source expects."""
    return [{"id": row_id, "title": title, "description": description} for row_id, title, description in rows]


def _screen_response(version, screen, data):
    return {"version": version, "screen": screen, "data": data}


def _error_response(version, message):
    """A dedicated ERROR screen rather than silently redisplaying the
    current one — a customer mid-Flow should see *something* changed, not
    wonder why tapping Submit did nothing. See docs/whatsapp_flows for the
    ERROR screen's layout (a message + a button back to SERVICE)."""
    return _screen_response(version, SCREEN_ERROR, {"message": message})


def _parse_quantity(raw):
    """A TextInput's value always arrives as a string (or None/empty when
    left blank) — never trust it as already-validated. Returns None for
    blank/non-numeric/zero-or-negative input, exactly like services.resolve_quantity."""
    if raw in (None, ""):
        return None
    try:
        quantity = int(str(raw).strip())
    except (TypeError, ValueError):
        return None
    return quantity if quantity > 0 else None


def _service_screen_response(version, contact):
    """SERVICE's own RadioButtonsGroup ships its options as a literal
    data-source in service_flow.json (not ${data.service_options}) — Meta's
    client was rendering an inline "missing required property 'data-source'"
    error on this component whenever the very first INIT round trip didn't
    resolve before the screen painted (draft-flow test sends being one
    reliable way to hit that race). The top-level category list is fixed
    enough that shipping it statically removes the dependency on that first
    request entirely.

    The one thing this response does still carry is
    `show_first_time_question` — SERVICE is the Flow's entry screen (there is
    no LANGUAGE screen; language is collected in chat before the Flow ever
    opens, see this module's docstring), and it also asks the qualifying
    "First time filing with us?" question that used to live on LANGUAGE.
    That question is wrapped in an `If` in the JSON gated on this flag so a
    returning contact reopening the Flow (MENU keyword, an ERROR retry, etc.)
    doesn't get asked it again once Contact.first_time_filer is already set —
    see _handle_service_screen for where the answer is persisted."""
    return _screen_response(version, SCREEN_SERVICE, {
        "show_first_time_question": contact.first_time_filer is None,
    })


def _deep_link_screen_response(version, target_screen):
    """Response for opening the mega Flow straight at `target_screen`
    (skipping SERVICE) — the natural-language intent classifier's launch
    path (see services.send_service_flow's deep_link_screen /
    Contact.pending_flow_type_candidate in tasks.py). Reuses the exact same
    per-screen data-building as _handle_service_screen's own routing, so a
    deep-linked open can never disagree with what tapping through SERVICE
    would have shown. Returns None for an unrecognised/unsupported target so
    the caller can fall back to SERVICE rather than erroring the whole open."""
    if target_screen == SCREEN_ITR_TYPE:
        logger.info("Flow deep-link INIT resolved target_screen=%s -> screen=%s", target_screen, SCREEN_ITR_TYPE)
        return _screen_response(version, SCREEN_ITR_TYPE, {
            "itr_type_options": _rows_to_options(services.flow_itr_type_options()),
        })
    flat_entry = _FLAT_SCREEN_CONFIG_BY_SCREEN.get(target_screen)
    if flat_entry is not None:
        _category_code, cfg = flat_entry
        logger.info("Flow deep-link INIT resolved target_screen=%s -> screen=%s", target_screen, cfg["screen"])
        return _screen_response(version, cfg["screen"], {
            cfg["data_key"]: _rows_to_options(cfg["options_fn"]()),
        })
    return None


def _entry_screen_response(version, contact, target_screen=None):
    """Whatever a contact should see when the Flow opens — straight to
    SERVICE (there's no LANGUAGE screen to show first; see this module's
    docstring), or straight to `target_screen` when a deep link was
    requested (see _deep_link_screen_response). Used for both a raw INIT
    (e.g. Meta's own draft-test trigger, which carries no target_screen at
    all) and as the fallback first screen in general, so the two open paths
    can't disagree.

    contact.language is expected to already be set by this point — the
    unconditional chat-based language gate in tasks.process_incoming_message
    runs before the Flow is ever opened — but this doesn't branch on it:
    there's nowhere left to send a language-less contact within the Flow
    itself, so if that invariant is ever violated, SERVICE is shown anyway
    rather than erroring the whole open."""
    logger.info("Flow INIT for contact=%s target_screen=%r", contact.id, target_screen)
    if not contact.language:
        logger.warning(
            "Flow opened for contact=%s with no language set — showing SERVICE anyway "
            "(the Flow has no LANGUAGE screen; language should already be set via chat)",
            contact.id,
        )
    if target_screen:
        deep_link = _deep_link_screen_response(version, target_screen)
        if deep_link is not None:
            return deep_link
        logger.warning(
            "Flow deep-link INIT requested unrecognised target_screen=%s — falling back to SERVICE",
            target_screen,
        )
    return _service_screen_response(version, contact)


_FIRST_TIME_VALUES = {"yes": True, "no": False}


def _handle_service_screen(version, contact, data):
    """Also persists Contact.first_time_filer from SERVICE's conditional
    "First time filing with us?" question (see _service_screen_response's
    docstring for why it lives here now, and why it's only ever shown once)
    — a lead-routing signal only (who needs a senior consultant vs a
    template reply), never a routing decision itself, so it's read and
    saved unconditionally before the actual category routing below, same
    as it was on the old LANGUAGE screen."""
    first_time = _FIRST_TIME_VALUES.get(data.get("first_time"))
    if first_time is not None:
        contact.first_time_filer = first_time
        contact.save(update_fields=["first_time_filer"])

    category_code = data.get("selected_service")

    if category_code == "itr":
        return _screen_response(version, SCREEN_ITR_TYPE, {
            "itr_type_options": _rows_to_options(services.flow_itr_type_options()),
        })
    flat_cfg = _FLAT_CATEGORY_SCREENS.get(category_code)
    if flat_cfg is not None:
        return _screen_response(version, flat_cfg["screen"], {
            flat_cfg["data_key"]: _rows_to_options(flat_cfg["options_fn"]()),
        })

    category = None
    for code, title, _ in services.flow_category_options():
        if code == category_code:
            category = title
            break
    return _screen_response(version, SCREEN_COMING_SOON, {
        "category_name": category or "That service",
    })


def _handle_itr_type_screen(version, contact, data):
    itr_type_code = data.get("itr_type")
    service = ServicePricing.objects.filter(
        service_code=itr_type_code, is_active=True, category__category_code="itr",
    ).first()
    if service is None:
        logger.warning("ITR_TYPE submit with unknown/inactive service_code: %s", itr_type_code)
        return _error_response(version, "Please pick a valid ITR filing type and try again.")

    itr_codes = services.all_service_codes_in_category("itr")

    if service.service_code in services.ITR_ADDON_PARENT_CODES:
        # A wizard parent's own price_display (e.g. "₹999 onwards") is a
        # marketing quote, not a line-item price — the real total is
        # bundled children + addons, so the parent itself is never put on
        # the cart (unlike the flat-priced branch below, where the parent
        # *is* the whole priced product). Switching ITR type wipes any
        # addons picked for a previously chosen type first, then silently
        # bundles this type's always-free children (Basic Salary/Interest/
        # Indian Dividend etc.) — see services.auto_include_bundled_children.
        services.replace_cart_scope(contact, itr_codes, set())
        services.auto_include_bundled_children(contact, service)
        return _screen_response(version, SCREEN_ITR_ADDONS, services.flow_itr_addon_screen_data(service, contact))

    # A flat-priced ITR type (F&O, Freelancer, LLP, Partnership Filing,
    # Company, Trust) has no add-on step — one tap straight to the review.
    services.replace_cart_scope(contact, itr_codes, {service.service_code})
    return _screen_response(version, SCREEN_REVIEW, services.build_flow_review_data(contact))


def _handle_itr_addons_screen(version, contact, data):
    itr_type_code = data.get("itr_type")
    parent = ServicePricing.objects.filter(
        service_code=itr_type_code, is_active=True, category__category_code="itr",
    ).first()
    if parent is None or parent.service_code not in services.ITR_ADDON_PARENT_CODES:
        logger.warning("ITR_ADDONS submit with unknown/invalid itr_type: %s", itr_type_code)
        return _error_response(version, "Something went wrong with your ITR type — please start again.")

    addons = data.get("addons")
    if isinstance(addons, str):
        addons = [addons]
    addons = set(addons or [])

    quantity_children = services.itr_quantity_children(parent)
    quantities = {}
    house_property = quantity_children["house_property"]
    partnership = quantity_children["partnership"]
    if house_property is not None:
        qty = _parse_quantity(data.get("house_property_qty"))
        if qty is not None:
            quantities[house_property.service_code] = qty
    if partnership is not None:
        qty = _parse_quantity(data.get("partnership_qty"))
        if qty is not None:
            quantities[partnership.service_code] = qty

    # Scope = this parent's own optional children only — the parent itself
    # and its always-bundled children (added in _handle_itr_type_screen)
    # must survive this replace untouched.
    scope = {s.service_code for s in services.optional_children(parent)}
    services.replace_cart_scope(contact, scope, addons, quantities)

    return _screen_response(version, SCREEN_REVIEW, services.build_flow_review_data(contact))


def _handle_flat_category_screen(version, contact, data, screen):
    """Shared submit handler for every flat, multi-select category screen
    (GST_FILING, TDS_FILING, BUSINESS_REG, NOTICES, BOOK_EXPERT) — same
    replace-the-whole-category-scope logic as the ITR add-ons screen, just
    without a parent to scope around."""
    category_code, cfg = _FLAT_SCREEN_CONFIG_BY_SCREEN[screen]
    selected = data.get(cfg["field"])
    if isinstance(selected, str):
        selected = [selected]
    selected = set(selected or [])

    scope = services.all_service_codes_in_category(category_code)
    services.replace_cart_scope(contact, scope, selected)

    return _screen_response(version, SCREEN_REVIEW, services.build_flow_review_data(contact))


_SCREEN_HANDLERS = {
    SCREEN_SERVICE: _handle_service_screen,
    SCREEN_ITR_TYPE: _handle_itr_type_screen,
    SCREEN_ITR_ADDONS: _handle_itr_addons_screen,
    **{
        cfg["screen"]: (lambda version, contact, data, screen=cfg["screen"]: _handle_flat_category_screen(version, contact, data, screen))
        for cfg in _FLAT_CATEGORY_SCREENS.values()
    },
}


def handle_flow_action(payload):
    """Routes one decrypted data-exchange request to whatever this screen
    needs, and returns the (unencrypted) response dict. `payload` has keys
    like action/screen/data/flow_token per Meta's protocol. The endpoint
    (views.whatsapp_flow_endpoint) has already handled the "ping" health
    check before this is ever called."""
    version = payload.get("version", "3.0")
    action = payload.get("action")
    screen = payload.get("screen")
    logger.info("Flow data_exchange received: action=%s screen=%s", action, screen)

    contact = services.contact_from_flow_token(payload.get("flow_token"))
    if contact is None:
        logger.warning("Flow request with invalid/unrecognised flow_token")
        return _error_response(version, "This session has expired — please reopen the menu from WhatsApp.")

    if action == "INIT":
        target_screen = services.target_screen_from_flow_token(payload.get("flow_token"))
        result = _entry_screen_response(version, contact, target_screen)
    else:
        handler = _SCREEN_HANDLERS.get(screen)
        if handler is None:
            logger.warning("Flow data_exchange for an unrecognised screen: %s", screen)
            return _error_response(version, "Something went wrong — please close this and reopen from WhatsApp.")

        try:
            result = handler(version, contact, payload.get("data") or {})
        except Exception:
            logger.exception("Flow data_exchange handling failed (screen=%s)", screen)
            return _error_response(version, "Something went wrong on our end — please try again.")

    # Lead-scoring signal, not a routing concern — see
    # services.record_flow_progress/SCREEN_INTENT_WEIGHTS. Keyed off the
    # screen actually being shown (result["screen"]), not the one submitted
    # from, so e.g. an ITR_TYPE submit credits ITR_ADDONS (where the contact
    # ends up), not ITR_TYPE (where they came from).
    services.record_flow_progress(contact, result["screen"])

    # Abandoned-cart tracking: landing on REVIEW unconfirmed starts (or
    # restarts) the nudge clock — see services.mark_review_reached and
    # tasks.send_abandoned_cart_nudges. Centralized here rather than in each
    # of the three handlers that can return SCREEN_REVIEW.
    if result["screen"] == SCREEN_REVIEW:
        services.mark_review_reached(contact)

    return result
