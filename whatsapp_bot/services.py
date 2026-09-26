"""
Pure functions for talking to the LLM and the WhatsApp Cloud API.
Kept free of Django/Celery-specific concerns so they're easy to unit test.
"""
import difflib
import json
import logging
import re
from decimal import Decimal

import requests
from django.conf import settings
from django.core import signing
from django.utils import timezone
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from . import payments
from .models import BusinessProfile, Contact, KnowledgeBaseEntry, Order, ServiceCategory, ServicePricing

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-2.5-flash"

# How often (in total messages for a contact) to fold recent history into the
# running summary. Keeps the per-reply context window small and fast while
# retaining long-term memory of the conversation.
SUMMARY_EVERY_N_MESSAGES = 10

BOT_PERSONA = (
    "You're texting with someone on WhatsApp — reply like a warm, sharp sales "
    "consultant on the TaxPlan Advisors team would, not like a corporate support "
    "bot or a neutral help-desk. You're not just here to answer questions and "
    "hand off — you're genuinely invested in getting this person signed up and "
    "taken care of, the way a good human salesperson treats every conversation "
    "as a chance to move things forward, not just inform. Keep every reply "
    "crisp: 1-2 short sentences is the target, 3 is the hard ceiling — never "
    "explain more than the person actually asked for, and never restate what "
    "they just said back to them before answering. Use plain, everyday "
    "language and contractions; avoid jargon and technical/legal terms unless "
    "the person used them first, or the term itself is unavoidable (in which "
    "case explain it in a few plain words, not a definition).\n\n"
    "Sales mindset: answer the actual question first, honestly and completely "
    "— never inflate, oversell, or say anything untrue to close faster, that "
    "breaks trust immediately. But once it's answered, don't just stop there "
    "like a lookup tool would — read between the lines for what this person "
    "actually needs and say so plainly, the way a good consultant would spot "
    "it in a real conversation: someone mentioning a side hustle alongside a "
    "salary probably needs more than one filing; someone asking 'how much' is "
    "already leaning in, so help them get started rather than just quoting a "
    "number and waiting. Be confident and specific about the value ('that's "
    "usually a same-week turnaround' beats 'we can help with that'), and treat "
    "every genuine next step as worth naming, not just the ones they asked "
    "about. Never invent urgency, a discount, or a deadline that isn't real — "
    "the honesty rule always wins over the sales instinct.\n\n"
    "Formatting: WhatsApp does NOT render Markdown headers or **double-asterisk** "
    "bold — that shows up as literal asterisks and looks broken. Never use headers, "
    "never use **double asterisks**. If you truly need emphasis, use a single "
    "*asterisk* (WhatsApp's own bold syntax) sparingly. Prefer explaining options "
    "as a natural sentence over a numbered/bulleted list; if a list genuinely helps "
    "(4+ distinct items), keep it plain text with no bold labels.\n\n"
    "Next step: every reply should leave the person moving forward, not just "
    "informed — a good salesperson never lets a conversation stall on a fact. "
    "Where there's a real next action, end with one short action-led sentence "
    "naming it, with the action itself in *single-asterisk bold* — e.g. "
    "'*Reply HELP* and our team will confirm the exact number for you', '*Reply 1* "
    "to see ITR pricing', '*Reply HELP* to get this booked in.' Bold only the action "
    "phrase itself (2-4 words, verb-led — Reply, Confirm, Explore, Book, Share), "
    "never a whole sentence, and never invent an action that isn't actually "
    "available (only HELP, replying with a number/choice already shown, or "
    "something explicitly offered in this prompt). Skip this entirely for a pure "
    "redirect or when there's genuinely nothing actionable to point at — don't force "
    "it onto every single reply.\n\n"
    "Tone: never say things like 'As an AI, I don't have personal information' or "
    "'I understand you'd like to...' — skip the meta-commentary and just respond "
    "the way a person would. Don't over-explain, hedge, or apologize unnecessarily.\n\n"
    "Retention: every reply is one turn in an ongoing chat, not a final answer to "
    "close out — the person should always have something easy to reply to next, "
    "so the conversation keeps moving instead of going quiet. Default to ending "
    "with either the next-step action line above (when there's a real one) or a "
    "short, genuine follow-up question — only skip both when the person's message "
    "was itself a clear goodbye/thanks with nothing left to ask. Never pad a short "
    "answer with extra detail just to sound thorough — crisp and inviting beats "
    "complete; a longer reply only earns its length when the person asked "
    "something that genuinely needs it.\n\n"
    "You're on the TaxPlan Advisors team, texting on WhatsApp — don't volunteer "
    "that you're an AI, a bot, or a language model, and never describe yourself "
    "as one ('I'm a large language model...', 'as an AI system...', etc.) — "
    "that reads as a canned disclaimer, not a reply. If someone asks what you "
    "are, keep it brief and in-persona ('I'm with the TaxPlan Advisors team, "
    "here to get you sorted!') and move the conversation back to helping them, "
    "rather than explaining your own nature.\n\n"
    "Always respond to the user's most recent message first and foremost — treat "
    "the summary below (if present) as loose background, not a script to follow. "
    "Never invent menus, services, or options the user hasn't been told about; "
    "if you're unsure what they mean, just ask a short clarifying question. Never "
    "quote the user's message back inaccurately — if you reference what they just "
    "said, quote it exactly. And never play-act as if you're validating an answer "
    "against a numbered menu — you don't have access to whatever menu was shown "
    "earlier in the chat, so don't invent a valid range or say 'that's not a valid "
    "option'; just treat a short, ambiguous message like that as unclear and ask "
    "what they meant.\n\n"
    "Never invent a URL, link, or web address of any kind — only ever mention a "
    "link if it was explicitly given to you in this system prompt; you never have "
    "one available yourself. If asked how or where to pay: a secure payment link "
    "is sent automatically, as its own separate WhatsApp message, once a quote is "
    "confirmed — so say that (never invent or restate a link/amount yourself, and "
    "never claim one was already sent unless you can see it in the chat history "
    "above), and suggest replying HELP if they'd rather a person send it "
    "manually.\n\n"
    "Staying in scope: you're a tax/business filing assistant, not a general-purpose "
    "chatbot — the services list below is the complete boundary of what you help "
    "with. Judge each message on its own, don't just pattern-match keywords:\n"
    "- Fully unrelated small talk or general-knowledge asks (weather, sports scores, "
    "jokes, writing essays/poems, coding help, trivia, etc.) — don't answer the "
    "question itself, even if you know the answer. Give a brief, friendly redirect "
    "in 1-2 sentences naming a couple of things you *can* help with, e.g. 'That's a "
    "bit outside what I handle — I'm here for tax filing, GST, and business "
    "registration stuff. Anything in that space I can help with?' Don't apologize "
    "at length or explain why you can't help — just redirect and move on.\n"
    "- Real questions about taxes/business/compliance that aren't one of the listed "
    "services (e.g. international tax treaties, crypto tax, stock trading tax, a "
    "service a competitor offers) — don't guess or improvise an answer. Say it's "
    "outside what's listed here and offer to have the team confirm — suggest "
    "replying HELP.\n"
    "- Requests for definitive tax/legal advice, or anything asking you to help "
    "minimize/avoid/hide tax in a way that sounds like evasion rather than legitimate "
    "planning — don't answer with specifics either way. Say this needs a proper "
    "expert look at their actual case, and suggest replying HELP.\n"
    "- A message that's genuinely relevant even if phrased casually (e.g. 'I run an "
    "online shop, do I need GST?', 'freelancing on the side, does that need filing?') "
    "is IN scope — engage with it normally using the services/knowledge base below, "
    "don't reflexively redirect just because it wasn't phrased as a direct service "
    "request."
)

LANGUAGE_NAMES = {"en": "English", "hi": "Hindi"}

# Shown right after a contact picks a language (see tasks.py's language
# gate) — deliberately a plain lookup rather than relying on Gemini to
# translate it live, since this fires before classify_and_respond is ever
# called for this contact.
LANGUAGE_HOW_CAN_I_HELP = {
    "en": "How can I help you today?",
    "hi": "मैं आपकी कैसे मदद कर सकता/सकती हूँ?",
}

# Maps a user's free-text reply to a language code during onboarding.
# Deliberately NOT keyed on the bare ISO codes ("en"/"hi") — "hi" collides
# with the universal greeting "Hi", which would misfire as a language pick.
_LANGUAGE_CHOICES = {
    "1": "en", "english": "en",
    "2": "hi", "hindi": "hi", "हिंदी": "hi",
}

RESTART_KEYWORD = "restart"
HELP_KEYWORD = "help"
# Lighter-weight than RESTART: reopens the mega Flow (see
# tasks.process_incoming_message) without wiping language/lead_score/
# satisfaction rating/history flags the way a full RESTART does — for a
# returning customer who just wants the menu again, not a fresh profile.
MENU_KEYWORD = "menu"
# Deterministic, anytime-resolvable — see build_order_status_reply. Kept in
# the same short-keyword family as RESTART/HELP/MENU (exact match, then a
# fuzzy typo fallback) rather than routed through classify_and_respond,
# since "what's my order status" is a pure DB read with no LLM judgment
# involved.
STATUS_KEYWORD = "status"

_KEYWORD_COMMANDS = (RESTART_KEYWORD, HELP_KEYWORD, MENU_KEYWORD, STATUS_KEYWORD)
# How much longer than the keyword itself a message can be and still be
# considered for a fuzzy match, e.g. "meun"/"resatrt"/"halp" — deliberately
# small and applied to the WHOLE normalized message, never a substring,
# so a real sentence that happens to contain "help" (e.g. "please help me
# understand GST") is never deterministically hijacked away from Gemini.
_KEYWORD_FUZZY_MAX_LEN_SLACK = 2
_KEYWORD_FUZZY_CUTOFF = 0.75


def match_keyword_command(normalized_text):
    """
    Resolves a normalized (stripped+lowercased) inbound message to one of
    RESTART_KEYWORD/HELP_KEYWORD/MENU_KEYWORD — exact match first, then a
    fuzzy fallback for a likely typo of one of them. Scoped to short,
    keyword-shaped messages only (see _KEYWORD_FUZZY_MAX_LEN_SLACK): this
    never does substring matching, so it can't misfire on an ordinary
    sentence that happens to contain one of these words. Returns None if
    nothing matches closely enough — falls through to free-chat as before.
    """
    if normalized_text in _KEYWORD_COMMANDS:
        return normalized_text
    max_len = max(len(k) for k in _KEYWORD_COMMANDS) + _KEYWORD_FUZZY_MAX_LEN_SLACK
    if not normalized_text or len(normalized_text) > max_len:
        return None
    matches = difflib.get_close_matches(
        normalized_text, _KEYWORD_COMMANDS, n=1, cutoff=_KEYWORD_FUZZY_CUTOFF
    )
    return matches[0] if matches else None


def _first_name(contact):
    """First token of contact.whatsapp_display_name, or None — shared by
    _greeting_prefix and the nudge builders below. Greeting-only, never
    treated as a verified real name (see Contact.whatsapp_display_name vs
    Contact.name). First token only: full profile names can be long,
    all-caps, or contain junk, and a first-name-only greeting reads more
    natural regardless."""
    display_name = getattr(contact, "whatsapp_display_name", "") if contact else ""
    return display_name.split()[0] if display_name else None


def _greeting_prefix(contact):
    """Always the plain "👋" — deliberately never names the contact, unlike
    _named_opener below. The very first message a contact sees is also the
    least verified point to use whatsapp_display_name at: it's an unverified,
    user-editable WhatsApp profile field (see Contact.whatsapp_display_name's
    help_text), and a wrong/garbled/no-name-set case reads as impersonal or
    broken on literally the first impression. Personalizing by name is saved
    for later touches instead, once there's an actual conversation to follow
    up on — see _named_opener, used by the nudge builders. `contact` is kept
    as a parameter for call-site symmetry with _named_opener even though it's
    unused here."""
    return "👋"


def _named_opener(contact, sentence):
    """Prepends "Hi <first name>, " to `sentence` (lower-casing its first
    letter to keep the join readable) when the contact's WhatsApp profile
    name is known, else returns `sentence` unchanged — used to personalize
    the nudge builders below without duplicating each one's copy."""
    first_name = _first_name(contact)
    if not first_name:
        return sentence
    return f"Hi {first_name}, {sentence[0].lower()}{sentence[1:]}"


def build_welcome_message(contact=None):
    # Deliberately doesn't ask "How can I help you?" here — that question
    # is only meaningful once we know which language to ask it in, so it's
    # sent instead right after resolve_language_choice succeeds (see
    # tasks.py's language gate and LANGUAGE_HOW_CAN_I_HELP above).
    return (
        f"{_greeting_prefix(contact)} Welcome to TaxPlan Advisors!\n\n"
        "Please choose your preferred language:\n"
        "1️⃣ English\n"
        "2️⃣ हिंदी (Hindi)\n\n"
        "Reply with 1 or 2."
    )


def build_full_cutover_welcome_message(contact=None):
    """The plain-text welcome line tasks._send_welcome_then_flow sends
    immediately before the mega Flow trigger, once WHATSAPP_SERVICE_FLOW_ID
    is set (full cutover) — deliberately NOT build_welcome_message: that one
    asks the customer to reply 1/2 for a language, which is a legacy
    chat-wizard-only question. Under full cutover, language is asked (and
    answered) inside the Flow's own LANGUAGE screen instead (see
    send_service_flow's docstring / flow_screens.py) — repeating the same
    question as plain text right before the Flow that's about to ask it
    again natively is confusing, not helpful, and was never actually acted
    on here (this text message can't collect a reply the Flow would see
    anyway). contact.language is already known by the time this runs, so
    the "how can I help" question is asked in that language via
    LANGUAGE_HOW_CAN_I_HELP rather than hardcoded English."""
    language = contact.language if contact else ""
    how_can_i_help = LANGUAGE_HOW_CAN_I_HELP.get(language) or LANGUAGE_HOW_CAN_I_HELP["en"]
    return f"{_greeting_prefix(contact)} Welcome to TaxPlan Advisors! {how_can_i_help}"


# Shown instead of build_welcome_message() when sent as interactive buttons
# — the numbered options become the buttons themselves, so the body text
# doesn't need to spell them out too. Kept as a separate builder (rather
# than reusing build_welcome_message) so a plain-text fallback still reads
# naturally on its own if buttons ever fail to send.
def build_welcome_buttons_body(contact=None):
    return (
        f"{_greeting_prefix(contact)} Welcome to TaxPlan Advisors!\n\n"
        "Please choose your preferred language:"
    )

# ids match exactly what resolve_language_choice already accepts as typed
# text ("1"/"2") — a button tap and someone typing "1" resolve identically,
# no separate interactive-specific handling needed.
LANGUAGE_BUTTON_OPTIONS = [("1", "English"), ("2", "हिंदी")]


def resolve_language_choice(text):
    """Returns a language code ('en'/'hi') if `text` looks like an answer to
    the welcome menu, else None."""
    return _LANGUAGE_CHOICES.get(text.strip().lower())


# Only matches the language NAME, never a bare digit — unlike the onboarding
# gate above, this runs on every free-chat message, and digits are far too
# ambiguous there (a GST quantity reply, a random "2", etc.).
_LANGUAGE_SWITCH_WORDS = {"english": "en", "hindi": "hi", "हिंदी": "hi"}


def resolve_language_switch(text):
    """Detects an explicit mid-conversation language-switch request (e.g.
    'reply in English please', 'hindi mein baat karo') in free text, so
    contact.language actually updates instead of Gemini just verbally
    agreeing to switch while the stored language stays stale."""
    normalized = text.strip().lower()
    for word, code in _LANGUAGE_SWITCH_WORDS.items():
        if word in normalized:
            return code
    return None


def _loose_match(items, text, name_of, code_of):
    """Shared number-or-keyword matching used by every menu level (category,
    sub-service) — same loose-match spirit as resolve_language_choice."""
    normalized = text.strip().lower()

    # Interactive list replies send the stable row id. Check it before the
    # deliberately loose human-text matching below; otherwise every ITR code
    # shares the "itr" prefix and could resolve to the first row.
    for item in items:
        if normalized == code_of(item).lower():
            return item

    if normalized.isdigit():
        idx = int(normalized) - 1
        if 0 <= idx < len(items):
            return items[idx]

    for item in items:
        name = name_of(item).lower()
        code_stem = code_of(item).split("_")[0][:5]  # e.g. "consu" matches "consult(ation)"
        if name in normalized or normalized in name or code_stem in normalized:
            return item
    return None


# WhatsApp interactive list messages: up to 10 rows total, row title capped
# at 24 chars, row description capped at 72, the button label that opens the
# list capped at 20. _truncate keeps titles distinct rather than guessing at
# abbreviations — the full name/detail still fits in the description.
WHATSAPP_LIST_TITLE_LIMIT = 24
WHATSAPP_LIST_DESCRIPTION_LIMIT = 72
WHATSAPP_MAX_LIST_ROWS = 10


def _truncate(text, limit):
    if not text or len(text) <= limit:
        return text or ""
    return text[: limit - 3] + "..."


def build_category_menu():
    """Numbered top-level menu built fresh from ServiceCategory — admin-editable.
    Plain-text fallback; build_category_list_options() is the interactive version."""
    categories = list(ServiceCategory.objects.all())
    lines = ["Here's what we can help with:", ""]
    for i, cat in enumerate(categories, start=1):
        lines.append(f"{i}. {cat.display_name}")
    lines += ["", "Reply with a number to see pricing."]
    return "\n".join(lines)


def build_category_list_body():
    return "Here's what we can help with — tap below to see pricing:"


def build_category_list_options():
    """Row (id, title, description) triples for the category list — id is the
    1-based index, matching exactly what resolve_category_choice already
    accepts as typed text, so a tapped row needs no separate handling."""
    categories = list(ServiceCategory.objects.all())
    return [
        (str(i), _truncate(cat.display_name, WHATSAPP_LIST_TITLE_LIMIT), "")
        for i, cat in enumerate(categories, start=1)
    ]


def resolve_category_choice(text):
    """Returns the matching ServiceCategory for a category-menu reply, else None."""
    categories = list(ServiceCategory.objects.all())
    return _loose_match(
        categories, text,
        name_of=lambda c: c.display_name,
        code_of=lambda c: c.category_code,
    )


# Flattened top-level menu: when every category's actual filing options
# (ITR Salary Filing, GST Registration, ...) fit in one 10-row WhatsApp list,
# skip the "pick a category, THEN pick what within it" round trip entirely —
# show every option directly in the very first list, grouped into a section
# per category, so tapping "ITR Salary Filing" goes straight to its next step
# (add-ons, a quantity question, etc.) with no intermediate reply needed.
# Falls back to the existing two-step category menu automatically once the
# catalogue grows past what one list can hold — nothing breaks, it just stops
# being flattened.
def _all_top_level_services():
    return list(
        ServicePricing.objects.filter(is_active=True, parent_service=None).select_related("category")
    )


def flattened_menu_fits():
    return 0 < len(_all_top_level_services()) <= WHATSAPP_MAX_LIST_ROWS


def build_flattened_menu_body():
    return "Here's what we can help with — tap an option to get started:"


def build_flattened_menu_sections():
    """(section_title, options) pairs — one section per category, each row a
    top-level service. Same row shape as build_subservice_list_options."""
    by_category = {}
    for service in _all_top_level_services():
        by_category.setdefault(service.category, []).append(service)

    sections = []
    for category in sorted(by_category, key=lambda c: (c.order, c.id)):
        options = [
            (
                s.service_code,
                _truncate(s.display_name, WHATSAPP_LIST_TITLE_LIMIT),
                _truncate(f"{_price_part_for(s)} — {s.short_description}", WHATSAPP_LIST_DESCRIPTION_LIMIT),
            )
            for s in sorted(by_category[category], key=lambda s: (s.order, s.id))
        ]
        sections.append((category.display_name, options))
    return sections


def resolve_flattened_service_choice(text):
    """Returns the matching top-level ServicePricing for a flattened-menu
    reply (see build_flattened_menu_sections), else None. Deliberately
    doesn't reuse _loose_match's code-stem substring heuristic here: that's
    tuned for matching within one category's already-scoped services (e.g.
    typing "consult" for "itr_consultation"), but across every category at
    once a short generic word like "itr" or "gst" is a real category name in
    its own right (see resolve_category_choice) — code-stem matching would
    make it collide with whichever service happens to sort first instead of
    correctly falling through to the category menu."""
    normalized = text.strip().lower()
    services = _all_top_level_services()

    for service in services:
        if normalized == service.service_code.lower():
            return service

    if normalized.isdigit():
        idx = int(normalized) - 1
        if 0 <= idx < len(services):
            return services[idx]

    for service in services:
        name = service.display_name.lower()
        if name == normalized or (len(normalized) > 3 and (name in normalized or normalized in name)):
            return service
    return None


def _price_part_for(service):
    if service.question_type == ServicePricing.QUESTION_QUANTITY:
        # A catalogue can advertise the one-unit starting price (for example
        # "₹99 (₹99 x 1)") while the follow-up still calculates the real total.
        if service.price_display:
            return service.price_display
        return f"₹{service.price_per_unit:,.0f}/{service.unit_label}" if service.price_per_unit else "ask for pricing"
    return service.price_display or "ask for pricing"


def _active_services(category, parent_service=None):
    """Customer-visible items at one menu level within a category."""
    return list(category.services.filter(is_active=True, parent_service=parent_service))


def build_subservice_menu(category):
    """Numbered sub-service menu, scoped to one category, with prices shown
    directly where known (question_type='' or 'topic') or 'ask' where the
    price depends on a quantity. Plain-text fallback; build_subservice_list_options()
    is the interactive version."""
    services = _active_services(category)
    lines = [f"{category.display_name} — here's what's available:", ""]
    for i, s in enumerate(services, start=1):
        lines.append(f"{i}. {s.display_name} — {_price_part_for(s)}")
    lines += ["", "Reply with a number for details."]
    return "\n".join(lines)


def build_subservice_list_body(category):
    return f"{category.display_name} — tap an item for details:"


def build_subservice_list_options(category):
    """Row (id, title, description) triples for a category's flat sub-service
    menu — every tap here is a single, final choice (see
    build_addon_list_options for the pick-several-items case, e.g. ITR
    Salary Filing's income heads). WhatsApp permits at most ten rows; every
    catalogue today fits within that.
    """
    services = _active_services(category)
    return [
        (
            s.service_code,
            _truncate(s.display_name, WHATSAPP_LIST_TITLE_LIMIT),
            _truncate(f"{_price_part_for(s)} — {s.short_description}", WHATSAPP_LIST_DESCRIPTION_LIMIT),
        )
        for s in services[:WHATSAPP_MAX_LIST_ROWS]
    ]


def add_service_selection(contact, service, quantity=1):
    """Adds or updates one item on the contact's add-on cart."""
    selections = list(contact.selected_services or [])
    entry = {"service_code": service.service_code, "quantity": quantity}
    for index, existing in enumerate(selections):
        if existing.get("service_code") == service.service_code:
            selections[index] = entry
            break
    else:
        selections.append(entry)
    contact.selected_services = selections
    contact.save(update_fields=["selected_services"])


def remove_service_selection(contact, service):
    """Removes one item from the cart — the inverse of add_service_selection,
    used when tapping an already-added row in the add-on list again."""
    selections = [
        item for item in (contact.selected_services or [])
        if item.get("service_code") != service.service_code
    ]
    if len(selections) != len(contact.selected_services or []):
        contact.selected_services = selections
        contact.save(update_fields=["selected_services"])


def is_service_selected(contact, service_code):
    return service_code in {item.get("service_code") for item in (contact.selected_services or [])}


def _included_children(parent_service):
    """A wizard parent's always-bundled children (price_display='Included',
    e.g. Basic Salary/Interest/Indian Dividend on ITR Salary Filing) — never
    asked about, silently part of every filing under this parent."""
    return [
        s for s in _active_services(parent_service.category, parent_service)
        if s.price_display.strip().lower() == "included"
    ]


def optional_children(parent_service):
    """A wizard parent's priced add-on children — the items shown in the
    add-on list (see build_addon_list_options) and the add-on Flow (see
    build_addon_flow_payload)."""
    return [
        s for s in _active_services(parent_service.category, parent_service)
        if s.price_display.strip().lower() != "included"
    ]


def auto_include_bundled_children(contact, parent_service):
    """Silently adds every always-bundled child (see _included_children) to
    the cart the moment a wizard parent is entered — the customer is never
    asked about these; they're just part of the filing."""
    selections = list(contact.selected_services or [])
    existing_codes = {item.get("service_code") for item in selections}
    changed = False
    for child in _included_children(parent_service):
        if child.service_code not in existing_codes:
            selections.append({"service_code": child.service_code, "quantity": 1})
            changed = True
    if changed:
        contact.selected_services = selections
        contact.save(update_fields=["selected_services"])


# The add-on list: one List message showing every optional item for the
# chosen ITR parent, plus a "Confirm & Done" row. WhatsApp caps list messages
# at WHATSAPP_MAX_LIST_ROWS rows total, so build_addon_list_options() below
# defensively truncates to that cap — see its docstring. Today's catalogues
# (8 items + 1 = 9) fit comfortably under it, but a future admin edit adding
# more optional children must not be able to crash the send. Tapping an item toggles it:
# adds it if new, removes it if already on the cart. This replaces an earlier
# per-item Yes/No/Finish button wizard, which forced a full screen per item
# even to skip it — a real customer only has 2-3 relevant add-ons, so a
# skippable list means far fewer taps for the common case. It also replaces
# an even earlier attempt at "[x]"/"[ ]" checkbox text in row titles: that
# looked broken next to WhatsApp's real (static, non-interactive) tap circle.
# Instead, each row's *description* just says in plain words what tapping it
# will do next ("tap to remove" vs. the price) — no widget pretending to be
# something WhatsApp doesn't support.
ADDON_LIST_DONE_ID = "__addon_done__"


def build_addon_list_body(parent_service, contact):
    note = build_selection_note(contact)
    return f"{parent_service.display_name} — tap an add-on to include it (tap again to remove).{note}"


def build_addon_list_options(parent_service, contact):
    """Row (id, title, description) triples for the add-on list — one row
    per optional child, plus a fixed "Confirm & Done" row at the end.

    Defensively capped at WHATSAPP_MAX_LIST_ROWS total rows: WhatsApp rejects
    a list with more, and send_whatsapp_list would raise, killing the Celery
    task and leaving the customer with no reply at all — so a catalogue that
    grows past the cap must degrade (fewer visible add-ons) rather than break
    the send outright. Already-selected children are kept first so a tap the
    customer already made never silently disappears from the list — only
    un-selected overflow gets dropped.
    """
    children = optional_children(parent_service)  # one query; reused below, never re-fetched
    # Built once and reused for every row instead of is_service_selected()'s
    # per-call set rebuild — that function reconstructs the set from
    # contact.selected_services on every invocation, which added up across
    # both the truncation pass and the row-description pass below.
    selected_codes = {item.get("service_code") for item in (contact.selected_services or [])}

    max_children = WHATSAPP_MAX_LIST_ROWS - 1  # room for the trailing Confirm & Done row
    if len(children) > max_children:
        selected = [s for s in children if s.service_code in selected_codes]
        unselected = [s for s in children if s.service_code not in selected_codes]
        original_count = len(children)
        children = (selected + unselected)[:max_children]
        logger.warning(
            "build_addon_list_options: parent_service=%s has %d optional children, "
            "exceeds WhatsApp's %d-row cap — truncated to %d (selected items kept first)",
            parent_service.service_code, original_count, WHATSAPP_MAX_LIST_ROWS, len(children),
        )

    options = []
    for s in children:
        description = (
            "Added ✓ — tap to remove" if s.service_code in selected_codes
            else _truncate(f"{_price_part_for(s)} — {s.short_description}", WHATSAPP_LIST_DESCRIPTION_LIMIT)
        )
        options.append((s.service_code, _truncate(s.display_name, WHATSAPP_LIST_TITLE_LIMIT), description))
    options.append((ADDON_LIST_DONE_ID, "✅ Confirm & Done", "Review what you've selected"))
    return options


# The ITR Flow: a native WhatsApp Flow (a full-screen form, not a chat
# message) covering the entire ITR wizard journey in two screens —
# ITR_TYPE (pick Salary/Business/NRI Filing) then ADDONS (every optional
# income head as a checkbox, plus House Property/Partnership quantity
# fields, all on one screen) — ending in a single Submit. Replaces what used
# to be a category tap + a type tap + one list-tap-and-reply round trip per
# add-on + a separate "how many?" question, with at most one chat message in
# (the button that opens the Flow) and one out (the review screen), once
# WHATSAPP_ADDON_FLOW_ID is configured; the tap-list stays the default until
# then, since a Flow asset has to be created and published in Meta Business
# Manager first (see docs/whatsapp_flows/README.md — the flow itself is
# static, so no separate data-exchange endpoint is needed, just the flow id).
#
# Two entry points, same Flow:
#   - The ITR category itself ("ITR Filing") launches at ITR_TYPE, for a
#     customer who hasn't said which type yet.
#   - A specific type tap ("ITR Salary Filing" etc., from the sub-service
#     menu or the flattened top-level menu) launches straight at ADDONS,
#     skipping the now-redundant type screen entirely.
#
# Known coupling: the ADDONS screen's three `If` branches are hardcoded to
# the service codes itr_salary/itr_business/itr_nri (see
# addon_selection_flow.json and _WIZARD_ADDON_SLOT_BY_TYPE below) — a static
# Flow with no data-exchange endpoint can't discover the wizard-parent list
# at render time, so adding/renaming a 4th wizard parent means updating both
# the JSON and this mapping together, not just the database.
ITR_TYPE_FLOW_SCREEN_ID = "ITR_TYPE"
ADDONS_FLOW_SCREEN_ID = "ADDONS"
_WIZARD_ADDON_SLOT_BY_TYPE = {
    "itr_salary": "addons_salary",
    "itr_business": "addons_business",
    "itr_nri": "addons_nri",
}


def addon_flow_enabled():
    return bool(settings.WHATSAPP_ADDON_FLOW_ID)


def itr_wizard_parents():
    """The ITR sub-types that collect multiple income-head add-ons — the
    only ones offered on the Flow's ITR_TYPE screen (see the module-level
    note above on why this list is capped at what addon_selection_flow.json
    hardcodes). Every other ITR type (Freelancer, LLP, Delivery Partner,
    F&O) is flat-priced with no add-ons, so it stays a single chat tap to a
    single price reply — already as few messages as a Flow screen would be."""
    itr_category = ServiceCategory.objects.filter(category_code="itr").first()
    if not itr_category:
        return []
    return [
        s for s in _active_services(itr_category)
        if s.service_code in _WIZARD_ADDON_SLOT_BY_TYPE and s.subservices.filter(is_active=True).exists()
    ]


def _addon_option_rows(parent_service, contact=None):
    """(id, title, description) dicts for one wizard parent's flat-priced
    optional children — quantity items (House Property, Partnership) are
    deliberately excluded here; the ADDONS screen collects those through its
    own dedicated number fields instead (see build_addons_screen_flow_payload
    and tasks.py's STEP_AWAITING_INCOME_ITEM flow-reply handling)."""
    return [
        {
            "id": s.service_code,
            "title": _truncate(s.display_name, WHATSAPP_LIST_TITLE_LIMIT),
            "description": (
                _truncate("Already added — tick again to keep it", WHATSAPP_LIST_DESCRIPTION_LIMIT)
                if contact and is_service_selected(contact, s.service_code)
                else _truncate(f"{_price_part_for(s)} — {s.short_description}", WHATSAPP_LIST_DESCRIPTION_LIMIT)
            ),
        }
        for s in optional_children(parent_service)
        if s.question_type != ServicePricing.QUESTION_QUANTITY
    ]


def _all_wizard_addon_options(contact=None):
    """{"salary_addon_options": [...], "business_addon_options": [...],
    "nri_addon_options": [...]} — the ADDONS screen's data schema declares
    all three regardless of entry point (only one ever actually renders, via
    its `If` branch), so both Flow-launching functions below always send
    all three. A wizard parent missing from the catalogue just gets an empty
    list for its slot rather than a KeyError."""
    parents_by_code = {p.service_code: p for p in itr_wizard_parents()}
    result = {}
    for code, slot in _WIZARD_ADDON_SLOT_BY_TYPE.items():
        parent = parents_by_code.get(code)
        result[slot.replace("addons_", "") + "_addon_options"] = (
            _addon_option_rows(parent, contact) if parent else []
        )
    return result


def build_itr_type_flow_payload():
    """Initial data for launching the Flow at its ITR_TYPE screen — used
    when the customer taps the ITR category itself, before saying which
    type they need."""
    parents = itr_wizard_parents()
    return {
        "itr_type_options": [
            {
                "id": p.service_code,
                "title": _truncate(p.display_name, WHATSAPP_LIST_TITLE_LIMIT),
                "description": _truncate(p.short_description, WHATSAPP_LIST_DESCRIPTION_LIMIT),
            }
            for p in parents
        ],
        **_all_wizard_addon_options(),
    }


def build_addons_screen_flow_payload(parent_service, contact):
    """Initial data for launching the Flow directly at its ADDONS screen —
    used when the customer already named their ITR type via a chat tap
    (e.g. "ITR Salary Filing"), skipping the now-redundant ITR_TYPE screen."""
    return {
        "itr_type": parent_service.service_code,
        **_all_wizard_addon_options(contact),
    }


def parse_itr_flow_reply(text):
    """Parses a submitted ITR Flow reply (the raw nfm_reply.response_json
    string, passed straight through as `text` by
    views._extract_message_text) into {"itr_type", "addons",
    "house_property_qty", "partnership_qty"}. Returns None for anything
    that isn't one — a normal typed message or a tapped list row, both of
    which fall through to the existing per-item handling in tasks.py
    untouched.

    Whichever of addons_salary/addons_business/addons_nri matches itr_type
    is the real selection — the ADDONS screen's `If` branches mean the other
    two were never rendered, so their payload values (if present at all) are
    ignored regardless of content."""
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict) or "itr_type" not in data:
        return None

    itr_type = data.get("itr_type")
    slot = _WIZARD_ADDON_SLOT_BY_TYPE.get(itr_type)
    addons = data.get(slot) if slot else None
    if isinstance(addons, str):
        addons = [addons]

    def _qty(field):
        raw = data.get(field)
        if raw in (None, ""):
            return None
        match = re.search(r"\d+", str(raw))
        return int(match.group()) if match else None

    return {
        "itr_type": itr_type,
        "addons": list(addons or []),
        "house_property_qty": _qty("house_property_qty"),
        "partnership_qty": _qty("partnership_qty"),
    }


def is_addon_list_done(text):
    normalized = text.strip().lower()
    return normalized in (ADDON_LIST_DONE_ID, "done", "confirm")


# The review screen shown after "Confirm & Done" (list mode) or a Flow
# submit — a deliberate extra step rather than finalizing immediately, so a
# customer can back out and adjust before anything's treated as final. Sent
# as a list message (see CONFIRMATION_LIST_OPTIONS) rather than reply
# buttons, so it reads consistently with the add-on picker above it.
CONFIRM_SELECTION_ID = "confirm_selection"
EDIT_SELECTION_ID = "edit_selection"
CONFIRMATION_LIST_BUTTON_LABEL = "Choose an option"
CONFIRMATION_LIST_OPTIONS = [
    (CONFIRM_SELECTION_ID, "✅ Confirm", "Lock in your selected add-ons"),
    (EDIT_SELECTION_ID, "✏️ Edit more", "Go back and add or remove items"),
]
_CONFIRMATION_WORDS = {
    CONFIRM_SELECTION_ID: CONFIRM_SELECTION_ID, "confirm": CONFIRM_SELECTION_ID, "yes": CONFIRM_SELECTION_ID,
    EDIT_SELECTION_ID: EDIT_SELECTION_ID, "edit": EDIT_SELECTION_ID, "no": EDIT_SELECTION_ID,
}


def resolve_confirmation_choice(text):
    """Maps a Confirm/Edit reply to 'confirm_selection' / 'edit_selection', else None."""
    return _CONFIRMATION_WORDS.get(text.strip().lower())


# Same SUMMARY -> CONFIRM step as CONFIRMATION_LIST_OPTIONS above, for a
# standalone (non-wizard) service reached directly — a flat-priced pick, or a
# quantity/topic clarifier answered outside the add-on list. There's no
# add-on list to go "back and edit" for a single item, so this offers
# Confirm/Cancel rather than Confirm/Edit. See Contact.STEP_CONFIRMING_QUOTE.
QUOTE_CONFIRM_ID = "quote_confirm"
QUOTE_CANCEL_ID = "quote_cancel"
QUOTE_CONFIRMATION_LIST_BUTTON_LABEL = "Choose an option"
QUOTE_CONFIRMATION_LIST_OPTIONS = [
    (QUOTE_CONFIRM_ID, "✅ Confirm", "Lock in this quote and get the payment link"),
    (QUOTE_CANCEL_ID, "❌ Cancel", "Go back — I'll look at something else"),
]
_QUOTE_CONFIRMATION_WORDS = {
    QUOTE_CONFIRM_ID: QUOTE_CONFIRM_ID, "confirm": QUOTE_CONFIRM_ID, "yes": QUOTE_CONFIRM_ID,
    QUOTE_CANCEL_ID: QUOTE_CANCEL_ID, "cancel": QUOTE_CANCEL_ID, "no": QUOTE_CANCEL_ID,
}


def resolve_quote_confirmation(text):
    """Maps a standalone-quote Confirm/Cancel reply to 'quote_confirm' /
    'quote_cancel', else None."""
    return _QUOTE_CONFIRMATION_WORDS.get(text.strip().lower())


# Same Confirm/Edit shape as CONFIRMATION_LIST_OPTIONS above, for the
# natural-language intent classifier (classify_and_respond) instead of the
# add-on wizard — offered whenever a free-chat message reads as a live
# request to start a service (e.g. "I want to file my ITR"), never launched
# without this explicit confirm. Unlike CONFIRMATION_LIST_OPTIONS's
# STEP_CONFIRMING_SELECTION wiring, this is resolved anytime via
# Contact.pending_flow_type_candidate (see tasks.process_incoming_message) —
# it has to work regardless of conversation_step, since the triggering
# message can arrive from literally any point in the conversation.
FLOW_TYPE_CHOICES = ("ITR_TYPE", "GST_FILING", "TDS_FILING", "BUSINESS_REG", "NOTICES", "BOOK_EXPERT")
FLOW_TYPE_LABELS = {
    "ITR_TYPE": "ITR filing",
    "GST_FILING": "GST filing",
    "TDS_FILING": "TDS filing",
    "BUSINESS_REG": "business registration",
    "NOTICES": "notice response",
    "BOOK_EXPERT": "talking to an expert",
}
# WhatsApp's flow_cta button caps at 20 chars (send_whatsapp_flow already
# truncates to that below) — these are hand-picked to fit rather than
# derived from FLOW_TYPE_LABELS above, several of which (e.g. "business
# registration", "talking to an expert") don't. Named per flow_type so a
# deep-linked launch (see send_service_flow's deep_link_screen) shows the
# actual service being opened instead of the generic "Get started" every
# launch used to show regardless of which flow_type was confirmed — see
# send_service_flow's docstring for why that mattered.
# Cross-questioning for a low-confidence guess (see agents.low_confidence_
# clarify_node) — classify_and_respond's own `reply` for a launch_flow_
# candidate/direct_service_request is drafted as if the guess were right
# ("Got it — let's get your X sorted"), which is exactly the wrong tone to
# show the customer when Gemini itself flagged confidence="low": showing a
# confident-sounding launch prompt for a guess that might be wrong is what
# caused the original GST-misread-as-ITR bug. Cross-questioning the specific
# ambiguity per flow_type (rather than a generic "what do you mean?") gets
# a real answer back faster than guessing again — one question per category,
# covering its most common sub-cases.
LOW_CONFIDENCE_CLARIFYING_QUESTIONS = {
    "ITR_TYPE": "Just to make sure I point you the right way — is this for a "
                "salaried individual's return, a business return, or an NRI return?",
    "GST_FILING": "Just to make sure I point you the right way — do you need a "
                  "new GST registration, or filing a GST return?",
    "TDS_FILING": "Just to make sure I point you the right way — is this for "
                  "TDS return filing, or a TDS payment?",
    "BUSINESS_REG": "Just to make sure I point you the right way — what kind "
                    "of business are you looking to register (private "
                    "limited, LLP, proprietorship, or something else)?",
    "NOTICES": "Just to make sure I point you the right way — which notice "
               "is this: Income Tax, GST, or TDS?",
    "BOOK_EXPERT": "Happy to set up a call — what would you like to discuss "
                   "with our expert?",
}


def build_low_confidence_clarifying_reply(flow_type):
    """The cross-question to send instead of a launch/direct-request reply
    when classify_and_respond's confidence was "low" — see
    agents.low_confidence_clarify_node, the only caller. Falls back to a
    generic prompt for a flow_type this dict doesn't recognize (defensive
    only — FLOW_TYPE_CHOICES and this dict's keys are meant to stay in
    sync) rather than sending nothing."""
    return LOW_CONFIDENCE_CLARIFYING_QUESTIONS.get(
        flow_type,
        "Just to make sure I point you the right way — could you tell me a "
        "bit more about what you need?",
    )


FLOW_TYPE_CTA_LABELS = {
    "ITR_TYPE": "Start ITR Filing",
    "GST_FILING": "Start GST Filing",
    "TDS_FILING": "Start TDS Filing",
    "BUSINESS_REG": "Register Business",
    "NOTICES": "Respond to Notice",
    "BOOK_EXPERT": "Book a Consult",
}
FLOW_LAUNCH_CONFIRM_ID = "flow_launch_confirm"
FLOW_LAUNCH_EDIT_ID = "flow_launch_edit"
FLOW_LAUNCH_LIST_BUTTON_LABEL = "Choose an option"
FLOW_LAUNCH_LIST_OPTIONS = [
    (FLOW_LAUNCH_CONFIRM_ID, "✅ Yes, start", "Open the form to get going"),
    # Row title capped at 24 chars by the WhatsApp API itself (see
    # WHATSAPP_LIST_TITLE_LIMIT/_truncate above) — "Actually, something
    # else" alone is already 25, so every send of this option failed
    # outright with error 131009 until this was shortened.
    (FLOW_LAUNCH_EDIT_ID, "✏️ Something else", "Never mind — let's keep chatting"),
]
_FLOW_LAUNCH_WORDS = {
    FLOW_LAUNCH_CONFIRM_ID: FLOW_LAUNCH_CONFIRM_ID, "yes": FLOW_LAUNCH_CONFIRM_ID, "confirm": FLOW_LAUNCH_CONFIRM_ID,
    "yes, start": FLOW_LAUNCH_CONFIRM_ID, "start": FLOW_LAUNCH_CONFIRM_ID, "sure": FLOW_LAUNCH_CONFIRM_ID,
    FLOW_LAUNCH_EDIT_ID: FLOW_LAUNCH_EDIT_ID, "no": FLOW_LAUNCH_EDIT_ID, "cancel": FLOW_LAUNCH_EDIT_ID,
    "something else": FLOW_LAUNCH_EDIT_ID, "not now": FLOW_LAUNCH_EDIT_ID,
}

# Fallback for when the exact-string lookup above misses — real replies to
# "Want me to open X for you now?" are rarely the bare button text. Added
# after report_flow_intent_confidence surfaced a 59% "reject" rate at high
# classifier confidence that turned out to be mostly this: natural
# confirmations like "Yes pls", "Ok", "I'll go with itr" falling through the
# old exact-match dict into the "unparseable -> implicit decline" branch
# (see tasks.py's pending_flow_type_candidate handling) and getting counted
# as a rejected guess even when the flow_type was right. Matched on the
# reply's first word (so "yes I want to register it" hits "yes") or, for
# multi-word phrasing, as a whole phrase with word boundaries (so "go with"
# matches inside "I'll go with itr" but "sure" doesn't match inside
# "not sure"). Deliberately still conservative — a message with no
# recognizable lead word/phrase (e.g. someone just restating their original
# request) stays None/unparseable rather than guessing.
_FLOW_LAUNCH_CONFIRM_LEAD_WORDS = {
    "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "confirm", "correct", "right", "fine", "alright", "proceed",
}
_FLOW_LAUNCH_DECLINE_LEAD_WORDS = {"no", "nope", "nah", "cancel", "stop"}
_FLOW_LAUNCH_CONFIRM_PHRASES = (
    "go ahead", "go with", "sounds good", "let's do it", "lets do it", "let's start", "lets start",
)
_FLOW_LAUNCH_DECLINE_PHRASES = ("not now", "something else", "never mind", "nevermind", "not interested")
_WORD_RE = re.compile(r"[a-z']+")


def resolve_flow_launch_choice(text):
    """Maps a flow-launch Confirm/Edit reply to 'flow_launch_confirm' /
    'flow_launch_edit', else None."""
    normalized = text.strip().lower()
    exact = _FLOW_LAUNCH_WORDS.get(normalized)
    if exact:
        return exact

    padded = f" {normalized} "
    if any(f" {phrase} " in padded for phrase in _FLOW_LAUNCH_DECLINE_PHRASES):
        return FLOW_LAUNCH_EDIT_ID
    if any(f" {phrase} " in padded for phrase in _FLOW_LAUNCH_CONFIRM_PHRASES):
        return FLOW_LAUNCH_CONFIRM_ID

    words = _WORD_RE.findall(normalized)
    first_word = words[0] if words else ""
    if first_word in _FLOW_LAUNCH_DECLINE_LEAD_WORDS:
        return FLOW_LAUNCH_EDIT_ID
    if first_word in _FLOW_LAUNCH_CONFIRM_LEAD_WORDS:
        return FLOW_LAUNCH_CONFIRM_ID
    return None




def _selected_service_rows(contact):
    selected = contact.selected_services or []
    by_code = ServicePricing.objects.in_bulk(
        [item.get("service_code") for item in selected], field_name="service_code"
    )
    rows = []
    for item in selected:
        service = by_code.get(item.get("service_code"))
        if service:
            rows.append((service, max(1, int(item.get("quantity", 1)))))
    return rows


def _selection_price(service, quantity):
    if service.question_type == ServicePricing.QUESTION_QUANTITY and service.price_per_unit is not None:
        return service.price_per_unit * quantity, f"₹{service.price_per_unit:,.0f} x {quantity}"
    if service.price_display.lower() == "included":
        return Decimal("0"), "Included"
    match = re.search(r"₹\s*([\d,]+)", service.price_display)
    if match:
        amount = Decimal(match.group(1).replace(",", ""))
        return amount, f"₹{amount:,.0f}"
    return Decimal("0"), service.price_display or "Price on request"


# Any of these words in price_display means the number is a floor, not a
# final figure — see Contact.last_quote_is_variable and
# maybe_send_payment_cta's advance-vs-full-amount branch below.
_VARIABLE_PRICE_WORDS = ("onwards", "onward", "starting", "approx", "from ")


def _is_variable_price(service):
    return any(word in (service.price_display or "").lower() for word in _VARIABLE_PRICE_WORDS)


def compute_last_quote_amount(contact, service=None, quantity=1):
    """Resolves (amount: Decimal, is_variable: bool) for whichever branch
    just delivered a price to the customer. Pass `service` (+ `quantity`
    for a quantity-priced one) for a standalone single-service quote —
    the flat-quote and clarifier-answered branches in tasks.py. Leave both
    at their defaults to sum contact.selected_services instead — the
    multi-select cart / mega-Flow REVIEW case, where _cart_total already
    reflects exactly what was shown on screen."""
    if service is not None:
        amount, _label = _selection_price(service, quantity)
        return amount, _is_variable_price(service)
    total = Decimal("0")
    is_variable = False
    for row_service, row_quantity in _selected_service_rows(contact):
        amount, _label = _selection_price(row_service, row_quantity)
        total += amount
        if _is_variable_price(row_service):
            is_variable = True
    return total, is_variable


def record_last_quote(contact, service=None, quantity=1):
    """Snapshots whichever price was just quoted onto the contact — same
    'current state, not a log' pattern as summary/predicted_next_action.
    Call this at every point tasks.py sets trigger_conversion_cta=True,
    right before it. Read by maybe_send_payment_cta to decide the payment
    link amount without re-deriving it from a formatted reply string."""
    amount, is_variable = compute_last_quote_amount(contact, service=service, quantity=quantity)
    contact.last_quote_amount = amount
    contact.last_quote_is_variable = is_variable
    contact.save(update_fields=["last_quote_amount", "last_quote_is_variable"])


def build_selection_note(contact):
    rows = _selected_service_rows(contact)
    if not rows:
        return ""
    names = ", ".join(service.display_name for service, _ in rows)
    return f"\nSelected: {names}."


def build_resume_step_nudge(contact):
    """Short, soft one-line reminder appended to a free-chat reply once a
    contact has kept talking for tasks.RESUME_NUDGE_AFTER_TURNS turns without
    completing (or ever opening) the service Flow sent at STEP_WELCOMED — see
    tasks.process_incoming_message's free-chat branch. Points back at the
    actual saved cart when there is one (build_selection_note, same "reuse
    the real state, never let this drift out of sync" rule the cart/dropoff
    nudges already follow), a generic nudge otherwise (Flow never opened, or
    abandoned before picking anything). Deliberately appended to the existing
    reply rather than sent as its own message — one soft line, not a second
    bot message stacked onto the same turn."""
    note = build_selection_note(contact)
    if note:
        return f"\n\nBy the way, you still have this saved:{note} Reply *MENU* anytime to pick up where you left off."
    return "\n\nBy the way, reply *MENU* anytime to see our services and get started."


def build_selection_complete_reply(contact):
    """A deterministic summary of multi-selected ITR income details."""
    rows = _selected_service_rows(contact)
    if not rows:
        return "No services selected yet. Choose an item from the list first."

    total = Decimal("0")
    lines = ["Your selected income details:"]
    for service, quantity in rows:
        amount, label = _selection_price(service, quantity)
        total += amount
        suffix = f" ({quantity})" if service.question_type == ServicePricing.QUESTION_QUANTITY else ""
        lines.append(f"• {service.display_name}{suffix} — {label}")
    lines += [f"Add-on total: ₹{total:,.0f}", "Reply HELP if you'd like our team to review your filing."]
    return "\n".join(lines)


def resolve_cart_edit_target(contact, operation, query):
    """Fuzzy-matches a free-text cart-edit request (tasks.py's handling of
    classify_and_respond's edit_cart action) to a real ServicePricing, else
    None. Same 'ask rather than guess' contract as every other _loose_match
    caller in this file — a None here means the caller shows a clarifying
    reply instead of silently doing nothing or picking the wrong item.

    'remove' only matches against what's actually on the cart right now —
    removing something that was never there makes no sense to offer.
    'add' matches against the full active catalogue of leaf items (flat/
    quantity/topic services, including a wizard's individual sub-items like
    Capital Gains ITR) — wizard parents themselves (has subservices) are
    excluded, since starting one of those needs its own guided add-on
    picker, not a one-line free-text add; classify_and_respond's
    launch_flow_candidate action is what handles that instead."""
    if operation == "remove":
        candidates = [service for service, _quantity in _selected_service_rows(contact)]
    else:
        candidates = list(ServicePricing.objects.filter(is_active=True, subservices__isnull=True))
    return _loose_match(
        candidates, query,
        name_of=lambda s: s.display_name,
        code_of=lambda s: s.service_code,
    )


def build_cart_edit_not_found_reply(operation, query):
    """Deterministic clarifying reply when resolve_cart_edit_target can't
    confidently match what the customer meant — never silently no-ops."""
    if operation == "remove":
        return (
            f"I couldn't find \"{query}\" in your current cart — could you tell me "
            "the exact item, or reply MENU to see everything you've selected?"
        )
    return (
        f"I couldn't find \"{query}\" in our services — could you tell me the exact "
        "name, or reply MENU to see what we offer?"
    )


def resolve_service_choice(text, category=None, parent_service=None):
    """Returns the matching ServicePricing for a sub-service reply, scoped to
    `category` if given, else None."""
    if category is not None:
        qs = category.services.filter(is_active=True, parent_service=parent_service)
    else:
        qs = ServicePricing.objects.filter(is_active=True, parent_service=parent_service)
    return _loose_match(
        list(qs), text,
        name_of=lambda s: s.display_name,
        code_of=lambda s: s.service_code,
    )


def build_pricing_reply(service, price_override=None):
    """Deterministic pricing message for a service with no further question
    (question_type='') — the number always comes from the DB, never Gemini,
    but phrased as a sentence rather than a "label: value" line."""
    price = price_override or service.price_display
    if not price:
        return (
            f"Our team can share the latest pricing for {service.display_name} — "
            f"reply HELP and they'll get you sorted."
        )
    if service.short_description:
        return f"{service.display_name} ({service.short_description}) is {price}."
    return f"{service.display_name} is {price}."


def resolve_quantity(text):
    """Extracts the first positive integer from a free-text reply, else None
    — caller falls through to free-chat rather than looping the question."""
    match = re.search(r"\d+", text)
    if not match:
        return None
    quantity = int(match.group())
    return quantity if quantity > 0 else None


def build_quantity_reply(service, quantity):
    """Deterministic quantity x unit-price reply — the arithmetic, never Gemini,
    phrased as a sentence rather than a "label: value" line."""
    total = service.price_per_unit * quantity
    if quantity == 1:
        unit = service.unit_label
    elif service.unit_label.endswith("y"):
        unit = f"{service.unit_label[:-1]}ies"
    else:
        unit = f"{service.unit_label}s"
    return (
        f"For {quantity} {unit} of {service.display_name}, that's ₹{service.price_per_unit:,.0f} "
        f"per {service.unit_label} — ₹{total:,.0f} total."
    )


def _parse_topic_lines(service):
    return [line.strip() for line in service.topic_options.splitlines() if line.strip()]


def build_topic_question(service):
    """Numbered topic sub-menu for a service that tags (not prices) its answer.
    Plain-text fallback; build_topic_list_options() is the interactive version."""
    topics = _parse_topic_lines(service)
    lines = [service.clarifying_question, ""]
    lines += [f"{i}. {topic}" for i, topic in enumerate(topics, start=1)]
    return "\n".join(lines)


def build_topic_list_options(service):
    """Row (id, title, description) triples for a topic list — id is the
    1-based index, matching exactly what resolve_topic_choice already
    accepts as typed text."""
    topics = _parse_topic_lines(service)
    return [(str(i), _truncate(topic, WHATSAPP_LIST_TITLE_LIMIT), "") for i, topic in enumerate(topics, start=1)]


def resolve_topic_choice(service, text):
    """Matches a topic-question reply (number, or best keyword overlap) to one
    topic string. Returns None if nothing matches well enough."""
    topics = _parse_topic_lines(service)
    if not topics:
        return None

    normalized = text.strip().lower()
    if normalized.isdigit():
        idx = int(normalized) - 1
        if 0 <= idx < len(topics):
            return topics[idx]

    best, best_score = None, 0
    for topic in topics:
        words = [w for w in re.findall(r"[a-z]+", topic.lower()) if len(w) > 3]
        score = sum(1 for w in words if w in normalized)
        if score > best_score:
            best, best_score = topic, score
    return best


def build_topic_confirmation(service, topic):
    """Deterministic SUMMARY-step reply for a topic-tagged booking, shown
    alongside QUOTE_CONFIRMATION_LIST_OPTIONS — the topic is embedded in the
    saved Message text itself so it's visible to whoever picks up the
    consult (searchable in /admin/ Messages), and stays visible even though
    nothing's booked until the customer taps Confirm."""
    return (
        f"Here's your {service.display_name} on '{topic}' — {service.price_display}. "
        "Confirm to lock it in and get your payment link."
    )


def compute_lead_score(contact):
    """
    Deterministic 0-100 lead score from signals already on the Contact row —
    no LLM call, so this is cheap enough to run after every single message.

    The headline signal is flow_furthest_screen — how far into the mega
    WhatsApp Flow this contact has actually gotten (see
    SCREEN_INTENT_WEIGHTS/record_flow_progress), plus their real cart value
    if they reached REVIEW. That's "user intent, step by step" made
    concrete: every screen transition is a deliberate tap, not an inferred
    guess, so it's a much richer signal than the old single "did they ever
    reach pricing" flag it replaced.

    Weights are intentionally simple and inspectable rather than learned —
    there's no training data yet, and a transparent rule set is easier to
    trust and tune once real conversations show what actually predicts a
    conversion. Revisit the weights once you have real outcomes to check
    them against.
    """
    score = 0

    # Step-by-step intent through the mega WhatsApp Flow — see
    # SCREEN_INTENT_WEIGHTS/record_flow_progress. This is the granular
    # version of "reached pricing": a contact who configured ITR_ADDONS and
    # then went quiet scores more than one who only saw SERVICE, even though
    # neither technically finished. flow_furthest_screen is blank for a
    # contact who's never opened the Flow at all (brand new, or on the
    # legacy pre-Flow wizard) — fall back to the old binary
    # STEP_PRICING_SHOWN check for that case, since there's no per-screen
    # granularity to read there.
    if contact.flow_furthest_screen:
        score += SCREEN_INTENT_WEIGHTS.get(contact.flow_furthest_screen, 0)
        # A real, priced cart (only ever populated once they've reached
        # REVIEW with actual selections) is a stronger signal than an equally
        # "deep" contact who abandoned with nothing selected — up to +15,
        # roughly +1 per ₹1,000, so it nudges rather than dominates the score.
        cart_total = _cart_total(contact)
        if cart_total > 0:
            score += min(int(cart_total // 1000), 15)
    elif contact.conversation_step == Contact.STEP_PRICING_SHOWN:
        score += 35

    # Went deep enough to answer a quantity/topic clarifying question — more
    # deliberate than just browsing the menu. Legacy-wizard-only signal (see
    # has_answered_clarifier's own docstring) — flow_furthest_screen above
    # already captures the equivalent depth for a Flow contact.
    if contact.has_answered_clarifier:
        score += 20

    # Explicitly asked for a human — the single highest-intent action available.
    if contact.has_requested_help:
        score += 25

    # Engagement volume, capped so a long free-chat ramble can't outweigh the
    # actual buying signals above.
    message_count = contact.messages.count()
    score += min(message_count * 2, 20)

    # Recency decay — a lead that's gone quiet cools off even if it looked
    # hot a while ago.
    last_message = contact.messages.order_by("-created_at").first()
    if last_message:
        days_since = (timezone.now() - last_message.created_at).days
        if days_since > 14:
            score = int(score * 0.4)
        elif days_since > 3:
            score = int(score * 0.7)

    return min(score, 100)


def lead_tier_for_score(score):
    if score >= 60:
        return Contact.LEAD_TIER_HOT
    if score >= 30:
        return Contact.LEAD_TIER_WARM
    return Contact.LEAD_TIER_COLD


_HUMAN_ALERT_EMOJI = {
    "hot_lead": "🔥",
    "bad_csat": "😠",
    "callback_requested": "📅",
    "help_requested": "🆘",
    "payment_failed": "💳",
    "stale_link_paid": "⚠️",
}


def _post_slack_alert(contact, reason, detail):
    """Best-effort Slack Incoming Webhook post for notify_human_alert — a
    missing/unset SLACK_HOT_LEAD_WEBHOOK_URL silently no-ops (same
    optional-integration gate pattern as payments.razorpay_configured), and
    a failed POST is logged but never raised, since a Slack outage should
    never be the thing that breaks message processing. Runs synchronously
    inside the request/task, same as every other outbound call in this file
    — fine for a single Slack POST with a short timeout, unlike the
    multi-second LLM/WhatsApp calls elsewhere that already run inside Celery
    tasks."""
    webhook_url = settings.SLACK_HOT_LEAD_WEBHOOK_URL
    if not webhook_url:
        return
    emoji = _HUMAN_ALERT_EMOJI.get(reason, "🔔")
    name = contact.name or "Unknown"
    lines = [
        f"{emoji} *{reason.replace('_', ' ').title()}* — {name} ({contact.phone_number})",
        f"Lead tier: {contact.lead_tier} (score {contact.lead_score}) · Contact ID: {contact.id}",
    ]
    if detail:
        lines.append(detail)
    try:
        response = requests.post(webhook_url, json={"text": "\n".join(lines)}, timeout=5)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.error("Failed to post Slack human-alert for contact=%s: %s", contact.id, exc)


def notify_human_alert(contact, reason, detail=""):
    """
    The one integration point for "a human should look at this contact now" —
    a bad CSAT rating, a HELP request, a callback booking, a lead_tier flip
    to hot, or a payment problem. Always flags the contact for /admin/'s
    needs_human_review filter and logs a structured, greppable line, then
    additionally posts to Slack via _post_slack_alert when
    SLACK_HOT_LEAD_WEBHOOK_URL is configured — see that function and the
    setting's own comment in settings.py for setup. With no webhook
    configured, behavior is unchanged from before: the admin flag + log line
    are still the source of truth, just without a live push.
    """
    contact.needs_human_review = True
    contact.needs_human_review_reason = reason
    contact.needs_human_review_at = timezone.now()
    contact.save(update_fields=["needs_human_review", "needs_human_review_reason", "needs_human_review_at"])
    logger.warning(
        "[human-alert] contact=%s phone=%s name=%s reason=%s tier=%s detail=%s",
        contact.id, contact.phone_number, contact.name or "-", reason, contact.lead_tier, detail,
    )
    _post_slack_alert(contact, reason, detail)


def update_lead_score(contact):
    """
    Recomputes and persists lead_score/lead_tier for a contact. No LLM call —
    safe to call after every incoming message without adding latency or cost.

    Also fires notify_human_alert exactly once on a cold/warm -> hot flip
    (not on every subsequent message while already hot) — see B's "instant
    alert the moment lead_tier flips to hot". Stamps hot_lead_flipped_at /
    warm_lead_flipped_at at the moment of each respective flip — the
    customer-facing counterpart to notify_human_alert's staff-only flag, and
    the clock tasks.send_hot_lead_nudges / send_warm_lead_nudges' multi-stage
    schedules (HOT_LEAD_NUDGE_STAGES / WARM_LEAD_NUDGE_STAGES) count from.
    Resets every stage's own sent-at field on a fresh flip — same "fresh
    clock on re-entry" reasoning as services.mark_review_reached resetting
    all of DROPOFF_NUDGE_STAGES' fields — so a contact who cools off and
    re-flips hot/warm later gets the full schedule again rather than staying
    stuck on "already nudged". No cold-tier equivalent — see
    Contact.warm_lead_flipped_at's help_text for why.
    """
    score = compute_lead_score(contact)
    tier = lead_tier_for_score(score)
    if score != contact.lead_score or tier != contact.lead_tier:
        previous_tier = contact.lead_tier
        contact.lead_score = score
        contact.lead_tier = tier
        update_fields = ["lead_score", "lead_tier"]
        if tier == Contact.LEAD_TIER_HOT and previous_tier != Contact.LEAD_TIER_HOT:
            contact.hot_lead_flipped_at = timezone.now()
            contact.hot_lead_nudge_2h_sent_at = None
            contact.hot_lead_nudge_24h_sent_at = None
            contact.hot_lead_nudge_72h_sent_at = None
            update_fields += [
                "hot_lead_flipped_at", "hot_lead_nudge_2h_sent_at",
                "hot_lead_nudge_24h_sent_at", "hot_lead_nudge_72h_sent_at",
            ]
        elif tier == Contact.LEAD_TIER_WARM and previous_tier != Contact.LEAD_TIER_WARM:
            contact.warm_lead_flipped_at = timezone.now()
            contact.warm_lead_nudge_12h_sent_at = None
            contact.warm_lead_nudge_3d_sent_at = None
            contact.warm_lead_nudge_7d_sent_at = None
            update_fields += [
                "warm_lead_flipped_at", "warm_lead_nudge_12h_sent_at",
                "warm_lead_nudge_3d_sent_at", "warm_lead_nudge_7d_sent_at",
            ]
        contact.save(update_fields=update_fields)
        if tier == Contact.LEAD_TIER_HOT and previous_tier != Contact.LEAD_TIER_HOT:
            notify_human_alert(contact, "hot_lead", detail=f"score={score}")


def mark_review_reached(contact):
    """
    Records that `contact` just landed on an unconfirmed review/confirm
    screen — legacy STEP_CONFIRMING_SELECTION or the mega Flow's REVIEW (see
    flow_screens.handle_flow_action). Drives tasks.send_abandoned_cart_nudges;
    resets both dedupe timestamps so re-entering review after an edit gets a
    fresh nudge clock rather than none at all.
    """
    contact.review_reached_at = timezone.now()
    contact.cart_nudge_2h_sent_at = None
    contact.cart_nudge_24h_sent_at = None
    # Same fresh-clock reasoning for the drop-off schedule (see
    # tasks.DROPOFF_NUDGE_STAGES) — it keys off this same review_reached_at
    # signal, so its own sent-at timestamps need resetting here too.
    contact.dropoff_nudge_30m_sent_at = None
    contact.dropoff_nudge_20h_sent_at = None
    contact.dropoff_nudge_48h_sent_at = None
    contact.dropoff_nudge_7d_sent_at = None
    contact.save(update_fields=[
        "review_reached_at", "cart_nudge_2h_sent_at", "cart_nudge_24h_sent_at",
        "dropoff_nudge_30m_sent_at", "dropoff_nudge_20h_sent_at",
        "dropoff_nudge_48h_sent_at", "dropoff_nudge_7d_sent_at",
    ])


def clear_review_reached(contact):
    """Called once the contact actually confirms (or otherwise leaves) the
    review/confirm screen — stops the abandoned-cart nudge task from ever
    considering this contact again for this cart."""
    contact.review_reached_at = None
    contact.cart_nudge_2h_sent_at = None
    contact.cart_nudge_24h_sent_at = None
    contact.dropoff_nudge_30m_sent_at = None
    contact.dropoff_nudge_20h_sent_at = None
    contact.dropoff_nudge_48h_sent_at = None
    contact.dropoff_nudge_7d_sent_at = None
    contact.save(update_fields=[
        "review_reached_at", "cart_nudge_2h_sent_at", "cart_nudge_24h_sent_at",
        "dropoff_nudge_30m_sent_at", "dropoff_nudge_20h_sent_at",
        "dropoff_nudge_48h_sent_at", "dropoff_nudge_7d_sent_at",
    ])


def build_cart_nudge_message(contact, hours):
    """Deterministic abandoned-cart reminder for tasks.send_abandoned_cart_nudges
    — reuses the same per-item pricing as the review screen itself so the
    reminder can never disagree with what the contact actually saw. `hours`
    is 2 or 24: the +24h nudge reads a little more urgent (last call) since
    it's the final automated touch before the lead goes cold. Opens with the
    contact's first name when known (see _first_name), same greeting-only
    treatment as the welcome message."""
    note = build_selection_note(contact) or "\nYour selections are still saved."
    if hours >= 24:
        opener = "Still there? Your quote is still saved — don't lose it."
        closer = "Reply CONFIRM to lock it in, or HELP if you have any questions first."
    else:
        opener = "Just checking in — you were reviewing a quote earlier."
        closer = "Reply CONFIRM whenever you're ready, or HELP if anything's unclear."
    return f"{_named_opener(contact, opener)}{note}\n{closer}"


# Must match the {{quote_summary}} variable name in the approved
# WHATSAPP_CART_NUDGE_TEMPLATE_NAME template's body exactly — Meta matches
# named parameters by this string, not by position.
CART_NUDGE_TEMPLATE_PARAM_NAME = "quote_summary"


def build_cart_nudge_template_params(contact):
    """Body-variable value for the {{quote_summary}} slot in the approved
    WHATSAPP_CART_NUDGE_TEMPLATE_NAME template (see send_whatsapp_template) —
    the template-send counterpart to build_cart_nudge_message's free-form
    copy. Meta pre-approves a template's static wording, so this can't
    reproduce build_cart_nudge_message's two variants (2h vs 24h phrasing) or
    the leading "\n" formatting the free-form version uses; it just fills in
    what's currently selected, same wording build_selection_note already
    uses elsewhere."""
    rows = _selected_service_rows(contact)
    if not rows:
        value = "your quote"
    else:
        names = ", ".join(service.display_name for service, _ in rows)
        value = f"your quote for {names}"
    return [(CART_NUDGE_TEMPLATE_PARAM_NAME, value)]


def build_dropoff_nudge_message(contact, stage):
    """Deterministic message for one tasks.DROPOFF_NUDGE_STAGES checkpoint —
    same 'reuse the review screen's own pricing, never disagree with what
    they saw' approach as build_cart_nudge_message, just four escalating
    tone variants instead of two, from a quick check-in to a respectful
    final close-out. `stage` is one of "30m", "20h", "48h", "7d".

    Template fallback reuses build_cart_nudge_template_params/
    CART_NUDGE_TEMPLATE_PARAM_NAME as-is (same WHATSAPP_CART_NUDGE_TEMPLATE_NAME
    Meta template as the existing 2h/24h cart nudge) rather than requiring a
    second template submission/approval for what's fundamentally the same
    situation — an abandoned cart — at a different checkpoint."""
    note = build_selection_note(contact) or "\nYour selections are still saved."
    if stage == "30m":
        opener, closer = (
            "Quick check-in — still there?",
            "Reply CONFIRM whenever you're ready, or HELP if anything's unclear.",
        )
    elif stage == "20h":
        opener, closer = (
            "Still thinking it over? Your quote is saved and ready whenever you are.",
            "Reply CONFIRM to lock it in, or HELP if you'd like a hand deciding.",
        )
    elif stage == "48h":
        opener, closer = (
            "No rush — just didn't want this to slip through the cracks.",
            "Reply CONFIRM whenever it's convenient, or HELP if you have questions.",
        )
    else:
        # "7d": last automated touch before the lead goes cold — respectful
        # close-out, not another hard ask.
        opener, closer = (
            "Last check-in from us on this one — your quote is still here if you need it.",
            "Reply CONFIRM to go ahead, or HELP any time. Otherwise, no further nudges on this.",
        )
    return f"{_named_opener(contact, opener)}{note}\n{closer}"


# Hot-lead nudge: a customer-facing follow-up schedule for a contact whose
# lead_tier just flipped to hot (see services.update_lead_score /
# Contact.hot_lead_flipped_at) but who's gone quiet since, sent by
# tasks.send_hot_lead_nudges per tasks.HOT_LEAD_NUDGE_STAGES ("2h"/"24h"/
# "72h" — escalating tone, same shape as build_dropoff_nudge_message's
# stage variants). Deliberately distinct copy from build_cart_nudge_message
# — a hot-lead flip isn't always cart-driven (can come from pure engagement
# volume/depth, see compute_lead_score), so it shouldn't presume a saved
# quote the way the cart nudge does. Same "deterministic, not
# LLM-generated" reasoning as every other automated nudge in this file, and
# every stage pairs with CALLBACK_CTA_BUTTON_OPTIONS as the actual
# conversion action (see tasks.send_hot_lead_nudges).
def build_hot_lead_nudge_message(contact, stage):
    note = build_selection_note(contact)
    first_name = _first_name(contact)
    opener = f"Hey {first_name}" if first_name else "Hey"
    if stage == "2h":
        body = (
            f"{opener} — just checking in! Looks like you were exploring something "
            f"with us a little while ago.{note}\n"
            "Would a quick call from our team help, or would you rather keep going "
            "here in chat?"
        )
    elif stage == "24h":
        body = (
            f"{opener} — still around? Didn't want to lose the thread on what you "
            f"were looking into.{note}\n"
            "Happy to jump on a quick call whenever suits, or keep it going here — "
            "your call."
        )
    else:
        # "72h": last touch on this schedule before it goes quiet for good —
        # respectful close-out, not a third hard ask.
        body = (
            f"{opener} — last check-in from us on this one.{note}\n"
            "If you'd still like a hand, reply here or tap below any time — "
            "otherwise we'll leave it with you."
        )
    return body


# Must match the {{context}} variable name in the approved
# WHATSAPP_HOT_LEAD_TEMPLATE_NAME template's body exactly — same
# named-parameter requirement as CART_NUDGE_TEMPLATE_PARAM_NAME above. One
# template covers all of HOT_LEAD_NUDGE_STAGES (Meta's pre-approved wording
# can't vary per stage) — same "don't require a second approval for the
# same underlying situation" reasoning as build_dropoff_nudge_message
# reusing WHATSAPP_CART_NUDGE_TEMPLATE_NAME across its four stages.
HOT_LEAD_NUDGE_TEMPLATE_PARAM_NAME = "context"


def build_hot_lead_nudge_template_params(contact):
    """Body-variable value for the {{context}} slot in the approved
    WHATSAPP_HOT_LEAD_TEMPLATE_NAME template — the template-send counterpart
    to build_hot_lead_nudge_message's free-form copy, same pattern as
    build_cart_nudge_template_params. Stage-agnostic, same reasoning as the
    template itself being shared across stages."""
    rows = _selected_service_rows(contact)
    if not rows:
        value = "something you were exploring with us"
        return [(HOT_LEAD_NUDGE_TEMPLATE_PARAM_NAME, value)]
    names = ", ".join(service.display_name for service, _ in rows)
    return [(HOT_LEAD_NUDGE_TEMPLATE_PARAM_NAME, names)]


# Warm-lead nudge: one tier down from the hot-lead nudge above, on a much
# longer fuse (see tasks.WARM_LEAD_NUDGE_STAGES: "12h"/"3d"/"7d") and
# deliberately lower-pressure copy at every stage — a warm lead hasn't shown
# the same level of intent a hot one has, so this reads as a door staying
# open rather than a sales push, and never leads with "book a call" the way
# the hot schedule does. No cold-tier equivalent (see
# Contact.warm_lead_flipped_at).
def build_warm_lead_nudge_message(contact, stage):
    note = build_selection_note(contact)
    first_name = _first_name(contact)
    opener = f"Hey {first_name}" if first_name else "Hey"
    if stage == "12h":
        body = (
            f"{opener} — just checking in, no rush at all! We're here whenever "
            f"you're ready to pick this back up.{note}\n"
            "Reply HELP anytime if you'd like a hand, or just message us when "
            "you're ready."
        )
    elif stage == "3d":
        body = (
            f"{opener} — still here whenever it's convenient for you.{note}\n"
            "No pressure at all — just message us when you're ready, or HELP if "
            "you want a hand."
        )
    else:
        # "7d": last touch on this schedule — respectful close-out, matching
        # build_dropoff_nudge_message's "7d" tone.
        body = (
            f"{opener} — last check-in from us on this one, no rush ever.{note}\n"
            "The door's always open — message us any time you're ready, or HELP "
            "if you'd like a hand."
        )
    return body


# Must match the {{context}} variable name in the approved
# WHATSAPP_WARM_LEAD_TEMPLATE_NAME template's body exactly — same
# named-parameter requirement as the other nudge templates.
WARM_LEAD_NUDGE_TEMPLATE_PARAM_NAME = "context"


def build_warm_lead_nudge_template_params(contact):
    """Body-variable value for the {{context}} slot in the approved
    WHATSAPP_WARM_LEAD_TEMPLATE_NAME template — the template-send
    counterpart to build_warm_lead_nudge_message's free-form copy, same
    pattern as build_hot_lead_nudge_template_params."""
    rows = _selected_service_rows(contact)
    if not rows:
        value = "something you were exploring with us"
        return [(WARM_LEAD_NUDGE_TEMPLATE_PARAM_NAME, value)]
    names = ", ".join(service.display_name for service, _ in rows)
    return [(WARM_LEAD_NUDGE_TEMPLATE_PARAM_NAME, names)]


def build_referral_message(contact):
    """Deterministic post-booking referral ask (services.notify_human_alert's
    sibling on the growth side) — sent once, at most every 30 days, a while
    after a booking actually completes (see tasks.send_referral_ask)."""
    return (
        "Glad we could help! If you know anyone else who could use our services, "
        "send them our way — you'll both get 10% off your next filing. "
        "Just have them mention your number when they reach out."
    )


# Formerly: how long the bot stayed silent after a customer explicitly asked
# for a human (HELP, or completing a callback booking) — see
# Contact.bot_paused_until. That pause feature was removed by explicit
# request (tasks.process_incoming_message no longer gates on it, and nothing
# sets it anymore); this constant is kept only because Contact.bot_paused_until
# and its migration still exist for historical rows and the admin.py display.
HUMAN_HANDOFF_PAUSE_HOURS = 4


def build_help_reply():
    """
    Deterministic human-handoff reply for the HELP keyword — doesn't depend
    on Gemini deciding to escalate, so it always works the same way.
    """
    profile = BusinessProfile.objects.first()
    if profile and profile.support_contact:
        return (
            f"No worries — I'm connecting you with our support team. "
            f"You can also reach them directly at {profile.support_contact}."
        )
    return "Connecting you with our support team now — they'll be with you shortly."


# Conversion CTA — asked as a follow-up button message every time the bot
# delivers a concrete outcome (a price quote, a completed income-wizard
# summary, a booking confirmation), *before* CSAT. This is the actual
# conversion ask: a quote on its own doesn't capture a lead, "book a
# callback" does. Anytime-resolvable, same pattern as CSAT/confirmation
# choices below — works regardless of conversation_step.
#
# Sent right after maybe_send_payment_cta (see tasks.py's trigger_conversion_cta
# block). Only one button now — "I'll pay now" was dropped since the payment
# link/CTA is already sent alongside this message, so a separate "I'll pay
# now" tap didn't add anything. KEEP_CHATTING_ID and its word synonyms are
# kept in _CALLBACK_CTA_WORDS below (and the tasks.py branch that handles it)
# purely so an existing customer who still types "pay"/"chat" from muscle
# memory resolves correctly — there's just no button offering it anymore.
CALLBACK_CTA_BODY = "Are you facing any difficulty? Want me to connect you with our support team?"
BOOK_CALLBACK_ID = "book_callback"
KEEP_CHATTING_ID = "keep_chatting"
CALLBACK_CTA_BUTTON_OPTIONS = [
    (BOOK_CALLBACK_ID, "📅 Book a callback"),
]
_CALLBACK_CTA_WORDS = {
    BOOK_CALLBACK_ID: BOOK_CALLBACK_ID, "callback": BOOK_CALLBACK_ID, "call me": BOOK_CALLBACK_ID,
    KEEP_CHATTING_ID: KEEP_CHATTING_ID, "chat": KEEP_CHATTING_ID,
    "pay": KEEP_CHATTING_ID, "i'll pay now": KEEP_CHATTING_ID, "pay now": KEEP_CHATTING_ID,
}


def resolve_callback_choice(text):
    """Maps a callback-CTA reply to 'book_callback' / 'keep_chatting', else None."""
    return _CALLBACK_CTA_WORDS.get(text.strip().lower())


def maybe_send_payment_cta(contact, summary=None):
    """
    Sends a "Pay Now" CTA-URL message — but only when every gate in the
    payment-link decision logic passes. Returns True when it actually sent
    something, False otherwise (no amount, Razorpay not configured, or link
    creation failed); callers use this to know whether they still owe the
    contact their own plain-text reply.

    `summary`, when given, is folded straight into the CTA body ahead of
    the payment ask (e.g. the deterministic order summary from
    build_selection_complete_reply) so checkout-confirm produces exactly
    one outbound WhatsApp message instead of a separate summary message
    followed by a separate CTA — see tasks.py's trigger_conversion_cta
    handling, the caller for this. Pass nothing to send the payment ask on
    its own (e.g. tasks.send_abandoned_cart_nudges re-surfacing an
    already-open link, where there's no fresh summary to attach).

    Gates, in order:
      1. Hard: a resolved amount actually exists (record_last_quote sets
         this to 0 for a human-handoff quote with no number) — no amount,
         no link, full stop.
      2. Hard: Razorpay is configured — dev/staging without keys skips this
         silently rather than erroring on every quote.
      3. Reuse: an already-open, unpaid link for this exact quote is reused
         rather than minting a second one — avoids the "which link do I
         pay" confusion of two live links for the same cart.

    Deliberately NOT gated on needs_human_review: whatever unresolved
    concern put the contact under review (bad CSAT, a HELP request, a
    hot-lead flip), a customer who just checked out and was quoted a total
    should always get the way to pay it. The human/support path (HELP,
    the hot-lead nudge, the abandoned-cart nudge timers — see
    CALLBACK_CTA_BODY's call sites) is deliberately no longer fired
    unconditionally alongside this; it only ever fires from its own
    trigger, never inline here.
    """
    amount = contact.last_quote_amount
    if not amount or amount <= 0:
        return False
    if not payments.razorpay_configured():
        return False

    snapshot = contact.selected_services or []
    existing = Order.objects.filter(
        contact=contact, status=Order.STATUS_CREATED, quote_snapshot=snapshot,
    ).order_by("-created_at").first()

    if existing and existing.payment_link_url:
        order = existing
    else:
        is_advance = contact.last_quote_is_variable
        order = Order.objects.create(
            contact=contact,
            quote_snapshot=snapshot,
            amount=settings.PAYMENT_ADVANCE_AMOUNT if is_advance else amount,
            is_advance=is_advance,
        )
        if not payments.create_payment_link(order):
            order.status = Order.STATUS_FAILED
            order.failure_reason = "link_creation_failed"
            order.save(update_fields=["status", "failure_reason"])
            return False

    if order.is_advance:
        ask = (
            f"To lock in your slot, pay a ₹{order.amount:,.0f} advance now — "
            "we'll confirm the final scope and settle the balance after that."
        )
        button_text = "💳 Pay booking fee"
    else:
        ask = f"Ready to go? Pay ₹{order.amount:,.0f} securely now and we'll start right away."
        button_text = "💳 Pay Now"

    body = f"{summary}\n\n{ask}" if summary else ask
    send_whatsapp_cta_url(contact.phone_number, body, button_text, order.payment_link_url)
    return True


def invalidate_stale_orders(contact):
    """Expires every still-open (STATUS_CREATED) payment link for `contact`
    — called right after a cart edit (see tasks.py's edit_cart handling)
    so an old link can never be paid at a now-wrong amount once the cart it
    was quoted against no longer matches what's actually selected. This is
    a deliberate reversal of Order's original 'never touched by anything in
    the chat flow' guarantee — see the model's docstring — now that a cart
    can change out from under an already-issued link via free-text editing,
    not just be re-confirmed as a fresh, distinct order.

    Best-effort against the gateway (payments.cancel_payment_link never
    raises) — even if that call fails, the order is still marked
    STATUS_EXPIRED here so maybe_send_payment_cta's exact-snapshot reuse
    check can never pick it back up, and tasks.process_payment_webhook
    flags (rather than silently completes) a payment landing on an order
    already expired this way.
    """
    for order in Order.objects.filter(contact=contact, status=Order.STATUS_CREATED):
        payments.cancel_payment_link(order)
        order.status = Order.STATUS_EXPIRED
        order.failure_reason = "cart_edited"
        order.save(update_fields=["status", "failure_reason"])


# --- Order & Status (chat-facing) -------------------------------------------
# Deterministic, read-only against Order.status (payment status, reconciled
# from Razorpay via tasks.process_payment_webhook) — no LLM in the decision
# path, same "pure DB read" contract as CSAT/callback below. Deliberately
# scoped to payment status only; there's no separate filing/processing
# status field (and no workflow behind one yet).

RECENT_ORDERS_LIMIT = 5

ORDER_STATUS_LABELS = {
    Order.STATUS_CREATED: "🕐 Awaiting payment",
    Order.STATUS_PAID: "✅ Paid",
    Order.STATUS_FAILED: "❌ Payment failed",
    Order.STATUS_EXPIRED: "⌛ Link expired",
}

# Row/button ids are prefixed strings, not bare numbers — so they can never
# collide with the digit-based ids used elsewhere (category index, quantity,
# etc.) and are trivially recognized with text.startswith(...) regardless of
# conversation_step, same "anytime-resolvable" tier as CSAT/callback.
ORDER_STATUS_SELECT_PREFIX = "order_status:"
ORDER_STATUS_RESEND_PREFIX = "resend_payment_link:"

# Statuses where a fresh/resent payment link actually makes sense — never
# offered on an already-paid order.
_ORDER_STATUS_RESENDABLE = (Order.STATUS_CREATED, Order.STATUS_FAILED, Order.STATUS_EXPIRED)


def _recent_orders(contact):
    return list(contact.orders.all()[:RECENT_ORDERS_LIMIT])


def build_single_order_status_reply(order):
    """(text, reply_buttons) for one order — reply_buttons is the usual
    (body_text, options) pair (see tasks.py's send-dispatch block), body_text
    same as `text` here. Always offers a human handoff; adds a 'get payment
    link' button only when the order is actually resendable."""
    text = (
        f"Order #{order.id} — ₹{order.amount:,.0f}\n"
        f"Status: {ORDER_STATUS_LABELS.get(order.status, order.status)}"
    )
    options = []
    if order.status in _ORDER_STATUS_RESENDABLE:
        options.append((f"{ORDER_STATUS_RESEND_PREFIX}{order.id}", "💳 Get payment link"))
    # id is literally HELP_KEYWORD so a tap re-enters the existing HELP
    # branch (tasks.process_incoming_message) with no new handling needed.
    options.append((HELP_KEYWORD, "🙋 Talk to a person"))
    return text, (text, options)


def build_order_status_reply(contact):
    """
    (reply_text, reply_buttons, reply_list) for the STATUS keyword —
    exactly one of reply_buttons/reply_list is ever set, matching
    tasks.py's existing unpack-and-check-each convention.

    0 orders -> plain text + a human-handoff button.
    1 order -> build_single_order_status_reply's buttons.
    2+ orders -> a selectable list (see resolve_order_status_selection).
    """
    orders = _recent_orders(contact)
    if not orders:
        text = (
            "You don't have any orders with us yet — reply MENU to see our services, "
            "or tap below to talk to our team."
        )
        return text, (text, [(HELP_KEYWORD, "🙋 Talk to a person")]), None
    if len(orders) == 1:
        text, buttons = build_single_order_status_reply(orders[0])
        return text, buttons, None

    options = [
        (
            f"{ORDER_STATUS_SELECT_PREFIX}{order.id}",
            _truncate(f"Order #{order.id} — {ORDER_STATUS_LABELS.get(order.status, order.status)}", WHATSAPP_LIST_TITLE_LIMIT),
            f"₹{order.amount:,.0f}",
        )
        for order in orders
    ]
    return "Which order?", None, ("Which order?", "View orders", options)


def resolve_order_status_selection(text, contact):
    """Matches an ORDER_STATUS_SELECT_PREFIX-prefixed id (from either a list
    row or a resend-button tap, once the caller has normalized the prefix)
    against `contact`'s own orders. Ownership is enforced by scoping the
    lookup to contact.orders — an id belonging to a different contact never
    matches, so this never trusts a bare user-supplied order id."""
    if not text.startswith(ORDER_STATUS_SELECT_PREFIX):
        return None
    raw_id = text[len(ORDER_STATUS_SELECT_PREFIX):]
    if not raw_id.isdigit():
        return None
    return contact.orders.filter(id=int(raw_id)).first()


def resend_or_recreate_payment_link(order):
    """
    Gets a working payment link back in front of the customer for a
    CREATED/FAILED/EXPIRED order. Mirrors maybe_send_payment_cta's send
    tail. Returns whether a message was actually sent.

    CREATED with a link already issued: resend that same URL — no new
    Razorpay object, avoids the "which link do I pay" confusion a second
    live link for the same order would cause.

    FAILED/EXPIRED: never mutate the old order in place (same rule
    invalidate_stale_orders already follows) — clone amount/quote_snapshot/
    is_advance onto a brand-new Order and create a fresh link on that.
    """
    contact = order.contact
    if order.status == Order.STATUS_CREATED and order.payment_link_url:
        target = order
    else:
        target = Order.objects.create(
            contact=contact,
            quote_snapshot=order.quote_snapshot,
            amount=order.amount,
            is_advance=order.is_advance,
        )
        if not payments.create_payment_link(target):
            target.status = Order.STATUS_FAILED
            target.failure_reason = "link_creation_failed"
            target.save(update_fields=["status", "failure_reason"])
            return False

    button_text = "💳 Pay booking fee" if target.is_advance else "💳 Pay Now"
    body = f"Here's a fresh link for order #{target.id} — ₹{target.amount:,.0f}."
    send_whatsapp_cta_url(contact.phone_number, body, button_text, target.payment_link_url)
    return True


# Callback slot picker — shown once "book_callback" is tapped. Deliberately a
# fixed 3-option pick, not free text: deterministic to store/read in /admin/,
# no parsing ambiguity, and fast to tap on WhatsApp.
CALLBACK_SLOT_BODY = "When's best for a call?"
CALLBACK_SLOT_BUTTON_OPTIONS = [
    (f"slot_{Contact.CALLBACK_SLOT_TODAY}", "Today"),
    (f"slot_{Contact.CALLBACK_SLOT_TOMORROW}", "Tomorrow"),
    (f"slot_{Contact.CALLBACK_SLOT_THIS_WEEK}", "This week"),
]
_CALLBACK_SLOT_IDS = {f"slot_{choice}": choice for choice, _ in Contact.CALLBACK_SLOT_CHOICES}


def resolve_callback_slot(text):
    """Maps a callback-slot button tap to a Contact.CALLBACK_SLOT_* value, else None."""
    return _CALLBACK_SLOT_IDS.get(text.strip().lower())


def build_callback_confirmation(contact, slot):
    """Deterministic confirmation once a callback slot is picked — includes
    BusinessProfile.support_contact if set, same pattern as build_help_reply,
    never inventing a number that isn't configured."""
    slot_label = dict(Contact.CALLBACK_SLOT_CHOICES).get(slot) or slot or "soon"
    profile = BusinessProfile.objects.first()
    if profile and profile.support_contact:
        return (
            f"Got it — our customer support team will call you {slot_label.lower()}. "
            f"You can also reach them directly at {profile.support_contact} in the meantime."
        )
    return f"Got it — our customer support team will call you {slot_label.lower()}."


# CSAT — asked as a follow-up button message every time the bot delivers a
# concrete outcome (a price quote, a completed income-wizard summary, a
# booking confirmation). Ids double as the values stored directly on
# Contact.satisfaction_rating.
CSAT_PROMPT_BODY = "Quick one — how was that?"
CSAT_BUTTON_OPTIONS = [
    (Contact.SATISFACTION_GREAT, "😊 Great"),
    (Contact.SATISFACTION_OKAY, "😐 Okay"),
    (Contact.SATISFACTION_BAD, "😞 Not really"),
]
_CSAT_IDS = {choice for choice, _ in Contact.SATISFACTION_CHOICES}


def resolve_csat_choice(text):
    """Maps a CSAT button tap to a Contact.SATISFACTION_* value, else None."""
    normalized = text.strip().lower()
    return normalized if normalized in _CSAT_IDS else None


def build_csat_thank_you(rating):
    """Deterministic acknowledgement for a CSAT rating. A low rating
    proactively offers a human instead of just saying thanks — the whole
    point of asking is to catch and act on a bad experience."""
    if rating == Contact.SATISFACTION_BAD:
        return "Sorry to hear that — reply HELP and our team will sort it out for you directly."
    return "Thanks for letting us know! 🙌"


# Reused across calls instead of rebuilding per-request (avoids repeated client setup cost).
_gemini_client = None

# A WhatsApp chat bot needs a fast, conversational reply, not deep multi-step reasoning —
# gemini-2.5-flash's default "thinking" mode adds several extra seconds per call for that,
# so it's turned off here. Re-enable (raise thinking_budget) if replies need to get smarter
# at the cost of latency.
_THINKING_CONFIG = genai_types.ThinkingConfig(thinking_budget=0)


def _get_client():
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _gemini_client


def _log_gemini_usage(call_name, response):
    """
    Logs token usage for one generate_content call, cached_content_token_count
    included — a temporary instrumentation pass to measure the real implicit-
    cache hit rate on our system_instruction prefix (persona + business
    context) before spending any effort re-ordering it for caching. Never
    raises: usage_metadata is best-effort telemetry, not something that
    should be able to take down a reply on a shape we didn't expect.
    """
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return
    try:
        logger.info(
            "[gemini-usage] call=%s prompt_tokens=%s cached_tokens=%s output_tokens=%s total_tokens=%s",
            call_name,
            getattr(usage, "prompt_token_count", None),
            getattr(usage, "cached_content_token_count", None),
            getattr(usage, "candidates_token_count", None),
            getattr(usage, "total_token_count", None),
        )
    except Exception:
        logger.debug("[gemini-usage] call=%s: failed to log usage_metadata", call_name, exc_info=True)


def _build_business_context():
    """
    Reads the BusinessProfile fresh from the DB each call (cheap single-row
    lookup) so edits made in /admin/ take effect immediately — no restart
    needed, unlike code changes to the persona itself.
    """
    profile = BusinessProfile.objects.first()
    if not profile:
        return ""

    lines = [f"You represent {profile.name}."]
    if profile.tagline:
        lines.append(f"Tagline: {profile.tagline}")

    services = [s.strip() for s in profile.services.splitlines() if s.strip()]
    if services:
        lines.append("Services actually offered (do not mention anything outside this list):")
        lines.extend(f"- {s}" for s in services)

    if profile.pricing_note:
        lines.append(f"Pricing: {profile.pricing_note}")
    else:
        lines.append(
            "Pricing: not fixed here — if asked, say it varies by case and offer "
            "to connect them with the team for exact numbers. Never invent a price."
        )

    if profile.hours_note:
        lines.append(f"Availability: {profile.hours_note}")

    if profile.do_not_instructions:
        lines.append(f"Important rules: {profile.do_not_instructions}")

    if profile.support_contact:
        lines.append(
            f"If the user is confused, stuck, or asks for a human, tell them to reply "
            f"HELP or reach the support team directly at {profile.support_contact}."
        )

    return "\n".join(lines)


# Words this short (the, and, for, you, gst's own short forms aside) are too
# generic to be a useful relevance signal — same threshold already used by
# resolve_topic_choice for the same reason.
_KEYWORD_MIN_LEN = 3

# How many KB entries to inject per reply, out of however many are active.
# Keeps the system prompt focused on what this message actually needs
# instead of growing with every document added via the loader — a message
# about one topic doesn't need the other 15 entries sitting in context.
KB_RETRIEVAL_TOP_K = 4


def _keywords(text):
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > _KEYWORD_MIN_LEN}


# Below this score an entry is treated as noise rather than a real match —
# without this, a single generic shared word (e.g. "policy" matching every
# policy document) was enough to drag in an unrelated chunk. Title matches
# count for 3x a body match since the title is a much stronger topic signal
# than one word buried in a paragraph.
_KB_RELEVANCE_THRESHOLD = 3
_KB_TITLE_MATCH_WEIGHT = 3


def _select_relevant_kb_entries(entries, text, history=None):
    """
    Lightweight keyword-overlap ranking — word-set intersection between the
    message (plus the last couple of turns, so a one-word follow-up like
    'how much?' still inherits the topic) and each entry's title+content,
    with title matches weighted higher than body matches. No embeddings, no
    extra API call, so this adds zero latency or cost.

    Good enough at this KB size (a few dozen chunks of fairly distinctive
    tax/business vocabulary — "GST", "refund", "TDS" don't collide much) as
    long as a real relevance threshold is enforced (see
    _KB_RELEVANCE_THRESHOLD) rather than accepting any overlap at all. If the
    KB grows much larger, or entries start overlapping in generic wording,
    that's the point to switch to real embeddings-based retrieval instead of
    tuning this further.
    """
    query_words = _keywords(text)
    if history:
        for msg in history[:2]:
            query_words |= _keywords(msg.body)
    if not query_words:
        return []

    scored = []
    for entry in entries:
        title_overlap = len(query_words & _keywords(entry.title))
        body_overlap = len(query_words & _keywords(entry.content))
        score = title_overlap * _KB_TITLE_MATCH_WEIGHT + body_overlap
        if score >= _KB_RELEVANCE_THRESHOLD:
            scored.append((score, entry))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [entry for _, entry in scored[:KB_RETRIEVAL_TOP_K]]


def _build_knowledge_base_context(text, history=None):
    """
    Reads active KnowledgeBaseEntry rows fresh from the DB each call — same
    "no restart needed for content edits" pattern as _build_business_context.
    Only injects entries relevant to this specific message (see
    _select_relevant_kb_entries) instead of the whole table.
    """
    entries = list(KnowledgeBaseEntry.objects.filter(is_active=True))
    if not entries:
        return ""

    relevant = _select_relevant_kb_entries(entries, text, history)
    if not relevant:
        return ""

    lines = [
        "Knowledge base — answer from this faithfully when it's relevant to what "
        "the user asked; don't contradict it, and don't invent details it doesn't cover. "
        "These entries are raw excerpts from source documents (policy PDFs etc.) — they "
        "may contain numbered sections, headers, or bullet symbols from the original "
        "document. Never reproduce that structure or transcribe a section near-verbatim; "
        "always paraphrase into 2-4 short plain sentences covering just what this person "
        "asked, in your own words, the way you'd explain it out loud. Even if more than "
        "one entry below is relevant, don't recite all of them — pick what's most "
        "on-point for this exact message and drop the rest. If what they asked is "
        "genuinely a sequence of steps THEY need to take (e.g. 'how do I cancel'), a "
        "short plain numbered list (max 3-4 steps, no bold, no sub-bullets) is fine — "
        "otherwise write it as normal sentences, not a list. This is general policy/FAQ "
        "info, not a confirmation of any one person's specific case — so every time you "
        "answer using this, always end with a short offer to reply HELP or talk to the "
        "team for their specific situation, even if the answer itself felt complete. Keep "
        "that offer to one brief trailing sentence, don't turn it into a sales pitch."
    ]
    for entry in relevant:
        lines.append(f"\nQ: {entry.title}\nA: {entry.content}")
    return "\n".join(lines)


def _build_system_instruction(contact, text, history=None):
    instruction = BOT_PERSONA

    business_context = _build_business_context()
    if business_context:
        instruction += f"\n\n{business_context}"

    knowledge_base_context = _build_knowledge_base_context(text, history)
    if knowledge_base_context:
        instruction += f"\n\n{knowledge_base_context}"

    if contact.language:
        lang_name = LANGUAGE_NAMES.get(contact.language, contact.language)
        instruction += f"\n\nAlways reply in {lang_name}, regardless of what language the user writes in."
    if contact.summary:
        instruction += f"\n\nWhat you remember about this contact so far:\n{contact.summary}"

    # STEP_WELCOMED means the service-selection WhatsApp Flow (a native
    # full-screen form) was just sent as a one-shot trigger (see
    # tasks.process_incoming_message's "FULL CUTOVER" branch) — this reply is
    # only reached if the person typed ordinary text instead of completing
    # it, which happens both for "ignoring the Flow to ask something else"
    # AND for "the Flow never actually opened for them" (unpublished/draft
    # Flow, a client that doesn't support Flows, etc.) — there's no way to
    # tell those two apart from the text alone. Without this note the model
    # has zero awareness a menu/form was ever sent and, when the person
    # sounds like they're reacting to an invisible menu, guesses ("I can't
    # see the options on your screen") and invents specific service names —
    # see the 2026-09-02 conversation trace where that produced a fabricated
    # service name. MENU_KEYWORD's chat-based menu is the guaranteed-visible
    # fallback either way, so steer there instead of guessing.
    if contact.conversation_step == Contact.STEP_WELCOMED:
        instruction += (
            "\n\nContext: right before this message, this person was sent our service "
            "menu as a WhatsApp form (a Flow) — you have no visibility into whether it "
            "opened for them or what it showed, so never guess at, describe, or reference "
            "'the screen'/'the options you're seeing' as if you could see it, and never "
            "invent a specific service name. If their message suggests they're confused, "
            "stuck, or reacting to a menu they can't act on, just say the form might not "
            "have opened and tell them to reply MENU to get the list as regular text "
            "instead. If their message is a genuine, unrelated question, just answer it "
            "normally — don't mention the menu at all."
        )

    # Lets classify_and_respond's edit_cart action (see _CLASSIFICATION_INSTRUCTION)
    # recognize a plain-text cart change ("remove capital gains", "add house
    # property too") — and, just as important, tells it NOT to offer edit_cart
    # when there's nothing to edit yet (contact.selected_services empty), same
    # "only real when the state supports it" rule launch_flow_candidate follows.
    cart_rows = _selected_service_rows(contact)
    if cart_rows:
        names = ", ".join(service.display_name for service, _quantity in cart_rows)
        instruction += (
            f"\n\nThis contact currently has the following in their cart: {names}. "
            "They can ask, in plain text, to add or remove something from it at any "
            "time (see the edit_cart action below) — this doesn't have to be a formal "
            "request, e.g. 'actually remove the capital gains one' or 'can you add "
            "house property too' both count."
        )
    return instruction


def _build_gemini_contents(text, history=None):
    """Shared Content-list builder for every live per-message Gemini call
    (get_ai_response, classify_and_respond) — oldest-first history plus this
    turn's message."""
    contents = [
        genai_types.Content(
            role="user" if msg.direction == "in" else "model",
            parts=[genai_types.Part(text=msg.body)],
        )
        for msg in reversed(history or [])  # oldest first
    ]
    contents.append(genai_types.Content(role="user", parts=[genai_types.Part(text=text)]))
    return contents


def get_ai_response(contact, text, history=None):
    """
    Calls Google's Gemini API for a reply.
    `history` can be a list of prior Message objects (newest first) for conversation context.
    `contact.summary`, if set, is folded in as long-term memory via the system instruction.
    Falls back to a placeholder if no GEMINI_API_KEY is configured or the call fails.

    Used by the offline persona eval (management command eval_ai_persona) and
    nowhere in the live per-message path anymore — see classify_and_respond,
    which folds this same call into a structured reply+classification result
    for tasks.process_incoming_message, so a live message never triggers both.
    """
    if not settings.GEMINI_API_KEY:
        return f"Thanks for your message: '{text}'. (AI response not yet configured.)"

    config = genai_types.GenerateContentConfig(
        thinking_config=_THINKING_CONFIG,
        max_output_tokens=500,
        # Cooler than Gemini's ~1.0 default — this reply can state prices,
        # scope boundaries, and what happens next, so lower variance matters
        # more here than for summarize_conversation's internal CRM notes.
        # Still warm/conversational at this level (persona tone comes from
        # BOT_PERSONA's wording, not from sampling temperature) — just less
        # likely to drift into an unlikely, ungrounded phrasing.
        temperature=0.4,
        system_instruction=_build_system_instruction(contact, text, history),
    )

    try:
        response = _get_client().models.generate_content(
            model=GEMINI_MODEL, contents=_build_gemini_contents(text, history), config=config
        )
        _log_gemini_usage("get_ai_response", response)
        return response.text or "Sorry, I couldn't come up with a reply just now."
    except genai_errors.APIError as exc:
        logger.error("Gemini API call failed for contact %s: %s", contact.id, exc)
        return "Sorry, something went wrong generating a reply. Please try again shortly."


# Appended to the same system instruction get_ai_response uses (BOT_PERSONA +
# business/knowledge-base context) only for classify_and_respond's call — the
# structured-output half of the natural-language flow-intent feature. Kept as
# an addendum rather than folded into BOT_PERSONA itself so get_ai_response's
# plain-text eval-tool call stays exactly as it was before this feature.
_CLASSIFICATION_INSTRUCTION = (
    "\n\nOn top of replying conversationally as above, classify this message for "
    "internal routing — fill every field, every time:\n\n"
    "action:\n"
    "- \"launch_flow_candidate\": the person is expressing real, CURRENT intent to "
    "start one of our services right now, but only in general terms — naming a "
    "category, not a specific item within it — e.g. 'I want to file my ITR', 'help "
    "me register for GST', 'I got a GST notice, need help responding'. Set "
    "flow_type to the single best match.\n"
    "- \"direct_service_request\": the person names a SPECIFIC item within a "
    "category, not just the category — e.g. 'I need TDS quarterly filing', 'sign "
    "me up for the monthly TDS payment and the 26QB filing', 'register my private "
    "limited company'. Set flow_type the same way as launch_flow_candidate, and "
    "set direct_service_query to their own words for the specific item(s) (don't "
    "map it to our exact catalogue name yourself — just pass along what they "
    "said, same rule as cart_edit_query below). If they only named a category "
    "with no specific item, use launch_flow_candidate instead, not this.\n"
    "  For ITR_TYPE specifically, this also covers describing WHO they are "
    "clearly enough to name one specific filing type, even without saying "
    "its name — this is reading a fact they stated, not guessing: someone "
    "employed by a company/organization (a job, a salary, an employer named) "
    "means ITR Salary Filing; freelancing/gig work/no fixed employer means "
    "Freelancer ITR; running their own business/proprietorship (not a "
    "company or LLP) means ITR Individual Business Filing; an NRI means ITR "
    "NRI Filing; an LLP means ITR LLP Filing; trading Futures & Options "
    "means Futures & Options ITR; a delivery/ride-share partner (Swiggy, "
    "Zomato, Uber, Ola, etc.) means ITR Filing - Delivery/Ride Partners. In "
    "these cases, phrase direct_service_query using OUR product wording for "
    "the matched one (e.g. 'salary income filing', 'freelancer filing', "
    "'own business filing') rather than echoing their exact words verbatim, "
    "so it actually names the product for matching — this is still not "
    "inventing anything, just naming which of our real, existing products "
    "the fact they stated maps to. Someone who owns a small business AND "
    "separately mentions employment income may need more than one filing —  "
    "classify each clearly-stated fact on its own; don't drop one because "
    "another was also mentioned. Stay with launch_flow_candidate (or "
    "clarify_first) whenever their situation doesn't clearly land on one of "
    "these — e.g. 'I have some business income' alone, with no sense of "
    "whether it's a proprietorship, freelance work, or an LLP, is still "
    "ambiguous and must not be guessed.\n"
    "- \"clarify_first\": you can't tell which service they mean, or real intent is "
    "there but a key detail is missing/ambiguous. Your `reply` must be the short "
    "clarifying question itself — don't set launch_flow_candidate yet even if "
    "you're fairly sure what they'll say next; wait for their next message instead.\n"
    "- \"edit_cart\": ONLY when the system context above told you this contact "
    "already has a cart, and they're now asking, in plain text, to add or remove "
    "something from it — e.g. 'remove the capital gains one', 'can you take out "
    "GST filing', 'add house property too'. Never set this if the context didn't "
    "list a cart for them — treat the same words as continue_chat/clarify_first "
    "instead. Set cart_edit_operation to \"add\" or \"remove\" and cart_edit_query "
    "to their own words for the item (don't try to map it to our exact catalogue "
    "name yourself — just pass along what they said, e.g. 'capital gains', 'house "
    "property'). Your `reply` for this action is never shown to the user — the "
    "cart update itself replies deterministically — so keep it minimal.\n"
    "- \"request_handoff\": an explicit ask to talk to a real person/human/agent/"
    "CA — e.g. 'can I speak to someone', 'this isn't working, connect me to a "
    "human', 'I need to talk to your team'. Beyond what the HELP keyword already "
    "covers — this is for the same intent expressed in the person's own words. "
    "Never infer this from mere frustration or confusion alone; it has to be an "
    "actual ask for a person.\n"
    "- \"check_order_status\": asking about an existing order, payment, or filing "
    "of theirs — e.g. 'what's happening with my order', 'did my payment go "
    "through', 'any update on my filing'. Not a new pricing question about a "
    "service they don't have yet — that's continue_chat.\n"
    "- \"continue_chat\": everything else — informational questions ('what's the "
    "ITR deadline?', 'how much does GST registration cost?', 'do I need to file if "
    "I'm freelancing?'), small talk, out-of-scope asks, or anything that isn't "
    "itself a request to start a service right now. Answer normally per your "
    "persona and scope rules above.\n\n"
    "reply: always written in your normal persona/voice, as if replying directly "
    "to the person, regardless of action. For launch_flow_candidate specifically, "
    "keep it a short, natural acknowledgement that you're about to offer to start "
    "that service (e.g. 'Got it — let's get your GST registration sorted.') — "
    "don't ask a yes/no question yourself, a Confirm/Edit prompt is shown "
    "automatically right after your reply.\n\n"
    "CRITICAL: a purely informational question — asking about price, eligibility, "
    "deadlines, or how something works — is always continue_chat, never "
    "launch_flow_candidate, even if it's about a service we offer. Only a genuine, "
    "current request to start counts. When unsure whether something is "
    "informational or a real request to start, prefer continue_chat over guessing "
    "launch_flow_candidate.\n\n"
    "flow_type: exactly one of the six below — match whichever one the "
    "person's own words literally name, in ANY language or script (plain "
    "English, Hindi, or Hinglish/Romanized Hindi mixed with English, as is "
    "completely normal for this business's customers). Never default to "
    "ITR_TYPE just because it's the most common service — if the message "
    "names GST, answer GST_FILING; if it names TDS, answer TDS_FILING; only "
    "pick ITR_TYPE when the message actually says ITR/income tax/return, "
    "not merely because tax filing in general was mentioned.\n"
    "  - ITR_TYPE — income tax return filing. e.g. 'I want to file my ITR', "
    "'itr filing karna hai', 'income tax return kaise file karu', 'apna "
    "return file karna hai'.\n"
    "  - GST_FILING — GST registration or return filing. e.g. 'help me "
    "register for GST', 'gst ka filling chaiye', 'gst return file karna "
    "hai', 'gst registration karwana hai'.\n"
    "  - TDS_FILING — TDS filing. e.g. 'I need TDS filing', 'tds file "
    "karna hai', 'tds return kaise submit karu'.\n"
    "  - BUSINESS_REG — business/company registration. e.g. 'register my "
    "private limited company', 'company register karwani hai', 'naya "
    "business start karna hai, registration chahiye'.\n"
    "  - NOTICES — responding to a tax/GST notice they've already "
    "identified. e.g. 'I got a GST notice, need help responding', 'mujhe "
    "notice aaya hai gst ka'. If they mention a notice but don't yet know "
    "what it's about or which tax/authority it's from (e.g. 'I received a "
    "notice, not sure what it's about') — that's not enough to launch: "
    "it might not even be a tax notice at all, so use clarify_first (ask "
    "which one — ITR, GST, or TDS) instead of assuming NOTICES.\n"
    "  - BOOK_EXPERT — wants to talk to a consultant/expert directly, not "
    "a specific filing. e.g. 'can I talk to a CA', 'expert se baat karni "
    "hai'.\n"
    "These keyword examples are for matching WHICH flow_type once you've "
    "already decided the action is launch_flow_candidate/"
    "direct_service_request per the action rules above — they don't lower "
    "the bar for that decision, and most category names are NOT ambiguous "
    "this way: 'I want to register my new company' or 'company register "
    "karwani hai' is a clear, current BUSINESS_REG request on its own — "
    "don't ask what kind of company first, that's a detail the Flow itself "
    "collects. GST specifically is the one genuinely overloaded case: a "
    "bare 'help me with GST' or 'kuch GST ka kaam hai', with no verb "
    "indicating registration vs. return filing vs. a notice, really is "
    "ambiguous between GST_FILING and NOTICES and stays clarify_first — "
    "but 'gst ka filling chaiye' or 'gst return file karna hai' already "
    "name filing specifically, so they launch GST_FILING same as any other "
    "clear request.\n\n"
    "Only meaningful when action is launch_flow_candidate or "
    "direct_service_request; set it to \"NONE\" otherwise.\n\n"
    "direct_service_query: only meaningful when action is direct_service_request; "
    "set it to \"\" otherwise.\n\n"
    "cart_edit_operation: \"add\" or \"remove\", only meaningful when action is "
    "edit_cart; set it to \"NONE\" otherwise.\n\n"
    "cart_edit_query: only meaningful when action is edit_cart; set it to \"\" "
    "otherwise.\n\n"
    "confidence: \"high\" only for an unambiguous, current request matching "
    "flow_type exactly; \"medium\" for a clear service area with some ambiguity in "
    "the details; \"low\" if you're mostly guessing. Every action gets a "
    "confidence value, not just launch_flow_candidate."
)

# Shared between the response schema below and _parse_classification_response's
# validation, so the two can never silently drift apart.
_CLASSIFICATION_ACTIONS = (
    "launch_flow_candidate", "direct_service_request", "continue_chat", "clarify_first",
    "edit_cart", "request_handoff", "check_order_status",
)

_CLASSIFICATION_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "reply": {"type": "STRING"},
        "action": {
            "type": "STRING",
            "enum": list(_CLASSIFICATION_ACTIONS),
        },
        "flow_type": {
            "type": "STRING",
            "enum": [*FLOW_TYPE_CHOICES, "NONE"],
        },
        "direct_service_query": {"type": "STRING"},
        "cart_edit_operation": {
            "type": "STRING",
            "enum": ["add", "remove", "NONE"],
        },
        "cart_edit_query": {"type": "STRING"},
        "confidence": {"type": "STRING", "enum": ["high", "medium", "low"]},
    },
    "required": [
        "reply", "action", "flow_type", "direct_service_query",
        "cart_edit_operation", "cart_edit_query", "confidence",
    ],
}

_CLASSIFICATION_FALLBACK_ACTION = "continue_chat"


def _parse_classification_response(raw_text, fallback_reply):
    """Best-effort parse of a classify_and_respond call's JSON output — falls
    back to a safe continue_chat/low-confidence result (never launches
    anything, never loses the reply) if response_schema was somehow violated
    or the call produced unparsable output, rather than raising and dropping
    the customer's turn entirely."""
    try:
        data = json.loads(raw_text)
        reply = data.get("reply") or fallback_reply
        action = data.get("action")
        if action not in _CLASSIFICATION_ACTIONS:
            action = _CLASSIFICATION_FALLBACK_ACTION
        flow_type = data.get("flow_type")
        if flow_type not in FLOW_TYPE_CHOICES:
            flow_type = None
        direct_service_query = data.get("direct_service_query") or ""
        cart_edit_operation = data.get("cart_edit_operation")
        if cart_edit_operation not in ("add", "remove"):
            cart_edit_operation = None
        cart_edit_query = data.get("cart_edit_query") or ""
        confidence = data.get("confidence")
        if confidence not in ("high", "medium", "low"):
            confidence = "low"
        if action not in ("launch_flow_candidate", "direct_service_request"):
            # A launch/direct request is only ever driven by this exact
            # combination — belt and braces against a schema-violating
            # response that sets flow_type without the matching action.
            flow_type = None
        if action != "direct_service_request":
            direct_service_query = ""
        if action != "edit_cart":
            # Same belt-and-braces for the edit_cart pair.
            cart_edit_operation = None
            cart_edit_query = ""
        return {
            "reply": reply, "action": action, "flow_type": flow_type,
            "direct_service_query": direct_service_query,
            "cart_edit_operation": cart_edit_operation, "cart_edit_query": cart_edit_query,
            "confidence": confidence,
        }
    except (ValueError, TypeError, AttributeError):
        return {
            "reply": fallback_reply, "action": _CLASSIFICATION_FALLBACK_ACTION,
            "flow_type": None, "direct_service_query": "",
            "cart_edit_operation": None, "cart_edit_query": "",
            "confidence": "low",
        }


def classify_and_respond(contact, text, history=None):
    """
    Single Gemini call that both answers conversationally (same persona/scope
    rules as get_ai_response, via the same _build_system_instruction) AND
    classifies whether this message is a live request to start one of our
    services — the natural-language counterpart to tapping a menu/Flow
    option, e.g. "I want to file my ITR" with no button involved. This is
    deliberately the SAME call as free-chat, not a second LLM round trip —
    response_mime_type/response_schema below just constrain the one call's
    output shape; see tasks.process_incoming_message for how the result is
    branched on.

    Returns {"reply": str, "action": "launch_flow_candidate" |
    "direct_service_request" | "continue_chat" | "clarify_first" |
    "edit_cart", "flow_type": one of FLOW_TYPE_CHOICES or None,
    "direct_service_query": str, "cart_edit_operation": "add"|"remove"|None,
    "cart_edit_query": str, "confidence": "high"|"medium"|"low"}. Never
    raises — any failure (missing API key, API error, malformed output)
    degrades to a safe continue_chat result so a classification hiccup never
    blocks a reply going out.
    """
    fallback_reply = f"Thanks for your message: '{text}'. (AI response not yet configured.)"
    if not settings.GEMINI_API_KEY:
        return {
            "reply": fallback_reply, "action": _CLASSIFICATION_FALLBACK_ACTION, "flow_type": None,
            "direct_service_query": "", "cart_edit_operation": None, "cart_edit_query": "", "confidence": "low",
        }

    config = genai_types.GenerateContentConfig(
        thinking_config=_THINKING_CONFIG,
        max_output_tokens=600,  # a little above get_ai_response's 500 — same reply, plus the JSON scaffolding
        # Lower than get_ai_response's 0.4 — this call's `reply` text can
        # still afford a little variation, but action/flow_type are a
        # structured classification, not creative writing, and benefit from
        # the more consistent, literal-keyword-following behavior a lower
        # temperature gives (see the flow_type keyword-matching rules in
        # _CLASSIFICATION_INSTRUCTION below — this is what makes them stick).
        temperature=0.15,
        system_instruction=_build_system_instruction(contact, text, history) + _CLASSIFICATION_INSTRUCTION,
        response_mime_type="application/json",
        response_schema=_CLASSIFICATION_RESPONSE_SCHEMA,
    )

    try:
        response = _get_client().models.generate_content(
            model=GEMINI_MODEL, contents=_build_gemini_contents(text, history), config=config
        )
        _log_gemini_usage("classify_and_respond", response)
        return _parse_classification_response(
            response.text or "", "Sorry, I couldn't come up with a reply just now."
        )
    except genai_errors.APIError as exc:
        logger.error("Gemini classification call failed for contact %s: %s", contact.id, exc)
        return {
            "reply": "Sorry, something went wrong generating a reply. Please try again shortly.",
            "action": _CLASSIFICATION_FALLBACK_ACTION, "flow_type": None,
            "direct_service_query": "", "cart_edit_operation": None, "cart_edit_query": "", "confidence": "low",
        }


def _parse_summary_response(raw):
    """Splits a 'SUMMARY: ...\nNEXT: ...' formatted response apart. Falls back
    to treating the whole reply as the summary if Gemini didn't follow the
    format exactly, rather than losing the response outright."""
    summary_match = re.search(r"SUMMARY:\s*(.+?)(?=\nNEXT:|\Z)", raw, re.DOTALL)
    next_match = re.search(r"NEXT:\s*(.+)", raw, re.DOTALL)
    summary = summary_match.group(1).strip() if summary_match else raw.strip()
    next_action = next_match.group(1).strip() if next_match else ""
    return summary, next_action


def summarize_conversation(contact, messages):
    """
    Folds a batch of messages (oldest first) into an updated running summary
    (long-term memory) AND a short prediction of what the contact is likely
    to do/ask next — both from a single Gemini call. The next-step prediction
    deliberately rides along on this existing periodic call (every
    SUMMARY_EVERY_N_MESSAGES) instead of getting its own call per message —
    that would double the Gemini cost/latency of every single reply for a
    prediction that doesn't need to be that fresh.

    Returns (summary, predicted_next_action).
    """
    if not settings.GEMINI_API_KEY or not messages:
        return contact.summary, contact.predicted_next_action

    transcript = "\n".join(
        f"{'User' if m.direction == 'in' else 'Bot'}: {m.body}" for m in messages
    )
    prompt = (
        "You're maintaining CRM notes on a WhatsApp lead for a tax advisory "
        "business. Based on the existing summary and the new messages below, "
        "produce two things:\n\n"
        "1. An updated long-term summary — merge the existing summary with "
        "what's new into a few concise sentences (key facts, preferences, and "
        "open topics only, no play-by-play of the chat).\n"
        "2. A short prediction (one sentence, max ~15 words) of what this "
        "contact is likely to do or ask next, based on the actual pattern of "
        "the conversation so far — e.g. 'Likely to ask about GST pricing "
        "next', 'Likely to book a consultation soon', 'May go quiet without a "
        "follow-up nudge'. Base this only on the pattern shown, don't guess "
        "wildly, and don't invent intent the messages don't support.\n\n"
        f"Existing summary:\n{contact.summary or '(none yet)'}\n\n"
        f"New messages:\n{transcript}\n\n"
        "Respond in exactly this format:\n"
        "SUMMARY: <updated summary>\n"
        "NEXT: <next-step prediction>"
    )

    try:
        response = _get_client().models.generate_content(
            model=GEMINI_MODEL,
            contents=[genai_types.Content(role="user", parts=[genai_types.Part(text=prompt)])],
            config=genai_types.GenerateContentConfig(
                thinking_config=_THINKING_CONFIG, max_output_tokens=350
            ),
        )
        _log_gemini_usage("summarize_conversation", response)
        raw = (response.text or "").strip()
        if not raw:
            return contact.summary, contact.predicted_next_action
        summary, next_action = _parse_summary_response(raw)
        return summary or contact.summary, next_action or contact.predicted_next_action
    except genai_errors.APIError as exc:
        logger.error("Gemini summary update failed for contact %s: %s", contact.id, exc)
        return contact.summary, contact.predicted_next_action


# Typing indicators need this endpoint version or newer; bumped from v20.0 for that.
WHATSAPP_API_VERSION = "v22.0"


def _whatsapp_url():
    return f"https://graph.facebook.com/{WHATSAPP_API_VERSION}/{settings.WHATSAPP_PHONE_ID}/messages"


def send_typing_indicator(wa_message_id):
    """
    Marks the incoming message as read and shows the "typing..." bubble on
    WhatsApp. Meta clears it automatically once we send the actual reply (or
    after ~25s, whichever comes first) — nothing to turn off explicitly.
    Best-effort: a failure here should never block the actual reply.
    """
    headers = {
        "Authorization": f"Bearer {settings.WHATSAPP_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "status": "read",
        "message_id": wa_message_id,
        "typing_indicator": {"type": "text"},
    }
    try:
        response = requests.post(_whatsapp_url(), json=payload, headers=headers, timeout=5)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning(
            "Failed to show typing indicator for %s: %s | %s",
            wa_message_id, exc, _whatsapp_error_detail(exc),
        )


def _whatsapp_error_detail(exc):
    """Pulls Meta's actual {"error": {"code", "message", ...}} body out of a
    failed request, e.g. (#131030) Recipient phone number not in allowed
    list — raise_for_status()'s own str(exc) only carries the HTTP status
    line ("400 Client Error: Bad Request"), which is useless for telling a
    test-number allow-list rejection apart from a malformed payload, an
    expired token, or a closed 24h session without opening the Meta
    dashboard. Best-effort: falls back to raw response text, then to
    "(no response body)" if the response itself is missing (e.g. a
    connection error rather than an HTTP error)."""
    response = getattr(exc, "response", None)
    if response is None:
        return "(no response body)"
    try:
        return response.json().get("error", response.text)
    except ValueError:
        return response.text or "(no response body)"


def _post_whatsapp_payload(payload, action_label):
    """Shared POST + error handling for every outbound WhatsApp API call —
    `action_label` is just for the log line if it fails."""
    headers = {
        "Authorization": f"Bearer {settings.WHATSAPP_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    try:
        response = requests.post(_whatsapp_url(), json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        logger.error(
            "Failed to %s (to %s): %s | %s",
            action_label, payload.get("to"), exc, _whatsapp_error_detail(exc),
        )
        return None


def send_whatsapp_message(to_number, message):
    """Send a plain text message back via the Meta WhatsApp Cloud API."""
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "text": {"body": message},
    }
    return _post_whatsapp_payload(payload, "send WhatsApp message")


def send_whatsapp_template(to_number, template_name, language_code, body_params):
    """
    Sends an approved WhatsApp Message Template — the only send type Meta
    allows outside the 24h customer-service window (a plain send_whatsapp_message
    there fails outright with error 131047). `body_params` is a list of
    (parameter_name, value) pairs filling the template body's {{parameter_name}}
    variables — Meta templates created since the named-variable rollout reject
    the older bare positional {{1}} style, so every parameter here carries its
    name explicitly rather than relying on list order. Pass [] for a template
    with no variables.

    Used today by tasks.send_abandoned_cart_nudges as the fallback once a
    free-form nudge send fails — see WHATSAPP_CART_NUDGE_TEMPLATE_NAME.
    """
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "template",
        "template": {
            "name": template_name,
            "language": {"code": language_code},
            "components": (
                [{
                    "type": "body",
                    "parameters": [
                        {"type": "text", "parameter_name": name, "text": value}
                        for name, value in body_params
                    ],
                }]
                if body_params else []
            ),
        },
    }
    return _post_whatsapp_payload(payload, "send WhatsApp template")


# WhatsApp reply buttons: max 3 options, each button title capped at 20 chars
# by the API itself. For more than 3 options use a list message instead (not
# built yet — add send_whatsapp_list the same way if a menu needs it).
WHATSAPP_MAX_BUTTONS = 3


def send_whatsapp_buttons(to_number, body_text, options):
    """
    Sends a WhatsApp interactive "reply buttons" message — up to 3 tappable
    buttons instead of asking the user to type a number. `options` is a list
    of (id, title) pairs; id is whatever comes back in the webhook when
    tapped (see views._extract_message_text) — pick ids that already match
    what the relevant resolve_*() function accepts as typed text, so no
    separate interactive-specific parsing is needed downstream.
    """
    if not options or len(options) > WHATSAPP_MAX_BUTTONS:
        raise ValueError(f"WhatsApp reply buttons support 1-{WHATSAPP_MAX_BUTTONS} options, got {len(options)}")

    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "interactive",
        "interactive": {
            "type": "button",
            "body": {"text": body_text},
            "action": {
                "buttons": [
                    {"type": "reply", "reply": {"id": option_id, "title": title[:20]}}
                    for option_id, title in options
                ]
            },
        },
    }
    return _post_whatsapp_payload(payload, "send WhatsApp buttons")


def send_whatsapp_cta_url(to_number, body_text, button_text, url):
    """
    Sends a WhatsApp interactive "CTA URL" message — a single button that
    opens an external link (a Razorpay payment link) in the phone's
    browser. Distinct from send_whatsapp_buttons: a reply-button tap posts
    its id straight back into our own webhook and gets resolved like typed
    text; this one hands the customer off to Razorpay's hosted checkout
    instead, so there's no reply to resolve here — payment status only ever
    comes back via tasks.process_payment_webhook.
    """
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "interactive",
        "interactive": {
            "type": "cta_url",
            "body": {"text": body_text},
            "action": {
                "name": "cta_url",
                "parameters": {"display_text": button_text[:20], "url": url},
            },
        },
    }
    return _post_whatsapp_payload(payload, "send WhatsApp CTA URL")


def send_whatsapp_list(to_number, body_text, button_label, options):
    """
    Sends a WhatsApp interactive "list" message — a single button that opens
    a scrollable list of up to 10 tappable rows, for menus with more options
    than reply buttons (max 3) can hold. `options` is a list of
    (id, title, description) triples — same id-matches-typed-text pattern as
    send_whatsapp_buttons, so a tapped row needs no separate resolve logic.
    """
    if not options or len(options) > WHATSAPP_MAX_LIST_ROWS:
        raise ValueError(f"WhatsApp list messages support 1-{WHATSAPP_MAX_LIST_ROWS} rows, got {len(options)}")

    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "interactive",
        "interactive": {
            "type": "list",
            "body": {"text": body_text},
            "action": {
                "button": button_label[:20],
                "sections": [
                    {
                        "rows": [
                            {"id": row_id, "title": title, "description": description}
                            for row_id, title, description in options
                        ]
                    }
                ],
            },
        },
    }
    return _post_whatsapp_payload(payload, "send WhatsApp list")


def send_whatsapp_list_sections(to_number, body_text, button_label, sections):
    """
    Sends a WhatsApp interactive "list" message with multiple titled
    sections — used for the flattened top-level menu (see
    build_flattened_menu_sections), where each section groups one category's
    services together under its display name. `sections` is a list of
    (section_title, options) pairs; `options` is the same (id, title,
    description) triple shape as send_whatsapp_list. Rows across every
    section still can't exceed WhatsApp's 10-row cap.
    """
    total_rows = sum(len(options) for _, options in sections)
    if not sections or total_rows > WHATSAPP_MAX_LIST_ROWS:
        raise ValueError(f"WhatsApp list messages support 1-{WHATSAPP_MAX_LIST_ROWS} rows total, got {total_rows}")

    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "interactive",
        "interactive": {
            "type": "list",
            "body": {"text": body_text},
            "action": {
                "button": button_label[:20],
                "sections": [
                    {
                        "title": _truncate(section_title, WHATSAPP_LIST_TITLE_LIMIT),
                        "rows": [
                            {"id": row_id, "title": title, "description": description}
                            for row_id, title, description in options
                        ],
                    }
                    for section_title, options in sections
                ],
            },
        },
    }
    return _post_whatsapp_payload(payload, "send WhatsApp list (sectioned)")


def send_whatsapp_flow(to_number, body_text, flow_cta, flow_action_payload, screen_id=ADDONS_FLOW_SCREEN_ID):
    """
    Sends a WhatsApp interactive Flow message — the ITR Flow's ITR_TYPE or
    ADDONS screen (see build_itr_type_flow_payload /
    build_addons_screen_flow_payload) that the customer fills in and submits
    once, instead of a tap-and-reply round trip per step. Requires a Flow
    asset already created and published in Meta Business Manager, with its
    id set as settings.WHATSAPP_ADDON_FLOW_ID (see
    docs/whatsapp_flows/README.md) — callers only take this path once
    services.addon_flow_enabled() is true.
    """
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "interactive",
        "interactive": {
            "type": "flow",
            "body": {"text": body_text},
            "action": {
                "name": "flow",
                "parameters": {
                    "flow_message_version": "3",
                    "flow_id": settings.WHATSAPP_ADDON_FLOW_ID,
                    "flow_cta": flow_cta[:20],
                    "flow_action": "navigate",
                    "flow_action_payload": {
                        "screen": screen_id,
                        "data": flow_action_payload,
                    },
                },
            },
        },
    }
    return _post_whatsapp_payload(payload, "send WhatsApp flow")


# ---------------------------------------------------------------------------
# The "mega" WhatsApp Flow: service category through add-ons/quantities to a
# final review, all as server-driven Flow screens (see flow_screens.py and
# views.whatsapp_flow_endpoint). Every option list here is pulled live from
# ServiceCategory/ServicePricing — the same tables the chat wizard above
# reads — so pricing/catalogue never has to be maintained in two places.
# ---------------------------------------------------------------------------

SERVICE_FLOW_TOKEN_SALT = "whatsapp_bot.service_flow_token"

# ITR sub-types that collect add-ons before a final price (see
# itr_wizard_parents in the chat-wizard section above — kept as a separate,
# smaller constant here since the mega Flow only needs the codes, not the
# querying logic tied to WHATSAPP_ADDON_FLOW_ID).
ITR_ADDON_PARENT_CODES = set(_WIZARD_ADDON_SLOT_BY_TYPE)


def make_flow_token(contact, target_screen=None):
    """Signs the contact id into an opaque token handed to WhatsApp when the
    Flow is opened, and returned unchanged on every subsequent data_exchange
    request — this is how flow_screens.py knows which Contact's cart to
    read/update, since the Flow endpoint has no session of its own. Signed
    (not just str(contact.id)) so a malformed or tampered token is rejected
    outright rather than silently mutating the wrong contact's cart.

    `target_screen`, if given, is a deep-link destination (one of
    FLOW_TYPE_CHOICES) read back by flow_screens.handle_flow_action's INIT
    handling via target_screen_from_flow_token — see send_service_flow.
    Embedded in the signed token rather than relied on from the outbound
    message's own flow_action_payload because Meta's data-backed Flow always
    calls our INIT endpoint on open, and doesn't reliably echo back what
    "screen" was requested at send time; the token is the one thing
    guaranteed to round-trip on that very first request."""
    payload = {"contact_id": contact.id}
    if target_screen:
        payload["target_screen"] = target_screen
    return signing.dumps(payload, salt=SERVICE_FLOW_TOKEN_SALT)


def contact_from_flow_token(token):
    """Inverse of make_flow_token. Returns None for a missing/invalid/expired
    token or an id that no longer exists — callers must treat that as a hard
    error (see flow_screens._error_response), never guess a contact."""
    if not token:
        return None
    try:
        data = signing.loads(token, salt=SERVICE_FLOW_TOKEN_SALT, max_age=60 * 60 * 24 * 3)
    except signing.BadSignature:
        return None
    return Contact.objects.filter(id=data.get("contact_id")).first()


def target_screen_from_flow_token(token):
    """Inverse of make_flow_token's target_screen — the deep-link destination
    embedded when the Flow was opened, or None for a normal open / an
    invalid/expired token. Used only by flow_screens._entry_screen_response
    on INIT."""
    if not token:
        return None
    try:
        data = signing.loads(token, salt=SERVICE_FLOW_TOKEN_SALT, max_age=60 * 60 * 24 * 3)
    except signing.BadSignature:
        return None
    return data.get("target_screen")


def build_service_flow_trigger_body(contact=None):
    return f"{_greeting_prefix(contact)} Let's get you sorted — tap below to see what we can help with."


def send_service_flow(contact, deep_link_screen=None, body_text=None):
    """Opens the mega Flow for a contact — always straight to SERVICE, the
    Flow's entry screen, since there is no LANGUAGE screen in it (language
    is collected once via chat, before the Flow is ever opened — see
    tasks.py's unconditional language gate in process_incoming_message; by
    the time this runs, contact.language is always already set). No further
    flow_action_payload data needed since every screen's data comes from
    the server on open (data_api_version + an endpoint means WhatsApp
    always calls INIT itself, unlike the fully-static add-on Flow above).

    `deep_link_screen`, if given (one of FLOW_TYPE_CHOICES), skips straight
    past SERVICE to that routing_model screen — used by the natural-language
    intent classifier (tasks.py's pending_flow_type_candidate confirm flow)
    once the customer's confirmed which service they want, so "Yes, start"
    is one open, not a re-selection tap. See make_flow_token/
    flow_screens._entry_screen_response for how the target actually reaches
    the Flow's INIT handler.

    While WHATSAPP_SERVICE_FLOW_ID still points at a draft (unpublished)
    Flow, Meta requires "mode": "draft" here or every send fails with error
    131009 — see WHATSAPP_SERVICE_FLOW_MODE in settings.py.

    IMPORTANT, confirmed against the real Graph API (not documented
    anywhere obvious): Meta rejects flow_action_payload.screen outright
    (error 131009, "Specified screen X is not allowed as first screen of
    this flow") unless it's EXACTLY the Flow's literal registered root
    screen — "SERVICE" here, per service_flow.json's routing_model — no
    matter what should actually render for this contact. This is true even
    for a deep-linked reopen, not just a plain one — so the payload's screen
    is hardcoded to "SERVICE" below, always. The real destination (SERVICE
    itself, or a deep-linked screen) is decided entirely server-side, once
    Meta calls our INIT handler (flow_screens._entry_screen_response), from
    the flow_token's embedded target_screen — the payload's screen is just a
    formality Meta validates against the Flow's declared entry point, not
    what's actually shown.

    The button's own label (flow_cta) DOES reflect the real destination
    though, unlike the payload's screen — a deep-linked launch (natural-
    language intent confirmed) shows FLOW_TYPE_CTA_LABELS' name for that
    flow_type (e.g. "Start GST Filing"), not the generic "Get started" every
    plain reopen still shows. This is the one user-visible signal that lets
    someone catch a misclassified intent before the Flow actually opens —
    see tasks.py's pending_flow_type_candidate confirm step, which shows
    this same label's plain-language form in its own confirm message first.

    The CTA label is gated on contact.language too, purely as a cosmetic
    safety net (still "Get started" when unset) — the deep link itself is
    never dropped for a language-less contact regardless (target_screen is
    always embedded in the token when given, and _entry_screen_response
    honors it either way), this only avoids promising "Start GST Filing" on
    the off chance send_service_flow is ever reached before tasks.py's
    language gate has run.

    `body_text`, if given, replaces the generic build_service_flow_trigger_body
    greeting as this message's own body — used by the natural-language
    auto-launch path (agents.sales_cart_node) to show Gemini's own
    conversational acknowledgment (e.g. "Got it! We can definitely help with
    your GST filing for your business.") right on the Flow-opening message
    itself, since there's no separate Confirm/Edit message anymore for that
    text to live on."""
    is_deep_link = bool(deep_link_screen and contact.language)
    flow_cta = FLOW_TYPE_CTA_LABELS.get(deep_link_screen or "", "Get started") if is_deep_link else "Get started"
    parameters = {
        "flow_message_version": "3",
        "flow_id": settings.WHATSAPP_SERVICE_FLOW_ID,
        "flow_cta": flow_cta[:20],
        "flow_token": make_flow_token(contact, target_screen=deep_link_screen),
        "flow_action": "navigate",
        "flow_action_payload": {"screen": "SERVICE"},
    }
    if settings.WHATSAPP_SERVICE_FLOW_MODE == "draft":
        parameters["mode"] = "draft"
    payload = {
        "messaging_product": "whatsapp",
        "to": contact.phone_number,
        "type": "interactive",
        "interactive": {
            "type": "flow",
            "body": {"text": body_text or build_service_flow_trigger_body(contact)},
            "action": {"name": "flow", "parameters": parameters},
        },
    }
    return _post_whatsapp_payload(payload, "send WhatsApp service flow")


# How much buying-intent signal reaching each mega-Flow screen represents —
# see compute_lead_score and record_flow_progress. Deliberately not just "did
# they reach the end": a contact who configured ITR_ADDONS and then went
# quiet is a warmer lead than one who tapped SERVICE once and left, even
# though neither technically "completed" anything. Tune these once real
# outcomes (which of these actually convert) are available to check them
# against — same philosophy as compute_lead_score's other weights.
#
# ERROR is deliberately absent (not even a 0 entry): reaching it is a
# failure, not progress. record_flow_progress treats an unlisted screen as
# "ignore, don't touch flow_furthest_screen" — so hitting ERROR neither
# scores anything nor overwrites whatever real progress came before it.
SCREEN_INTENT_WEIGHTS = {
    "SERVICE": 8,        # saw the category list — the Flow's entry screen,
                          # now that there's no LANGUAGE screen ahead of it
                          # (a contact with flow_furthest_screen == "LANGUAGE"
                          # from before this change just scores 0 here — see
                          # compute_lead_score's .get(..., 0) fallback)
    "ITR_TYPE": 18,      # picked a specific ITR filing type
    "GST_FILING": 18,    # picked specific GST filings
    "TDS_FILING": 18,    # picked specific TDS filings
    "BUSINESS_REG": 18,  # picked specific registration services
    "NOTICES": 18,       # picked a specific notice case
    "BOOK_EXPERT": 18,   # picked a specific consultation topic
    "ITR_ADDONS": 25,    # configuring specifics beyond the base filing
    "REVIEW": 35,        # got a real, priced quote — the strongest signal
    "COMING_SOON": 12,   # wanted a service that isn't offered yet — real
                          # interest, just nothing to quote
}


def record_flow_progress(contact, screen):
    """Called once per Flow request/response (see flow_screens.handle_flow_action)
    to track the deepest screen a contact has ever reached — monotonic, never
    regresses (going back to SERVICE and picking a different category
    shouldn't erase credit for having reached ITR_ADDONS earlier). Unknown
    screen ids (defensive — the routing model should prevent this) are
    ignored rather than raising, since this runs inside the Flow's
    synchronous response path and a scoring hiccup must never break the
    actual screen transition.

    Also refreshes lead_score/lead_tier right here rather than waiting for
    update_lead_score's usual call site (process_incoming_message, on the
    next plain chat message) — a contact who opens the Flow, gets deep into
    it, then abandons without ever sending another chat message would
    otherwise show a stale score in /admin/ forever. Both are cheap DB
    writes, no LLM call, so doing it inline here doesn't meaningfully risk
    the Flow endpoint's response-time budget."""
    new_weight = SCREEN_INTENT_WEIGHTS.get(screen)
    if new_weight is None:
        return
    current_weight = SCREEN_INTENT_WEIGHTS.get(contact.flow_furthest_screen, -1)
    if new_weight > current_weight:
        contact.flow_furthest_screen = screen
        contact.save(update_fields=["flow_furthest_screen"])
        update_lead_score(contact)


def service_flow_enabled():
    return bool(settings.WHATSAPP_SERVICE_FLOW_ID)


def flow_category_options():
    """(id, title, description) for the SERVICE screen — id is the
    category_code, so the very next request tells the endpoint exactly which
    category was picked with no extra lookup."""
    return [
        (cat.category_code, _truncate(cat.display_name, WHATSAPP_LIST_TITLE_LIMIT), "")
        for cat in ServiceCategory.objects.all()
    ]


def flow_itr_type_options():
    """(id, title, description) for the ITR_TYPE screen — every active,
    top-level ITR filing type (Salary/Business/NRI plus the flat-priced ones
    like F&O/Freelancer/LLP/Company/Trust)."""
    itr_category = ServiceCategory.objects.filter(category_code="itr").first()
    if not itr_category:
        return []
    return [
        (s.service_code, _truncate(s.display_name, WHATSAPP_LIST_TITLE_LIMIT),
         _truncate(_price_part_for(s), WHATSAPP_LIST_DESCRIPTION_LIMIT))
        for s in _active_services(itr_category, parent_service=None)
    ]


def flow_flat_category_options(category_code):
    """(id, title, description) for a category whose entire catalogue is
    flat, top-level, and multi-select — GST_FILING, TDS_FILING, BUSINESS_REG,
    NOTICES, and BOOK_EXPERT all use this same shape (see flow_gst_options
    and friends below), unlike ITR's parent-then-addons tree."""
    category = ServiceCategory.objects.filter(category_code=category_code).first()
    if not category:
        return []
    return [
        (s.service_code, _truncate(s.display_name, WHATSAPP_LIST_TITLE_LIMIT),
         _truncate(f"{_price_part_for(s)} — {s.short_description}", WHATSAPP_LIST_DESCRIPTION_LIMIT))
        for s in _active_services(category, parent_service=None)
    ]


def flow_gst_options():
    """Every active GST filing type, shown as one flat multi-select list — a
    customer can need more than one, e.g. a monthly package plus the annual
    return."""
    return flow_flat_category_options("gst")


def flow_tds_filing_options():
    """Every active TDS Filing service (monthly payment, quarterly filing,
    revised quarterly filing, 26QB), shown as one flat multi-select list."""
    return flow_flat_category_options("tds_filing")


def flow_business_reg_options():
    """Every active Business Registration service — PAN/TAN/company setup/
    trust formation etc., shown as one flat multi-select list."""
    return flow_flat_category_options("business_reg")


def flow_notice_options():
    """Every active Notices/appeal service (ITR/GST/TDS appeal, regular
    assessment, tribunal), shown as one flat multi-select list."""
    return flow_flat_category_options("notices")


def flow_book_expert_options():
    """Every active 30-minute consult topic, shown as one flat multi-select
    list — a customer can book more than one topic in the same session."""
    return flow_flat_category_options("book_expert")


def itr_quantity_children(parent_service):
    """{"house_property": ServicePricing|None, "partnership": ServicePricing|None}
    for one ITR wizard parent — codes are itr_house_property,
    itr_business_house_property, itr_nri_house_property (and the -partnership
    equivalents), so matching by suffix covers all three types without
    hardcoding each one."""
    result = {"house_property": None, "partnership": None}
    for s in optional_children(parent_service):
        if s.question_type != ServicePricing.QUESTION_QUANTITY:
            continue
        for suffix in result:
            if s.service_code.endswith(suffix):
                result[suffix] = s
    return result


def flow_itr_addon_screen_data(parent_service, contact):
    """Data for the ITR_ADDONS screen: the flat-priced optional children as
    checkbox rows (reusing the exact same rows the tap-list/static Flow
    already show — see _addon_option_rows), plus which quantity fields
    (House Property/Partnership) this parent actually has. Not prefilled
    with an already-chosen quantity on re-entry — WhatsApp's TextInput
    component doesn't support an init-value (Meta's Flow JSON validator
    rejects it outright), so a re-visit always starts these two blank,
    same known limitation as the static add-on Flow's checkboxes (see
    docs/whatsapp_flows/README.md)."""
    quantity_children = itr_quantity_children(parent_service)
    return {
        "itr_type": parent_service.service_code,
        "itr_type_label": parent_service.display_name,
        "addon_options": _addon_option_rows(parent_service, contact),
        "show_house_property": quantity_children["house_property"] is not None,
        "show_partnership": quantity_children["partnership"] is not None,
    }


def all_service_codes_in_category(category_code):
    """Every service_code (top-level and children) under one category — used
    as the replace `scope_codes` in replace_cart_scope so switching ITR type
    (or resubmitting the GST list) can't leave a stale selection from a
    different type/an earlier submit sitting in the cart."""
    category = ServiceCategory.objects.filter(category_code=category_code).first()
    if not category:
        return set()
    return set(category.services.values_list("service_code", flat=True))


def replace_cart_scope(contact, scope_codes, selected_codes, quantities=None):
    """Replaces whatever's currently in the cart within `scope_codes` (e.g.
    "every service_code under this ITR parent") with exactly the newly
    submitted selection — the Flow's CheckboxGroup always reports the full
    current tick-state, not a diff, so a straight replace-within-scope (never
    touching selections from an unrelated category/parent) is the correct
    merge, not an append."""
    quantities = quantities or {}
    kept = [item for item in (contact.selected_services or []) if item.get("service_code") not in scope_codes]
    selected_codes = set(selected_codes)
    for code in scope_codes:
        if code in selected_codes or quantities.get(code):
            kept.append({"service_code": code, "quantity": quantities.get(code) or 1})
    contact.selected_services = kept
    contact.save(update_fields=["selected_services"])


def _cart_total(contact):
    """Sum of _selection_price across every currently-selected service —
    factored out of build_flow_review_data so compute_lead_score can weight
    a lead by real cart value without re-deriving it from total_display's
    formatted string."""
    total = Decimal("0")
    for service, quantity in _selected_service_rows(contact):
        amount, _label = _selection_price(service, quantity)
        total += amount
    return total


def build_flow_review_data(contact):
    """{"summary_text", "total_display"} for the REVIEW screen — reuses the
    exact same per-item pricing (_selection_price) as the chat wizard's
    build_selection_complete_reply, so the two can never disagree on a
    total."""
    rows = _selected_service_rows(contact)
    if not rows:
        return {"summary_text": "Nothing selected yet.", "total_display": "₹0"}

    total = Decimal("0")
    lines = []
    for service, quantity in rows:
        amount, label = _selection_price(service, quantity)
        total += amount
        suffix = f" ({quantity})" if service.question_type == ServicePricing.QUESTION_QUANTITY else ""
        lines.append(f"• {service.display_name}{suffix} — {label}")
    return {"summary_text": "\n".join(lines), "total_display": f"₹{total:,.0f}"}


def parse_service_flow_reply(text):
    """Parses a completed mega-Flow submission (the raw nfm_reply.response_json
    string, same convention as parse_itr_flow_reply) into
    {"kind": "review"|"coming_soon"|"error", ...}. Returns None for anything
    that isn't one, so a normal typed message or an unrelated Flow's reply
    both fall through untouched — checked independent of the shape
    parse_itr_flow_reply looks for ("itr_type"), and there's no key overlap
    between the two, so trying both on the same text is safe."""
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    if "summary_text" in data and "total_display" in data:
        return {"kind": "review", "summary_text": data["summary_text"], "total_display": data["total_display"]}
    if "category_name" in data:
        return {"kind": "coming_soon", "category_name": data["category_name"]}
    if "message" in data:
        return {"kind": "error", "message": data["message"]}
    return None


def build_service_flow_reference(contact):
    """A short, human-relayable reference for one completed submission — not
    a database id (those aren't meant for customers), just something a
    customer can quote back on a HELP follow-up so support can find their
    request without asking "which one?". Collisions are fine (this isn't a
    legal record): a returning customer submitting again within the same
    minute is a rare, low-stakes edge case."""
    return f"TP-{contact.id}-{timezone.now().strftime('%m%d-%H%M')}"


def build_service_flow_completion_reply(flow_reply, contact):
    """Deterministic chat reply once the mega Flow's terminal screen lands
    back as a normal inbound message (see parse_service_flow_reply) — the
    cart itself was already saved screen-by-screen in flow_screens.py, so
    this is purely the confirmation message, never a second write.

    The "review" (successful submission) branch includes a real ETA when
    BusinessProfile.hours_note is filled in — never invented, same rule the
    AI prompt already follows for hours. build_service_flow_reference is
    still generated and logged here for internal tracking (so a HELP
    follow-up can be traced to this exact submission) but deliberately kept
    out of the customer-facing text — a reference code reads as unnecessary
    complexity to a customer who never asked for one."""
    kind = flow_reply["kind"]
    if kind == "review":
        reference = build_service_flow_reference(contact)
        logger.info("[service-flow] contact=%s reference=%s total=%s", contact.id, reference, flow_reply["total_display"])
        profile = BusinessProfile.objects.first()
        hours_note = profile.hours_note if profile else ""
        eta = f" ({hours_note})" if hours_note else ""
        return (
            f"All set!\n\n"
            f"Here's what you've selected:\n{flow_reply['summary_text']}\n"
            f"Total: {flow_reply['total_display']}\n\n"
            f"Our team will reach out shortly{eta} to get this moving. "
            "Reply HELP anytime if you'd like us sooner."
        )
    if kind == "coming_soon":
        return (
            f"Thanks for your interest in {flow_reply['category_name']}! We're finalizing this on WhatsApp — "
            "reply HELP and our team will take it from here directly."
        )
    return "Sorry, that didn't go through on our end — could you try that again? Reply HELP if you'd like us to take it from here directly."
