"""
Chat-only guided intake — lets a customer fill in the same service selections
the WhatsApp Flow form collects (flow_screens.py / service_flow.json), but
turn-by-turn in plain chat instead of tapping through the Flow. The Flow
itself (services.FLOW_LAUNCH_LIST_OPTIONS, resolve_flow_launch_choice,
send_service_flow) is completely untouched by this module — it remains the
only button-driven launch path. This is a second, independent way in,
triggered purely by agents.guided_intake_node detecting a specific request in
free text (see services._CLASSIFICATION_ACTIONS's "direct_service_request").

Core idea: never persist "which question we're on." Every turn,
compute_missing_slot(contact, flow_type) re-derives the single next thing to
ask from the real catalogue (ServiceCategory/ServicePricing) and whatever's
already on contact.selected_services — the same cart the Flow itself builds
towards. That's what makes an interruption free: if a turn doesn't answer the
pending slot, nothing is touched, and the exact same question is recomputed
and re-asked next turn — "resuming" needs no separate bookkeeping.

Contact.active_intake_flow_type is the only new persisted marker: "a guided
chat intake is in progress, for this flow_type." The one ITR-only exception —
which wizard parent (itr_salary/itr_business/itr_nri) was picked, while its
add-ons/quantities are still pending — reuses the existing
Contact.pending_parent_service_code field (already documented for exactly
this "parent currently collecting multiple child selections" purpose), not a
new field, since a wizard parent's own service_code is deliberately never put
on the cart itself (see _apply_answer's "itr_type" branch, mirroring
flow_screens._handle_itr_type_screen).

Built as a small LangGraph StateGraph, same no-checkpointer style as
agents.py: state is the live Contact plus values recomputed from the DB each
call, never persisted graph state.
"""

import re
from typing import TypedDict

from langgraph.graph import END, StateGraph

from . import services
from .models import Contact, ServicePricing

# --- Catalogue mapping ---------------------------------------------------

# Every flat, top-level, multi-select category — same set/shape as
# flow_screens._FLAT_CATEGORY_SCREENS, just keyed by flow_type instead of
# WhatsApp screen id.
_FLAT_FLOW_TYPES = {
    "GST_FILING": ("gst", services.flow_gst_options),
    "TDS_FILING": ("tds_filing", services.flow_tds_filing_options),
    "BUSINESS_REG": ("business_reg", services.flow_business_reg_options),
    "NOTICES": ("notices", services.flow_notice_options),
    "BOOK_EXPERT": ("book_expert", services.flow_book_expert_options),
}

_ADDON_SKIP_WORDS = {
    "none", "no", "nothing", "nothing else", "skip", "that's it", "thats it",
    "no thanks", "nope", "not really", "no addons", "nil",
}

# Free-text answers are split on these before each fragment is matched
# against the real catalogue — lets "quarterly filing and the monthly one"
# resolve as two picks in one message instead of forcing one-at-a-time.
_FRAGMENT_SPLIT = re.compile(r",| and | & |\n")

# Stripped out of a candidate's display name before word-overlap scoring
# (see _match_specific) — mostly the category/product words every sibling in
# a category shares (filing, tax, itr, gst, tds, ...). Without this, a
# message that only names the CATEGORY ("help me with TDS filing") would
# "match" whichever item happens to share that word, guessing instead of
# asking — the exact failure mode services._loose_match's code-stem
# heuristic has when every code in a category shares the same prefix.
_GENERIC_WORDS = {
    "the", "a", "an", "for", "of", "and", "or", "please", "need", "needed",
    "want", "wanted", "i", "to", "my", "me", "do", "does", "did", "with",
    "help", "about", "what", "how", "much", "is", "are", "it", "that", "this",
    "one", "filing", "return", "returns", "service", "services", "tax",
    "gst", "tds", "itr", "registration", "notice", "notices", "consult",
    "consultation", "consultant", "expert", "business", "response",
    "appeal", "assessment", "individual",
}


def _word_set(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _match_specific(candidates, fragment):
    """Resolves `fragment` to exactly one of `candidates`, or None — same
    "ask rather than guess" contract as services.resolve_cart_edit_target.
    Deliberately NOT services._loose_match: that helper's code-stem
    heuristic assumes distinctly-prefixed codes, but every service_code
    within one category here shares the same category prefix (tds_*,
    itr_*, ...), which would make mentioning the category name alone
    false-match the first candidate. Matches instead on how much of a
    candidate's OWN distinctive vocabulary (its display name, minus
    _GENERIC_WORDS) shows up in the fragment, as a fraction of that
    candidate's distinctive word count — picking the best-covered
    candidate, and refusing to guess on a tie or on zero distinctive
    overlap."""
    normalized = fragment.strip().lower()
    if not normalized:
        return None

    for item in candidates:
        if normalized == item.service_code.lower():
            return item
    if normalized.isdigit():
        idx = int(normalized) - 1
        if 0 <= idx < len(candidates):
            return candidates[idx]

    fragment_words = _word_set(normalized)
    scored = []
    for item in candidates:
        distinctive = _word_set(item.display_name) - _GENERIC_WORDS
        if not distinctive:
            continue
        overlap = distinctive & fragment_words
        if overlap:
            scored.append((len(overlap) / len(distinctive), item))
    if not scored:
        return None
    scored.sort(key=lambda pair: -pair[0])
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        return None  # ambiguous between two equally-covered candidates
    return scored[0][1]


# --- Missing-slot computation ----------------------------------------------

def compute_missing_slot(contact, flow_type):
    """The single next thing to ask for, given the real catalogue and
    whatever's already on contact.selected_services — or None once nothing
    is left. Recomputed fresh every call; never trust a stale caller-held
    value across turns."""
    if flow_type == "ITR_TYPE":
        return _compute_itr_missing_slot(contact)
    mapping = _FLAT_FLOW_TYPES.get(flow_type)
    if not mapping:
        return None
    category_code, options_fn = mapping
    scope = services.all_service_codes_in_category(category_code)
    selected = {item.get("service_code") for item in (contact.selected_services or [])}
    if selected & scope:
        return None
    return {"kind": "flat", "category_code": category_code, "flow_type": flow_type, "options": options_fn()}


def _compute_itr_missing_slot(contact):
    if contact.pending_parent_service_code and contact.pending_parent_service_code in services.ITR_ADDON_PARENT_CODES:
        parent = ServicePricing.objects.filter(
            service_code=contact.pending_parent_service_code, is_active=True, category__category_code="itr",
        ).first()
        if parent:
            return {"kind": "itr_addons", "parent": parent}
        # Stale pointer (service since deactivated) — clear it and fall
        # through to asking the type again, same "never invent" contract.
        contact.pending_parent_service_code = ""
        contact.save(update_fields=["pending_parent_service_code"])

    itr_codes = services.all_service_codes_in_category("itr")
    selected = {item.get("service_code") for item in (contact.selected_services or [])}
    if selected & itr_codes:
        return None
    return {"kind": "itr_type", "flow_type": "ITR_TYPE", "options": services.flow_itr_type_options()}


# --- Free-text extraction --------------------------------------------------
# "Ask rather than guess" throughout, same contract as
# services.resolve_cart_edit_target: an extractor returns None (not applied)
# rather than picking a plausible-looking wrong item.

def _extract_flat_answer(category_code, text):
    candidates = list(
        ServicePricing.objects.filter(category__category_code=category_code, is_active=True, parent_service__isnull=True)
    )
    matched = []
    for fragment in _FRAGMENT_SPLIT.split(text):
        fragment = fragment.strip()
        if not fragment:
            continue
        item = _match_specific(candidates, fragment)
        if item and item not in matched:
            matched.append(item)
    return matched or None


def _extract_itr_type_answer(text):
    candidates = list(
        ServicePricing.objects.filter(category__category_code="itr", is_active=True, parent_service__isnull=True)
    )
    return _match_specific(candidates, text)


def _extract_addon_answer(parent, text):
    """Returns (matched_service_codes: set, quantities: dict) or None if
    nothing in `text` was recognized at all (an interruption, not an
    answer). An explicit skip word ("none") is a valid, real answer — empty
    sets, not None."""
    normalized = text.strip().lower()
    if normalized in _ADDON_SKIP_WORDS:
        return set(), {}

    quantity_children = services.itr_quantity_children(parent)
    quantities = {}
    for key, child in quantity_children.items():
        if child is None:
            continue
        keyword = r"house\s*property" if key == "house_property" else r"partnership"
        match = re.search(keyword + r"[^\d]{0,15}(\d+)", normalized) or re.search(r"(\d+)[^\d]{0,15}" + keyword, normalized)
        if match:
            quantities[child.service_code] = int(match.group(1))

    addon_candidates = [s for s in services.optional_children(parent) if s.question_type != ServicePricing.QUESTION_QUANTITY]
    matched = set()
    for fragment in _FRAGMENT_SPLIT.split(text):
        fragment = fragment.strip()
        if not fragment:
            continue
        item = _match_specific(addon_candidates, fragment)
        if item:
            matched.add(item.service_code)

    if not matched and not quantities:
        return None
    return matched, quantities


# --- Applying an answer ----------------------------------------------------

def _apply_answer(contact, slot, text):
    """Tries to resolve `text` as an answer to `slot` and, if it matches,
    writes it into contact.selected_services via the same primitives the
    native Flow's own submission handlers use (services.add_service_selection
    / replace_cart_scope / auto_include_bundled_children — see
    flow_screens._handle_itr_type_screen / _handle_itr_addons_screen /
    _handle_flat_category_screen for the reference behavior this mirrors).
    Returns True if applied, False if `text` didn't answer this slot at all
    (an interruption — caller leaves everything untouched)."""
    kind = slot["kind"]

    if kind == "flat":
        matches = _extract_flat_answer(slot["category_code"], text)
        if not matches:
            return False
        for service in matches:
            services.add_service_selection(contact, service)
        return True

    if kind == "itr_type":
        service = _extract_itr_type_answer(text)
        if not service:
            return False
        itr_codes = services.all_service_codes_in_category("itr")
        if service.service_code in services.ITR_ADDON_PARENT_CODES:
            services.replace_cart_scope(contact, itr_codes, set())
            services.auto_include_bundled_children(contact, service)
            contact.pending_parent_service_code = service.service_code
            contact.save(update_fields=["pending_parent_service_code"])
        else:
            services.replace_cart_scope(contact, itr_codes, {service.service_code})
        return True

    if kind == "itr_addons":
        parent = slot["parent"]
        result = _extract_addon_answer(parent, text)
        if result is None:
            return False
        matched_codes, quantities = result
        scope = {s.service_code for s in services.optional_children(parent)}
        services.replace_cart_scope(contact, scope, matched_codes, quantities)
        contact.pending_parent_service_code = ""
        contact.save(update_fields=["pending_parent_service_code"])
        return True

    return False


# --- Building the "what's next" prompt --------------------------------------

_LIST_BUTTON_LABEL = "Choose an option"


def _build_ask_reply(slot):
    kind = slot["kind"]

    if kind == "flat":
        label = services.FLOW_TYPE_LABELS.get(slot["flow_type"], "that")
        body = f"Sure — which {label} service do you need? Tell me, or tap below."
        return body, None, (body, _LIST_BUTTON_LABEL, slot["options"])

    if kind == "itr_type":
        body = "Which ITR filing type is this for? Tell me, or tap below."
        return body, None, (body, _LIST_BUTTON_LABEL, slot["options"])

    if kind == "itr_addons":
        parent = slot["parent"]
        addon_names = [
            s.display_name for s in services.optional_children(parent)
            if s.question_type != ServicePricing.QUESTION_QUANTITY
        ]
        quantity_children = services.itr_quantity_children(parent)
        extras = []
        if quantity_children["house_property"] is not None:
            extras.append("how many house properties")
        if quantity_children["partnership"] is not None:
            extras.append("how many partnership firms")
        extra_line = f" Also tell me {' and '.join(extras)} you have, if any." if extras else ""
        names_line = f" Options: {', '.join(addon_names)}." if addon_names else ""
        body = (
            f"Got it — {parent.display_name}.{names_line} Do you need any of these too?{extra_line} "
            "Reply 'none' if that's everything."
        )
        return body, None, None

    return None, None, None


# --- Graph state and nodes --------------------------------------------------

class IntakeState(TypedDict, total=False):
    contact: Contact
    text: str
    flow_type: str
    slot: dict
    matched: bool
    next_slot: dict
    reply_text: str
    reply_buttons: tuple
    reply_list: tuple
    trigger_conversion_cta: bool


def _node_resolve(state):
    contact = state["contact"]
    flow_type = state["flow_type"]
    slot = compute_missing_slot(contact, flow_type)
    if slot is None:
        # Nothing was actually missing — a stale/already-cleared marker.
        # Clear it and let the caller fall through to normal free-chat.
        contact.active_intake_flow_type = ""
        contact.save(update_fields=["active_intake_flow_type"])
        return {"slot": None, "matched": False, "next_slot": None}

    matched = _apply_answer(contact, slot, state["text"])
    next_slot = compute_missing_slot(contact, flow_type) if matched else slot
    return {"slot": slot, "matched": matched, "next_slot": next_slot}


def _route_after_resolve(state):
    if state["slot"] is None:
        return "not_applicable"
    if not state["matched"]:
        return "interrupted"
    if state["next_slot"] is None:
        return "complete"
    return "ask"


def _node_ask(state):
    reply_text, reply_buttons, reply_list = _build_ask_reply(state["next_slot"])
    return {"reply_text": reply_text, "reply_buttons": reply_buttons, "reply_list": reply_list}


def _node_complete(state):
    """Everything the flow_type needed is now on the cart. Lands on exactly
    the same SUMMARY -> CONFIRM -> PAYMENT path a cart edit already uses
    (see agents.add_to_cart, whose tail this mirrors) rather than
    reimplementing completion — the pre-existing STEP_CONFIRMING_QUOTE
    handler in tasks.py, and maybe_send_payment_cta via
    trigger_conversion_cta, take it from here unchanged.

    pending_parent_service_code is ITR_TYPE-specific state (which wizard
    parent's add-ons are still being collected — see
    _compute_itr_missing_slot) — only cleared when THIS completion is
    itself the ITR_TYPE intake's. A different flow_type completing (e.g.
    GST_FILING, started as an interruption of a still-pending ITR add-ons
    question — see agents.guided_intake_node) must not wipe that unrelated
    pending marker out from under the ITR intake it belongs to; clearing it
    unconditionally here previously did exactly that, silently dropping the
    interrupted ITR add-ons question instead of it ever being resumed."""
    contact = state["contact"]
    contact.active_intake_flow_type = ""
    update_fields = ["active_intake_flow_type"]
    if state["flow_type"] == "ITR_TYPE" and contact.pending_parent_service_code:
        contact.pending_parent_service_code = ""
        update_fields.append("pending_parent_service_code")
    if contact.pending_service_code:
        contact.pending_service_code = ""
        update_fields.append("pending_service_code")
    contact.save(update_fields=update_fields)
    services.invalidate_stale_orders(contact)
    services.record_last_quote(contact)
    contact.conversation_step = Contact.STEP_CONFIRMING_QUOTE
    contact.save(update_fields=["conversation_step"])
    services.mark_review_reached(contact)
    summary = services.build_selection_complete_reply(contact)
    return {
        "reply_text": f"That's everything for this one. Here's what you've got:\n\n{summary}",
        "reply_buttons": None, "reply_list": None, "trigger_conversion_cta": True,
    }


def _node_not_applicable(state):
    return {"reply_text": None, "reply_buttons": None, "reply_list": None}


def _node_interrupted(state):
    # Deliberately a no-op: active_intake_flow_type and the cart are left
    # exactly as they were. The caller falls through to the normal free-chat
    # path for this turn, and next turn's compute_missing_slot re-asks the
    # same pending question — that's the whole "resume" mechanism.
    return {"reply_text": None, "reply_buttons": None, "reply_list": None}


# --- Graph assembly ----------------------------------------------------------
# Built once at import time, reused across every call — same pattern as
# agents.py's COMPILED_GRAPH. No checkpointer: state is the live Contact plus
# values recomputed from the DB each call, never persisted graph state.

_builder = StateGraph(IntakeState)
_builder.add_node("resolve", _node_resolve)
_builder.add_node("ask", _node_ask)
_builder.add_node("complete", _node_complete)
_builder.add_node("not_applicable", _node_not_applicable)
_builder.add_node("interrupted", _node_interrupted)
_builder.set_entry_point("resolve")
_builder.add_conditional_edges("resolve", _route_after_resolve, {
    "ask": "ask", "complete": "complete", "not_applicable": "not_applicable", "interrupted": "interrupted",
})
for _node in ("ask", "complete", "not_applicable", "interrupted"):
    _builder.add_edge(_node, END)
COMPILED_INTAKE_GRAPH = _builder.compile()


def pending_nudge(contact):
    """A soft one-line reminder to append to a free-chat reply when
    contact.active_intake_flow_type is still set after a turn that didn't
    answer it (see tasks.process_incoming_message's free-chat branch) —
    same 'append to the existing reply, never send a second message'
    pattern as services.build_resume_step_nudge. This is what actually
    delivers "resume the original task" for an interruption: run_guided_intake
    itself leaves state untouched and defers to free-chat for a real answer
    to the interruption, so the pending question has to be re-surfaced here,
    once that answer is in, rather than repeated as its own bot turn."""
    flow_type = contact.active_intake_flow_type
    if not flow_type:
        return ""
    slot = compute_missing_slot(contact, flow_type)
    if slot is None:
        return ""
    question, _buttons, _list = _build_ask_reply(slot)
    if not question:
        return ""
    return f"\n\nWhenever you're ready — {question[0].lower()}{question[1:]}"


def run_guided_intake(contact, text):
    """Entry point called from tasks.process_incoming_message whenever
    contact.active_intake_flow_type is set. Returns
    (reply_text, reply_buttons, reply_list, trigger_conversion_cta).
    reply_text is None for both "not applicable" (marker already cleared)
    and "interrupted" (state left untouched, ready to resume) — the caller
    treats either the same way: fall through to the normal free-chat branch
    for this turn."""
    flow_type = contact.active_intake_flow_type
    if not flow_type:
        return None, None, None, False
    final_state = COMPILED_INTAKE_GRAPH.invoke({"contact": contact, "text": text, "flow_type": flow_type})
    return (
        final_state.get("reply_text"),
        final_state.get("reply_buttons"),
        final_state.get("reply_list"),
        final_state.get("trigger_conversion_cta", False),
    )
