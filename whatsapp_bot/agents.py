"""
Multi-agent layer for the free-text side of the bot — a LangGraph
Supervisor/Router in front of four specialist nodes (Sales & Cart, Guided
Intake, Human Handoff, Order & Status), sitting downstream of
services.classify_and_respond in tasks.process_incoming_message.

Guided Intake (guided_intake_node, backed by intake.run_guided_intake) was
the chat-only alternative to filling the WhatsApp Flow form, reached via
classify_and_respond's direct_service_request action. That action now
auto-launches the deep-linked Flow via sales_cart_node too, same as
launch_flow_candidate — guided_intake_node itself is dead code now (see its
own docstring); intake.run_guided_intake stays alive only to finish out a
chat wizard already in progress for a contact from before this change.

Deliberately ONE reliable LLM, not three: classify_and_respond (still called
from tasks.py, unchanged call site) is the single Gemini round-trip for a
turn — it both drafts a conversational reply and classifies which specialist
should act. The graph in this file owns only the deterministic dispatch and
mutation that follows that one call, via named "tool" functions
(add_to_cart, create_handoff_ticket, get_order_status) each specialist node
calls. A cheaper/different model per node is a later, separable change (see
the module docstring on GEMINI_MODEL in services.py) — nothing here assumes
a single shared model.

The scripted menu/quote/payment funnel (Contact.conversation_step) is
untouched by any of this — this graph only ever runs for the free-chat
fallback path, same as classify_and_respond did on its own before.
"""

import logging
from typing import TypedDict

from django.utils import timezone
from langgraph.graph import END, StateGraph

from .intake import _build_ask_reply, compute_missing_slot, run_guided_intake
from .models import Contact, FlowIntentClassification
from .services import (
    FLOW_TYPE_LABELS,
    add_service_selection,
    build_cart_edit_not_found_reply,
    build_help_reply,
    build_low_confidence_clarifying_reply,
    build_order_status_reply,
    build_selection_complete_reply,
    invalidate_stale_orders,
    mark_review_reached,
    notify_human_alert,
    record_last_quote,
    remove_service_selection,
    resolve_cart_edit_target,
)

logger = logging.getLogger(__name__)


# --- Tool functions ----------------------------------------------------
# Thin, independently callable/testable — each wraps existing, already-
# guardrailed services.py primitives. The graph nodes below are the only
# callers today, but nothing here depends on the graph.

def add_to_cart(contact, operation, query):
    """The Sales & Cart agent's cart-mutation tool. Free text ("remove
    capital gains", "add house property too") never gets to invent a cart
    item or price itself — resolve_cart_edit_target only ever matches
    against the real catalogue (see its docstring), returning None (handled
    below) rather than guessing. The mutation itself is always deterministic
    Python (add/remove_service_selection), never the LLM.

    Extracted from tasks.process_incoming_message's old inline edit_cart
    handling — same behavior, now a reusable, independently testable
    function. Returns (reply_text, trigger_conversion_cta).
    """
    target = resolve_cart_edit_target(contact, operation, query)
    if not target:
        return build_cart_edit_not_found_reply(operation, query), False

    if operation == "add":
        add_service_selection(contact, target)
        verb = "Added"
    else:
        remove_service_selection(contact, target)
        verb = "Removed"
    record_last_quote(contact)
    # The edited cart no longer matches whatever amount any already-issued
    # payment link was quoted at — invalidate those (locally and on
    # Razorpay's side) rather than leave a stale, still-payable link lying
    # around. See services.invalidate_stale_orders's docstring.
    invalidate_stale_orders(contact)
    # Same SUMMARY -> CONFIRM -> PAYMENT hold as every other quote/summary
    # reply — Confirm/Cancel, not Confirm/Edit: there's no
    # pending_parent_service_code to return to here. pending_service_code is
    # cleared too, not just left stale — STEP_CONFIRMING_QUOTE's Cancel
    # handler reads it to know which single item to drop, and a cart edit is
    # never about just one pre-named item.
    contact.conversation_step = Contact.STEP_CONFIRMING_QUOTE
    contact.pending_parent_service_code = ""
    contact.pending_service_code = ""
    contact.save(update_fields=[
        "conversation_step", "pending_parent_service_code", "pending_service_code",
    ])
    mark_review_reached(contact)
    summary = build_selection_complete_reply(contact)
    return f"{verb} {target.display_name}. Here's your updated cart:\n\n{summary}", True


def create_handoff_ticket(contact, reason, detail=""):
    """The Human Handoff agent's tool — flags the contact for staff triage
    (services.notify_human_alert) and returns the reply text. Used to also
    pause the bot on this contact (Contact.bot_paused_until /
    services.HUMAN_HANDOFF_PAUSE_HOURS, the same mechanism the HELP keyword
    and a completed callback booking used in tasks.py) but that pause
    feature was removed — the bot now keeps auto-replying after a handoff
    instead of going silent for staff to pick up, so staff still gets
    notified here but the customer is never left hanging.

    Deliberately does NOT touch has_requested_help — that flag's documented
    meaning is specifically "ever sent the HELP keyword," not "asked in any
    words," so a free-text ask routed here shouldn't silently redefine it.
    """
    notify_human_alert(contact, reason, detail=detail)
    return build_help_reply()


def get_order_status(contact):
    """The Order & Status agent's tool — thin wrapper over
    services.build_order_status_reply, a pure DB read against Order.status
    (gateway-reconciled payment status). No LLM in this decision, same as
    the STATUS keyword's existing handling."""
    return build_order_status_reply(contact)


# --- Graph state ---------------------------------------------------------

class AgentState(TypedDict, total=False):
    contact: Contact
    text: str
    # The dict services.classify_and_respond already returns — computed
    # once, before the graph is invoked (see run_agent_graph), so this stays
    # a single Gemini call per turn, not a second one inside the graph.
    classification: dict
    reply_text: str
    reply_buttons: tuple
    reply_list: tuple
    # Set by sales_cart_node's launch_flow_candidate branch (auto-launch —
    # see its docstring) — tasks.process_incoming_message's dispatch opens
    # the deep-linked mega Flow straight to this flow_type's screen instead
    # of sending reply_text/reply_buttons/reply_list as a normal message.
    reply_service_flow_target: str
    # The Flow-opening message's own body text, alongside
    # reply_service_flow_target — Gemini's own conversational reply, shown
    # right on the Flow card since there's no separate confirm message for
    # it to live on anymore. None (services.send_service_flow's own generic
    # greeting) for every other path that still sets reply_service_flow_target
    # (e.g. tasks.py's legacy pending_flow_type_candidate confirm branch).
    reply_service_flow_body: str
    trigger_conversion_cta: bool
    agent: str  # which node handled this turn — logging/debugging only


# --- Nodes -----------------------------------------------------------------

def supervisor_node(state):
    """Entry point / router node. classification was already produced by
    services.classify_and_respond before this graph was invoked — this node
    exists to give the graph a real Supervisor entry point (see
    route_after_supervisor) without a second LLM round-trip per turn."""
    return {}


def sales_cart_node(state):
    contact = state["contact"]
    classification = state["classification"]
    text = state["text"]

    if classification["action"] in ("launch_flow_candidate", "direct_service_request"):
        # Auto-launch straight away — route_after_supervisor only ever sends
        # a "high"/"medium" confidence guess here ("low" goes to
        # low_confidence_clarify_node instead), so by the time this branch
        # runs the guess is trusted enough to act on immediately rather than
        # spend a whole extra message/tap on a Confirm/Edit prompt first (see
        # report_flow_intent_confidence's historical reject rates before this
        # changed). Reuses Gemini's own natural-voice `reply` as the Flow
        # message's own body (via reply_service_flow_body/send_service_flow's
        # body_text) so the acknowledgment still reaches the customer, just
        # on the same message as the Flow button instead of a message before
        # it. Recovery from a wrong guess is just typing the real ask on the
        # next message — same free-chat re-classification as any other turn,
        # no dedicated "Something else" button needed.
        #
        # direct_service_request (a specific item named, e.g. "TDS quarterly
        # filing", or ITR inferred from stated employment details) used to go
        # to guided_intake_node instead — resolve the item server-side and
        # ask about add-ons in chat. That's legacy now (see
        # route_after_supervisor's docstring): the customer configures
        # add-ons on the Flow's own screen instead. One real trade-off from
        # merging this in: the specific item they already named isn't
        # pre-checked on that screen (flow_screens.py's deep-link responses
        # don't apply a default selection) — they see the same full
        # checklist as a vague ask would, just landed on directly instead of
        # via SERVICE.
        flow_type = classification["flow_type"]
        # Durable counterpart to the [flow-intent] log line in tasks.py —
        # see FlowIntentClassification's docstring for why a log line alone
        # isn't enough to ever answer "is confidence=high trustworthy?" with
        # real data. Recorded (and resolved) immediately as auto_launched —
        # there's no separate Confirm/Edit answer to wait on anymore.
        FlowIntentClassification.objects.create(
            contact=contact, flow_type=flow_type, confidence=classification["confidence"], message_text=text,
            decision=FlowIntentClassification.DECISION_AUTO_LAUNCHED, resolved_at=timezone.now(),
        )
        logger.info("[flow-intent] contact=%s flow_type=%s decision=auto_launched", contact.id, flow_type)
        reply_text = classification["reply"]
        return {
            "reply_text": reply_text, "reply_service_flow_target": flow_type,
            "reply_service_flow_body": reply_text, "agent": "sales_cart",
        }

    # edit_cart — the only other action route_after_supervisor sends here.
    reply_text, trigger_conversion_cta = add_to_cart(
        contact, classification["cart_edit_operation"], classification["cart_edit_query"],
    )
    return {"reply_text": reply_text, "trigger_conversion_cta": trigger_conversion_cta, "agent": "sales_cart"}


def guided_intake_node(state):
    """DEAD as of the direct_service_request auto-launch change —
    route_after_supervisor no longer ever returns "guided_intake", so
    nothing in the compiled graph reaches this node anymore.
    direct_service_request now auto-launches the deep-linked Flow via
    sales_cart_node too (see its docstring), same as launch_flow_candidate.
    Left defined (and still wired into the graph below) only so the module
    still imports/compiles cleanly and in case a future flow_type needs the
    chat wizard back for some reason — not because anything calls it today.

    intake.run_guided_intake ITSELF is still very much alive though — a
    contact who already has active_intake_flow_type set from before this
    change is still walked through it directly by
    tasks.process_incoming_message's active_intake_flow_type branch (which
    calls run_guided_intake straight from intake.py, not through this node)
    until that one in-progress wizard finishes.

    Entry point for the chat-only guided intake (intake.run_guided_intake).
    Starts contact.active_intake_flow_type for this turn, then immediately
    tries to resolve the SAME message's own text as the answer to the first
    missing slot — a message that named a specific item (e.g. "TDS quarterly
    filing") shouldn't have to be repeated back to the customer as a
    question. If it isn't specific enough to resolve on its own, the
    customer just sees the first slot's question, no different from typing
    anything else mid-intake.

    Rules 12-14 (interrupt/preserve/resume): a customer can name a second,
    different service (flow_type) while a first one's own intake question is
    still pending — e.g. "Do you need any addons?" (ITR_TYPE) answered
    instead with "I also need GST nil return for my business" (GST_FILING).
    That's handled as its own request below (never dropped in favor of the
    old one — rule 11, detect intent every message), but the interrupted
    flow_type's pending question must survive to be resumed after, not be
    silently lost just because active_intake_flow_type got overwritten to
    the new one. See the resume block at the end of this function."""
    contact = state["contact"]
    classification = state["classification"]
    flow_type = classification["flow_type"]

    previous_flow_type = contact.active_intake_flow_type
    interrupted_slot = None
    if previous_flow_type and previous_flow_type != flow_type:
        interrupted_slot = compute_missing_slot(contact, previous_flow_type)

    contact.active_intake_flow_type = flow_type
    contact.save(update_fields=["active_intake_flow_type"])

    reply_text, reply_buttons, reply_list, trigger_conversion_cta = run_guided_intake(
        contact, classification["direct_service_query"] or state["text"],
    )
    if reply_text is None:
        # The opening message wasn't itself a specific-enough answer (e.g.
        # "I need to register my company" names the category but not which
        # registration) — active_intake_flow_type stays set, so fall back to
        # simply asking the first missing slot's question directly rather
        # than leaving the customer with no reply this turn.
        slot = compute_missing_slot(contact, flow_type)
        if slot is None:
            # Nothing was actually missing (unexpected, but never leave the
            # customer without a reply) — fall through to Gemini's own reply.
            contact.active_intake_flow_type = ""
            contact.save(update_fields=["active_intake_flow_type"])
            return {"reply_text": classification["reply"], "agent": "guided_intake"}
        reply_text, reply_buttons, reply_list = _build_ask_reply(slot)

    if interrupted_slot is not None and compute_missing_slot(contact, flow_type) is None:
        # The interrupting request (flow_type) is itself fully resolved this
        # turn (nothing left pending for it) — safe to hand the next turn
        # back to the flow_type it interrupted, rather than leaving it
        # dropped. If flow_type is NOT yet resolved (still asking its own
        # next question), that question takes priority for the next reply
        # instead — active_intake_flow_type is left on flow_type as already
        # set above, and interrupted_slot's own state (e.g.
        # pending_parent_service_code) is untouched either way (see
        # intake._node_complete's docstring), so nothing is lost even if
        # this doesn't fire this turn.
        contact.active_intake_flow_type = previous_flow_type
        contact.save(update_fields=["active_intake_flow_type"])
        resume_question, _resume_buttons, _resume_list = _build_ask_reply(interrupted_slot)
        if resume_question:
            previous_label = FLOW_TYPE_LABELS.get(previous_flow_type, "that")
            reply_text = (
                f"{reply_text}\n\nAlso, back to your {previous_label} — "
                f"{resume_question[0].lower()}{resume_question[1:]}"
            )
            reply_buttons = None
            reply_list = None

    return {
        "reply_text": reply_text, "reply_buttons": reply_buttons, "reply_list": reply_list,
        "trigger_conversion_cta": trigger_conversion_cta, "agent": "guided_intake",
    }


def human_handoff_node(state):
    reply_text = create_handoff_ticket(state["contact"], "chat_requested_human", detail=state["text"])
    return {"reply_text": reply_text, "agent": "human_handoff"}


def order_status_node(state):
    reply_text, reply_buttons, reply_list = get_order_status(state["contact"])
    return {"reply_text": reply_text, "reply_buttons": reply_buttons, "reply_list": reply_list, "agent": "order_status"}


def general_chat_node(state):
    # continue_chat / clarify_first, and the defensive fallthrough for a
    # launch/edit_cart classification that didn't actually pass its guard
    # (see route_after_supervisor) — reply is Gemini's own drafted `reply`,
    # unedited, same as today.
    return {"reply_text": state["classification"]["reply"], "agent": "general_chat"}


def low_confidence_clarify_node(state):
    """A launch_flow_candidate/direct_service_request classification whose
    own confidence is "low" gets cross-questioned instead of acted on — see
    route_after_supervisor's confidence gate, ahead of the flow_type check
    that would otherwise send it to sales_cart/guided_intake.

    Deliberately does NOT reuse classification["reply"]: that text was
    drafted by Gemini as if the guess were correct (e.g. "Got it — let's
    get your GST registration sorted"), which is exactly the overconfident
    tone that caused the original GST-misread-as-ITR bug — a customer
    shouldn't see a specific, confident-sounding guess (or have the wrong
    Flow screen auto-launched outright, which is what sales_cart_node now
    does for anything above "low" — see its docstring) for something Gemini
    itself flagged as uncertain. build_low_confidence_clarifying_reply asks
    a real, targeted question instead, so the next message gets a genuine
    answer to route on rather than another guess. This is now the only
    remaining checkpoint before a Flow auto-launches.

    Deliberately does NOT set pending_flow_type_candidate or create a
    FlowIntentClassification row — nothing was actually proposed to the
    customer here for them to confirm/reject, so there's nothing pending to
    track; the [flow-intent] log line in tasks.py still records the raw
    guess (flow_type + confidence=low) for later analysis regardless."""
    flow_type = state["classification"]["flow_type"]
    return {"reply_text": build_low_confidence_clarifying_reply(flow_type), "agent": "low_confidence_clarify"}


def route_after_supervisor(state):
    """Faithful port of tasks.py's old if/elif guards, so the exact same
    edge cases (an empty cart, a launch action with no valid flow_type)
    keep falling through to general_chat instead of the specialist.

    launch_flow_candidate and direct_service_request are routed identically
    now — both auto-launch the deep-linked Flow via sales_cart_node (see its
    docstring). direct_service_request used to go to guided_intake_node
    instead (resolve the specific item + ask about add-ons in chat); that
    path is legacy-only now (kept for a contact who already has
    active_intake_flow_type set from before this change — see tasks.py) —
    nothing routes a fresh message there anymore, since configuring add-ons
    on the Flow's own screen replaces the chat wizard entirely."""
    c = state["classification"]
    if c["action"] in ("launch_flow_candidate", "direct_service_request") and c["flow_type"]:
        if c["confidence"] == "low":
            # Cross-question rather than act on a guess Gemini itself
            # flagged as uncertain — checked ahead of the action-specific
            # branches below so it applies to both actions identically.
            return "low_confidence_clarify"
        return "sales_cart"
    if c["action"] == "edit_cart" and c["cart_edit_operation"] and state["contact"].selected_services:
        return "sales_cart"
    if c["action"] == "request_handoff":
        return "human_handoff"
    if c["action"] == "check_order_status":
        return "order_status"
    return "general_chat"


# --- Graph assembly --------------------------------------------------------
# Built once at import time, reused across every call — same "build once,
# invoke many" pattern as services._get_client()'s cached Gemini client. No
# checkpointer: state holds a live Contact instance and is never persisted
# or resumed across turns, it only needs to live for one graph.invoke() call.

_builder = StateGraph(AgentState)
_builder.add_node("supervisor", supervisor_node)
_builder.add_node("sales_cart", sales_cart_node)
_builder.add_node("guided_intake", guided_intake_node)
_builder.add_node("human_handoff", human_handoff_node)
_builder.add_node("order_status", order_status_node)
_builder.add_node("general_chat", general_chat_node)
_builder.add_node("low_confidence_clarify", low_confidence_clarify_node)
_builder.set_entry_point("supervisor")
_builder.add_conditional_edges("supervisor", route_after_supervisor, {
    "sales_cart": "sales_cart",
    "guided_intake": "guided_intake",
    "human_handoff": "human_handoff",
    "order_status": "order_status",
    "general_chat": "general_chat",
    "low_confidence_clarify": "low_confidence_clarify",
})
for _node in (
    "sales_cart", "guided_intake", "human_handoff", "order_status", "general_chat", "low_confidence_clarify",
):
    _builder.add_edge(_node, END)
COMPILED_GRAPH = _builder.compile()


def run_agent_graph(contact, text, classification):
    """Entry point called from tasks.process_incoming_message once
    services.classify_and_respond has produced a classification for this
    turn. Returns a dict with safe defaults for whichever keys the taken
    branch didn't set."""
    final_state = COMPILED_GRAPH.invoke({
        "contact": contact, "text": text, "classification": classification,
    })
    return {
        "reply_text": final_state.get("reply_text"),
        "reply_buttons": final_state.get("reply_buttons"),
        "reply_list": final_state.get("reply_list"),
        "reply_service_flow_target": final_state.get("reply_service_flow_target"),
        "reply_service_flow_body": final_state.get("reply_service_flow_body"),
        "trigger_conversion_cta": final_state.get("trigger_conversion_cta", False),
        "agent": final_state.get("agent"),
    }
