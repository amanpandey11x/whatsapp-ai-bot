# Conversational State — Non-Negotiable Rules

These rules govern every change to the message-handling path (`whatsapp_bot/tasks.py`,
`whatsapp_bot/agents.py`, `whatsapp_bot/services.py`, `whatsapp_bot/intake.py`,
`whatsapp_bot/models.py`). They are **not guidelines** — a PR that violates one of
these must be rejected or fixed before merge, including PRs authored by Claude.
If a requested change conflicts with a rule below, say so explicitly instead of
silently complying.

Mapping key, since this project already has concrete stand-ins for the generic terms:

| Generic term            | This project's concrete implementation |
|--------------------------|-----------------------------------------|
| `user_id`                | `Contact.phone_number` (unique, indexed) |
| `conversation_id`        | `Contact.id` (one Contact = one conversation; no multi-thread per number today) |
| fast active state        | Redis (via Celery broker/result backend + any ad-hoc locks) |
| permanent record store   | PostgreSQL/SQLite via Django ORM (`Contact`, `Message`, `FlowIntentClassification`, etc.) |
| structured state machine | `Contact.conversation_step` (scripted funnel) **and** the LangGraph `StateGraph` in `agents.py` (free-chat fallback) |
| LLM                      | `services.classify_and_respond` (Gemini) |
| backend APIs/tools       | `services.py` primitives + the "tool" functions in `agents.py` (`add_to_cart`, `create_handoff_ticket`, `get_order_status`, etc.) |

---

## The rules

1. **Never process a message without loading conversation state first.** Every
   entry point in `tasks.process_incoming_message` must load the `Contact` row
   (and its `conversation_step` / LangGraph state) before doing anything else —
   no branch may act on a message using only the raw webhook payload.

2. **Every user and conversation must have a stable identifier.** Use
   `Contact.phone_number` and `Contact.id` as those identifiers. Never key state,
   caches, locks, or logs by transient values (message id, request id, in-memory
   object id) as a substitute.

3. **Every conversation has exactly one clearly defined current state at a time.**
   That is `Contact.conversation_step` for the scripted funnel, or the LangGraph
   node currently active for the free-chat graph — never both ambiguously, and
   never left `None`/undefined mid-turn.

4. **Structured state is kept separately from raw message history.**
   `Contact`/model fields (structured) and `Message` rows (raw history) are
   distinct; do not encode structured state as a magic string inside a stored
   message, and do not reconstruct structured state by re-parsing message
   history at request time.

5. **Never ask the user for information already present in state.** Before
   building any prompt/question (`_build_ask_reply`, guided intake, Flow
   screens), check `Contact` fields and `compute_missing_slot` — only ask for
   what is genuinely missing.

6. **After every meaningful user message, persist updated state.** Any node/tool
   that changes `conversation_step`, cart contents, quote amounts, or LangGraph
   state must `save()` the `Contact` (or otherwise commit) before the turn ends
   — no state change may live only in a local variable past the handler's return.

7. **Never let in-memory Python variables be the source of truth for
   persistent conversation state.** Local variables, module-level globals, or
   Celery task closures are transient by construction; the source of truth is
   always the DB row (and Redis for fast/ephemeral state). A restart or a task
   retry must reproduce identical behavior from persisted state alone.

8. **Use Redis for fast/active state, PostgreSQL for permanent records** — this
   project already does this via Celery. Don't invent a third ad-hoc store
   (e.g., a JSON file, an in-process dict) for anything that must survive a
   worker restart or be shared across workers.

9. **The LLM (`classify_and_respond`) is never the source of truth for prices,
   payments, orders, or service availability.** Those always come from
   `services.py` / `ServicePricing` / order models. The LLM may draft language
   and classify intent; it may not compute or assert a price, payment status,
   or availability that isn't read from the DB.

10. **Transactional/business operations go through backend functions, not LLM
    output.** Cart mutation, handoff ticket creation, order status lookup, and
    payment processing must go through the existing `services.py` functions /
    `agents.py` tool functions — never through the LLM directly emitting a
    "result" for these.

11. **Detect user intent on every message.** `classify_and_respond` (or the
    LangGraph router) must classify intent every turn, even mid-flow — don't
    short-circuit classification just because a scripted step is in progress,
    unless rule 12/13 already resolved the branch deterministically.

12. **High-priority intents interrupt the current state.** Human handoff
    (`create_handoff_ticket` / `HUMAN_HANDOFF_PAUSE_HOURS`), cancellation,
    payment problems, and complaints must be able to preempt
    `conversation_step` or the active LangGraph node, regardless of where the
    user is in the funnel or guided intake.

13. **Preserve the previous task across an interruption.** When an
    interruption (rule 12) fires, snapshot enough state (`conversation_step`,
    partially-collected intake slots, cart-in-progress) to resume — don't
    discard funnel/intake progress just because a handoff or complaint fired.

14. **Resume the previous task after the interruption is resolved.** Once a
    handoff/complaint/cancellation is closed, conversation_step / graph state
    returns to (or explicitly re-enters) the point it was interrupted from,
    rather than restarting the funnel from scratch.

15. **Track collected vs. missing information separately.** `Contact` fields
    that are set = collected; `compute_missing_slot` (or equivalent) determines
    what's missing. Never infer "missing" by re-deriving it from a free-text
    reply each time — keep it as explicit, queryable state.

16. **Ask for only the next required piece of information at a time.** Guided
    intake and Flow screens must prompt for one missing slot per turn, not a
    batch of unrelated questions.

17. **Validate LLM-extracted information before storing it.** Anything
    `classify_and_respond` or a LangGraph node extracts from free text
    (quantity, topic, income head, phone/email, etc.) must pass validation
    (type/format/range/enum membership against `ServicePricing`,
    `CONVERSATION_STEP_CHOICES`, etc.) before being written to `Contact` or
    used to mutate the cart.

18. **Never assume ambiguous replies like "yes"/"that one"/"do it" without
    checking prior context.** Resolve these against the last bot message /
    last offered option stored in state (e.g., `last_quote_amount`,
    `pending_flow_type_candidate`) — never guess.

19. **Require explicit confirmation before irreversible or important actions.**
    Payment initiation, order creation, cart submission, and cancellation must
    have an explicit confirm step (`STEP_CONFIRMING_SELECTION`,
    `STEP_CONFIRMING_QUOTE`, or an equivalent LangGraph confirm node) — never
    fire on an inferred "yes."

20. **Maintain a conversation summary for long conversations.** Keep
    `Contact.summary` (already a field) up to date so prompts to the LLM don't
    require replaying full message history.

21. **Send only relevant context to the LLM, not the entire history, whenever
    possible.** Build `classify_and_respond` prompts from `Contact.summary` +
    recent turns + relevant structured state — not an unbounded `Message`
    history dump.

22. **Never invent information when required data is unavailable.** If a price,
    slot, or order status isn't in the DB, say so or ask — never have the LLM
    fabricate a plausible-looking value.

23. **If confidence is low, ask a clarification question** rather than guessing
    intent, a slot value, or which cart item a reply refers to.

24. **If the bot cannot safely/correctly complete the task, hand off to a
    human.** Route through `create_handoff_ticket` / `notify_human_alert`
    rather than looping the user or returning a best-effort wrong answer.

25. **State transitions must be explicit, logged, and recoverable.** Every
    `conversation_step` change and every LangGraph node transition should be
    traceable (via logging and/or a persisted field) — no silent/implicit step
    changes buried in unrelated code paths.

26. **Every state must have a defined entry condition, required information,
    allowed actions, exit condition, and next possible states.** When adding a
    new `conversation_step` or LangGraph node, document these five things
    (docstring or comment is sufficient) — don't add an undocumented state.

27. **Conversation state must survive server/worker restarts.** Since state
    lives in Postgres/SQLite + Redis (not process memory), this should already
    hold — any new feature must preserve it (no new state kept only in a
    running process).

28. **Conversation state must be recoverable if a WhatsApp message is processed
    twice** (webhook retry / at-least-once delivery). Handlers must be
    idempotent with respect to `Message.whatsapp_message_id` /
    `Message.id` dedup, not just "usually only run once."

29. **Prevent duplicate execution of important operations** (order creation,
    payment processing, handoff ticket creation). Use existing uniqueness
    constraints/dedup checks (e.g., `whatsapp_message_id` uniqueness) or add
    an explicit idempotency guard before adding any new side-effecting
    operation — never rely on "the webhook probably won't retry."

30. **Always prioritize the user's actual request over forcing them into a
    predefined WhatsApp Flow.** If a user's free text clearly states what they
    want, route them via `direct_service_request` / guided intake / the
    LangGraph free-chat path instead of insisting they complete the Flow form,
    per the existing design note in `agents.py`.

---

## Applying this while working with LangGraph specifically

- The `StateGraph` in `agents.py` is the free-chat fallback path only — the
  scripted funnel (`Contact.conversation_step`) is a separate state machine.
  Rules above apply to **both**; don't satisfy a rule in one and skip it in
  the other.
- LangGraph node functions are the natural place to enforce rules 6, 7, 9, 10,
  25, 26: each node should read persisted `Contact` state in, call
  validated/guardrailed tool functions, and persist state out — no node should
  hold state only in its local `TypedDict` beyond the current invocation.
- When adding a new specialist node, treat it as a new "state" for rule 26:
  document entry condition, required info, allowed actions (which tool
  functions it may call), exit condition, and which nodes/END it can transition
  to.
- Interruption/resume (rules 12–14) should be modeled as a router-level check
  that can redirect to a handoff/complaint node from any node, while the
  interrupted node's partial state remains persisted for later resumption —
  not as special-casing inside every individual node.
