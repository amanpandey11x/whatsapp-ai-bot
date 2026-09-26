# WhatsApp Flows

Two separate Flows live in this project — don't confuse them:

- **The ITR add-on Flow** (this file, below) — a small, fully static,
  single-screen-or-two Flow with no server round trip. `WHATSAPP_ADDON_FLOW_ID`.
- **The mega service Flow** (`service_flow.json`) — the full server-driven
  journey: pick a category, then ITR type/add-ons or GST filings, ending in a
  live-computed review screen. Every screen's data comes from
  `whatsapp_bot.views.whatsapp_flow_endpoint` via `flow_screens.py`, reading
  the same `ServiceCategory`/`ServicePricing` tables the chat wizard uses —
  see the module docstrings in `flow_screens.py` and `services.py` for the
  screen-by-screen logic. `WHATSAPP_SERVICE_FLOW_ID`, created via
  `manage.py create_service_flow` (needs `WHATSAPP_FLOW_ENDPOINT_URL` set to
  this project's real, internet-reachable `/api/whatsapp/flow/` URL, and the
  public-key setup below already done). All 5 categories are wired: ITR
  (type-then-addons), and GST/Business Registration/Notices/Book Expert as
  flat multi-select lists (see `flow_screens._FLAT_CATEGORY_SCREENS`). A
  `COMING_SOON` screen stays as a safety net for any category_code the
  catalogue doesn't recognise (a stale client, a future 6th category not
  wired here yet), not a real destination today. Not yet wired as the
  automatic welcome-message trigger — call `services.send_service_flow(contact)`
  to open it for a contact once you're ready to switch that over.

## ITR add-on WhatsApp Flow (optional)

By default, the ITR add-on picker is a WhatsApp **list** message — every
optional item as a tappable row, one round trip (tap → bot recomputes the
cart → bot re-sends the list) per add-on. That's already backend-computed
(no LLM call, no extra thinking on the bot's side) but it's still one message
exchange per item.

Setting `WHATSAPP_ADDON_FLOW_ID` switches this to a **WhatsApp Flow**: a
single native form screen with a checkbox per add-on. The customer ticks
everything that applies and hits *Submit* once — the whole selection lands on
the backend in one message, applied in a single pass, landing straight on the
same "Confirm & Done" review screen. No code changes needed to turn it on —
just publish the flow once and set the id.

## One-time setup

1. Make sure `.env` has `WHATSAPP_ACCESS_TOKEN` and `META_WA_BUSINESS_ACCOUNT_ID`
   set (the WABA id, not the phone number id — find it in Meta Business
   Manager → WhatsApp Accounts).
2. From `whatsapp_ai_bot/` (same folder as `manage.py`), run:

   ```
   python manage.py create_addon_flow
   ```

   This creates the flow, uploads `addon_selection_flow.json` as its one
   screen, and leaves it as a draft. Preview it in Meta Business Manager →
   WhatsApp Manager → Flows.
3. Once it looks right, either publish it from that UI, or re-run:

   ```
   python manage.py create_addon_flow --publish
   ```

   Publishing is irreversible for that flow's structure — a published flow
   can't be edited further, only replaced by creating a new one (re-run the
   command, it always creates a fresh flow id).
4. Copy the printed flow id into `.env` as `WHATSAPP_ADDON_FLOW_ID` and
   restart the Celery worker. Leave it blank to keep the tap-list instead —
   nothing else about the bot depends on this being set.

## Why no server-side data endpoint is needed

`addon_selection_flow.json` is a fully static, single-screen flow — its
options/prices come from `${data.*}` placeholders that
`services.build_addon_flow_payload` fills in and sends along with every
message (`flow_action_payload.data`), and the *Submit* button is a `complete`
action, not a `data_exchange` call to a server mid-flow. That means:

- Admin-edited prices/descriptions on `ServicePricing` still show up
  immediately, same as the tap-list — nothing is baked into the flow JSON
  itself.
- No extra endpoint, no request signing/encryption setup, no webhook
  changes beyond what's already there (a submitted flow arrives as a normal
  incoming message, type `interactive` / `nfm_reply` — see
  `views._extract_message_text` and `services.parse_addon_flow_reply`).

If a future add-on screen needs to fetch something live from the backend
mid-flow (not just at open time), that's the point to add a real Flow
endpoint — not needed for this one.

## Known limitations

- Quantity-priced add-ons (House Property, Partnership Firm) are shown as
  plain checkboxes on the Flow screen too. Ticking one that wasn't already
  selected still detours to the existing "how many?" text question after the
  Flow submits (`tasks.py`, `STEP_AWAITING_INCOME_ITEM`) — the Flow itself
  doesn't collect a quantity inline. This only affects those two items; every
  flat-priced add-on is fully handled by the one submit.
- The checkbox screen can't come back pre-ticked with what's already on the
  cart (e.g. re-entering via "Edit more") — WhatsApp only allows a
  CheckboxGroup's `init-value` when it sits *outside* a Form, but the Submit
  button needs it *inside* the Form to collect `${form.addons}`. Already-added
  items are instead flagged in their description ("Already added — tick
  again to keep it") — leaving one unticked on a re-submit drops it from the
  cart, same as unticking it would.

## Funnel additions (qualifying question, abandoned-cart nudge)

There is no `LANGUAGE` screen in `service_flow.json` — language is
collected once, in chat, before the Flow is ever opened (see tasks.py's
unconditional language gate in `process_incoming_message`), so `SERVICE` is
the Flow's entry screen. `SERVICE` also carries the "First time filing with
us?" (Yes/No) qualifying question, saved to `Contact.first_time_filer` — a
lead-routing signal only (who needs a senior consultant vs a template
reply), not a pricing input. It's wrapped in an `If` block driven by
`show_first_time_question` (see `flow_screens._service_screen_response`) so
it's only asked once, not on every reopen. **Any edit to `service_flow.json`
(this included) requires re-publishing the Flow** before it takes effect for
real customers — re-run `manage.py create_service_flow` (or push the updated
JSON via Meta Flow Builder), then `--publish` once it looks right in
preview, same as the one-time setup above.

Reaching `REVIEW` unconfirmed (or the legacy chat wizard's Confirm/Edit
screen) starts a nudge clock (`services.mark_review_reached`) — a Celery
Beat task (`whatsapp_bot.tasks.send_abandoned_cart_nudges`, scheduled in
`myproject/celery.py`) reminds the contact at +2h and +24h if they never
confirm. **This requires `celery -A myproject beat` running alongside the
worker** (see `start_bot.ps1`) — without it, nudges (and the post-booking
referral ask) silently never fire.

**Known limitation:** WhatsApp only allows free-form message sends within
24h of the customer's last inbound message. The +2h nudge is safely inside
that window; the +24h nudge (and the +2h one for a contact who's gone
unusually quiet) will fail without an approved WhatsApp Message Template —
the task still attempts the send and logs the failure clearly rather than
silently pretending it worked. Submitting a real template for approval in
Meta Business Manager, and swapping `send_whatsapp_message` for a template
send in `send_abandoned_cart_nudges`, is a follow-up once that's set up.
