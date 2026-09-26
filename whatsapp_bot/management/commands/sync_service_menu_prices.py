"""
Regenerates the SERVICE screen's literal RadioButtonsGroup data-source in
docs/whatsapp_flows/service_flow.json from live ServicePricing data (a
"From ₹X" teaser per top-level category) and, with --upload, re-uploads the
result to the draft Flow on Meta — same asset-upload pattern as
create_service_flow.py.

Why this exists: SERVICE deliberately stopped depending on the live
data-exchange endpoint at render time (see flow_screens._service_screen_response
and its docstring) — that was the fix for WhatsApp's "RadioButtonsGroup
'service' is missing required property 'data-source'" error, which showed
up whenever the very first INIT round trip didn't resolve before the screen
painted. Going static bought resilience, but a genuinely static price would
quietly drift from real ServicePricing rows and become the exact kind of
"hallucinated fallback" flow_screens.py's own docstring says never to ship.
This command is the answer: SERVICE's numbers still come from the DB, just
at JSON-generation time (run by hand whenever pricing changes) rather than
render time (every single flow open).
"""

import json
import re
from decimal import Decimal
from pathlib import Path

import requests
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from whatsapp_bot.models import ServiceCategory, ServicePricing

GRAPH_API_VERSION = "v22.0"
FLOW_JSON_PATH = Path(__file__).resolve().parents[3] / "docs" / "whatsapp_flows" / "service_flow.json"

_PRICE_RE = re.compile(r"₹\s*([\d,]+)")

# category_code -> (title shown on SERVICE, base description before the
# price teaser is appended) — kept here rather than pulled from
# ServiceCategory.display_name/url since those aren't meant as customer-
# facing blurb text; edit this dict if the pitch copy should change.
_CATEGORY_COPY = {
    "itr": ("ITR Filing", "Income tax return filing"),
    "gst": ("GST Filing", "GST returns & registration"),
    "tds_filing": ("TDS Filing", "TDS payment & return filing"),
    "business_reg": ("Business Registration", "PAN, TAN, company setup & more"),
    "notices": ("Notices", "Help with a tax notice"),
    "book_expert": ("Book Expert", "Talk to a tax expert"),
}


def _min_price(category):
    """Cheapest active, top-level (non-add-on) service in a category, as a
    Decimal — None if nothing in it has a parseable numeric price (e.g.
    every row is 'ask for pricing' or 'Included')."""
    amounts = []
    for service in category.services.filter(is_active=True, parent_service=None):
        if service.question_type == ServicePricing.QUESTION_QUANTITY and service.price_per_unit is not None:
            amounts.append(service.price_per_unit)
            continue
        match = _PRICE_RE.search(service.price_display or "")
        if match:
            amounts.append(Decimal(match.group(1).replace(",", "")))
    return min(amounts) if amounts else None


class Command(BaseCommand):
    help = (
        "Regenerates SERVICE's price teasers in service_flow.json from live "
        "ServicePricing data. Run by hand whenever category pricing changes "
        "meaningfully — SERVICE itself doesn't hit the live endpoint on open, "
        "so this is how it stays honest without giving that up. Pass --upload "
        "to also push the result to WHATSAPP_SERVICE_FLOW_ID on Meta."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--upload", action="store_true",
            help="Also upload the updated JSON to the draft Flow on Meta (WHATSAPP_SERVICE_FLOW_ID).",
        )

    def handle(self, *args, **options):
        if not FLOW_JSON_PATH.exists():
            raise CommandError(f"Flow JSON not found at {FLOW_JSON_PATH}")
        flow = json.loads(FLOW_JSON_PATH.read_text(encoding="utf-8"))

        try:
            service_screen = next(s for s in flow["screens"] if s["id"] == "SERVICE")
            radio = next(
                c for c in service_screen["layout"]["children"][0]["children"]
                if c.get("type") == "RadioButtonsGroup" and c.get("name") == "service"
            )
        except StopIteration:
            raise CommandError("Couldn't find SERVICE's RadioButtonsGroup — has the JSON's shape changed?")

        updated = []
        for category_code, (title, base_description) in _CATEGORY_COPY.items():
            category = ServiceCategory.objects.filter(category_code=category_code).first()
            price = _min_price(category) if category else None
            description = f"{base_description} — from ₹{price:,.0f}" if price is not None else base_description
            updated.append({"id": category_code, "title": title, "description": description})
            self.stdout.write(f"  {category_code}: {description}")

        radio["data-source"] = updated
        FLOW_JSON_PATH.write_text(json.dumps(flow, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self.stdout.write(self.style.SUCCESS(f"Updated {FLOW_JSON_PATH}"))

        if not options["upload"]:
            self.stdout.write("Run again with --upload to push this to Meta's draft Flow.")
            return

        if not settings.WHATSAPP_ACCESS_TOKEN:
            raise CommandError("WHATSAPP_ACCESS_TOKEN is not set in .env.")
        if not settings.WHATSAPP_SERVICE_FLOW_ID:
            raise CommandError("WHATSAPP_SERVICE_FLOW_ID is not set in .env.")

        headers = {"Authorization": f"Bearer {settings.WHATSAPP_ACCESS_TOKEN}"}
        resp = requests.post(
            f"https://graph.facebook.com/{GRAPH_API_VERSION}/{settings.WHATSAPP_SERVICE_FLOW_ID}/assets",
            headers=headers,
            data={"name": "flow.json", "asset_type": "FLOW_JSON"},
            files={"file": ("flow.json", FLOW_JSON_PATH.read_bytes(), "application/json")},
            timeout=15,
        )
        if not resp.ok:
            raise CommandError(f"Upload failed ({resp.status_code}): {resp.text}")
        body = resp.json()
        if body.get("validation_errors"):
            raise CommandError(f"Meta rejected the updated JSON: {json.dumps(body['validation_errors'], indent=2)}")
        self.stdout.write(self.style.SUCCESS("Uploaded to Meta — validation_errors: []"))
