"""
One-time setup helper: creates and publishes the server-driven "mega"
WhatsApp Flow (service category through add-ons/quantities to a final
review — docs/whatsapp_flows/service_flow.json) in Meta Business Manager,
wired to this project's live data-exchange endpoint
(whatsapp_bot.views.whatsapp_flow_endpoint). Not called anywhere at
runtime — run it by hand once, then copy the printed flow id into
WHATSAPP_SERVICE_FLOW_ID (.env). See docs/whatsapp_flows/README.md.

Unlike create_addon_flow.py (a fully static Flow, no endpoint_uri needed),
this Flow declares data_api_version in its JSON, so Meta requires an
endpoint_uri at creation time — WHATSAPP_FLOW_ENDPOINT_URL must already be
a real, internet-reachable HTTPS URL, and public-key registration
(manage.py setup_flow_encryption) must already be done, before this
command's ping/preview will actually succeed end-to-end.
"""

import json
from pathlib import Path

import requests
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

GRAPH_API_VERSION = "v22.0"
FLOW_JSON_PATH = Path(__file__).resolve().parents[3] / "docs" / "whatsapp_flows" / "service_flow.json"


class Command(BaseCommand):
    help = (
        "Creates the mega service WhatsApp Flow in Meta Business Manager, uploads "
        "its screen JSON, and wires it to this project's live data-exchange "
        "endpoint. Prints the flow id to put in WHATSAPP_SERVICE_FLOW_ID once "
        "it's published (--publish) or reviewed by hand."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--publish", action="store_true",
            help="Also publish the flow immediately after upload (irreversible — "
            "a published flow's structure can no longer be edited, only a new "
            "one created). Without this flag the flow is left as a draft so you "
            "can preview it in Meta Business Manager first.",
        )
        parser.add_argument(
            "--name", default="TaxPlan Advisors — Service Flow",
            help="Display name for the flow in Meta Business Manager.",
        )

    def handle(self, *args, **options):
        if not settings.WHATSAPP_ACCESS_TOKEN:
            raise CommandError("WHATSAPP_ACCESS_TOKEN is not set in .env.")
        if not settings.WHATSAPP_WABA_ID:
            raise CommandError(
                "META_WA_BUSINESS_ACCOUNT_ID is not set in .env — this is your "
                "WhatsApp Business Account id (not the phone number id), find it "
                "in Meta Business Manager > WhatsApp Accounts."
            )
        if not settings.WHATSAPP_FLOW_ENDPOINT_URL:
            raise CommandError(
                "WHATSAPP_FLOW_ENDPOINT_URL is not set in .env — the public HTTPS "
                "URL of this project's /api/whatsapp/flow/ path (Meta calls it "
                "directly, so it must be internet-reachable, not localhost)."
            )
        if not FLOW_JSON_PATH.exists():
            raise CommandError(f"Flow JSON not found at {FLOW_JSON_PATH}")

        headers = {"Authorization": f"Bearer {settings.WHATSAPP_ACCESS_TOKEN}"}
        base = f"https://graph.facebook.com/{GRAPH_API_VERSION}"

        self.stdout.write("Creating flow shell...")
        create_resp = requests.post(
            f"{base}/{settings.WHATSAPP_WABA_ID}/flows",
            headers=headers,
            json={
                "name": options["name"],
                "categories": ["OTHER"],
                "endpoint_uri": settings.WHATSAPP_FLOW_ENDPOINT_URL,
            },
            timeout=15,
        )
        self._raise_for_error(create_resp, "create the flow")
        flow_id = create_resp.json()["id"]
        self.stdout.write(self.style.SUCCESS(f"Created flow id={flow_id}"))

        self.stdout.write("Uploading screen JSON...")
        flow_json_bytes = FLOW_JSON_PATH.read_bytes()
        upload_resp = requests.post(
            f"{base}/{flow_id}/assets",
            headers=headers,
            data={"name": "flow.json", "asset_type": "FLOW_JSON"},
            files={"file": ("flow.json", flow_json_bytes, "application/json")},
            timeout=15,
        )
        self._raise_for_error(upload_resp, "upload the flow JSON")
        self.stdout.write(self.style.SUCCESS("Uploaded flow.json"))

        if options["publish"]:
            self.stdout.write("Publishing flow (irreversible)...")
            publish_resp = requests.post(f"{base}/{flow_id}/publish", headers=headers, timeout=15)
            self._raise_for_error(publish_resp, "publish the flow")
            self.stdout.write(self.style.SUCCESS("Published."))
        else:
            self.stdout.write(
                "Left as a draft. Open it in Meta Business Manager > WhatsApp Manager > "
                "Flows to preview it (use the 'Preview' health check, which pings your "
                "endpoint), then either publish there or re-run this command with --publish."
            )

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS(f"Set this in .env:  WHATSAPP_SERVICE_FLOW_ID={flow_id}"))

    def _raise_for_error(self, response, action_label):
        if response.ok:
            return
        try:
            detail = json.dumps(response.json(), indent=2)
        except ValueError:
            detail = response.text
        raise CommandError(f"Failed to {action_label} ({response.status_code}):\n{detail}")
