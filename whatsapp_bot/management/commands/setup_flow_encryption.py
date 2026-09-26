"""
One-time setup: generates the RSA keypair used to decrypt/encrypt WhatsApp
Flow data-exchange requests, saves the private key locally (encrypted with
a passphrase), and registers the public key with Meta for this phone number.

Run once from whatsapp_ai_bot/ (same folder as manage.py):

    python manage.py setup_flow_encryption

Re-running replaces the local key file and re-registers a fresh keypair —
only do that deliberately (e.g. suspected key compromise), since Meta only
keeps one active public key per phone number and old encrypted requests in
flight would fail to decrypt against a new key.
"""

import getpass
from pathlib import Path

import requests
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

GRAPH_API_VERSION = "v22.0"


class Command(BaseCommand):
    help = "Generates the Flow endpoint's RSA keypair and registers the public key with Meta."

    def add_arguments(self, parser):
        parser.add_argument(
            "--no-passphrase", action="store_true",
            help="Save the private key unencrypted (not recommended outside local dev). "
            "Without this flag you'll be prompted to set a passphrase, which must then "
            "also be set as WHATSAPP_FLOW_PRIVATE_KEY_PASSPHRASE in .env.",
        )

    def handle(self, *args, **options):
        if not settings.WHATSAPP_ACCESS_TOKEN:
            raise CommandError("WHATSAPP_ACCESS_TOKEN is not set in .env.")
        if not settings.WHATSAPP_PHONE_ID:
            raise CommandError("WHATSAPP_PHONE_ID is not set in .env.")

        key_path = Path(settings.WHATSAPP_FLOW_PRIVATE_KEY_PATH)
        if key_path.exists():
            confirm = input(
                f"A key already exists at {key_path} — regenerating replaces it and "
                f"re-registers a new public key with Meta. Continue? [y/N] "
            )
            if confirm.strip().lower() != "y":
                self.stdout.write("Aborted.")
                return

        passphrase = None
        if not options["no_passphrase"]:
            passphrase = getpass.getpass("Passphrase to encrypt the private key with: ")
            confirm_passphrase = getpass.getpass("Confirm passphrase: ")
            if passphrase != confirm_passphrase:
                raise CommandError("Passphrases didn't match.")
            if not passphrase:
                raise CommandError("Empty passphrase — pass --no-passphrase to explicitly skip one.")

        self.stdout.write("Generating a 2048-bit RSA keypair...")
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

        encryption = (
            serialization.BestAvailableEncryption(passphrase.encode())
            if passphrase else serialization.NoEncryption()
        )
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=encryption,
        )
        public_pem = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

        key_path.parent.mkdir(parents=True, exist_ok=True)
        key_path.write_bytes(private_pem)
        try:
            key_path.chmod(0o600)  # owner read/write only — best-effort, no-op on some filesystems
        except OSError:
            pass
        self.stdout.write(self.style.SUCCESS(f"Private key saved to {key_path}"))

        self.stdout.write("Registering public key with Meta...")
        headers = {"Authorization": f"Bearer {settings.WHATSAPP_ACCESS_TOKEN}"}
        resp = requests.post(
            f"https://graph.facebook.com/{GRAPH_API_VERSION}/{settings.WHATSAPP_PHONE_ID}/whatsapp_business_encryption",
            headers=headers,
            data={"business_public_key": public_pem.decode()},
            timeout=15,
        )
        if not resp.ok:
            raise CommandError(f"Failed to register public key ({resp.status_code}): {resp.text}")
        self.stdout.write(self.style.SUCCESS("Public key registered with Meta."))

        self.stdout.write("")
        if passphrase:
            self.stdout.write(self.style.WARNING(
                "Add this to .env (never commit it):\n"
                f"WHATSAPP_FLOW_PRIVATE_KEY_PASSPHRASE={passphrase}"
            ))
        if str(key_path) != settings.WHATSAPP_FLOW_PRIVATE_KEY_PATH:
            self.stdout.write(f"Also set WHATSAPP_FLOW_PRIVATE_KEY_PATH={key_path} in .env if it isn't already.")
        self.stdout.write("Restart the Celery worker and Django server after updating .env.")
