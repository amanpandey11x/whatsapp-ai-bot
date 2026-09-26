"""
Encryption/decryption for the WhatsApp Flow data-exchange endpoint, per
Meta's protocol: https://developers.facebook.com/docs/whatsapp/flows/guides/implementingyourflowendpoint

Every request Meta sends here is encrypted with a per-request AES key (itself
RSA-encrypted with our public key); every response must be encrypted back
with that same AES key, under a bit-flipped version of the request's IV.
Kept as a standalone module (no Django imports beyond settings) so the
crypto itself is easy to unit test without spinning up the full request
cycle.
"""

import base64
import json

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from django.conf import settings


class FlowDecryptionError(Exception):
    """Raised when a request can't be decrypted — the view responds 421 to
    this, which tells WhatsApp to re-fetch our public key and retry."""


_private_key_cache = None


def _load_private_key():
    """Reads and caches the RSA private key from
    settings.WHATSAPP_FLOW_PRIVATE_KEY_PATH (see management command
    setup_flow_encryption). Cached at module level since it's read from
    disk and doesn't change without a server restart."""
    global _private_key_cache
    if _private_key_cache is None:
        try:
            with open(settings.WHATSAPP_FLOW_PRIVATE_KEY_PATH, "rb") as f:
                pem_bytes = f.read()
        except OSError as exc:
            raise FlowDecryptionError(
                f"Couldn't read the Flow private key at {settings.WHATSAPP_FLOW_PRIVATE_KEY_PATH} "
                f"— run `python manage.py setup_flow_encryption` first."
            ) from exc
        passphrase = settings.WHATSAPP_FLOW_PRIVATE_KEY_PASSPHRASE
        _private_key_cache = serialization.load_pem_private_key(
            pem_bytes, password=passphrase.encode() if passphrase else None,
        )
    return _private_key_cache


def decrypt_request(body):
    """Decrypts one incoming Flow data-exchange request body (a dict with
    base64 encrypted_flow_data/encrypted_aes_key/initial_vector) into
    (decrypted_payload_dict, aes_key_bytes, iv_bytes). Raises
    FlowDecryptionError on any failure — malformed input, wrong key, a
    tampered/corrupted payload — so the view can respond 421 uniformly."""
    try:
        encrypted_flow_data = base64.b64decode(body["encrypted_flow_data"])
        encrypted_aes_key = base64.b64decode(body["encrypted_aes_key"])
        iv = base64.b64decode(body["initial_vector"])
    except (KeyError, ValueError, TypeError) as exc:
        raise FlowDecryptionError("Malformed request body") from exc

    try:
        aes_key = _load_private_key().decrypt(
            encrypted_aes_key,
            padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None),
        )
    except ValueError as exc:
        raise FlowDecryptionError("Couldn't decrypt the AES key") from exc

    try:
        # encrypted_flow_data is ciphertext with the 16-byte GCM tag
        # appended, exactly the format AESGCM.decrypt expects as `data`.
        decrypted = AESGCM(aes_key).decrypt(iv, encrypted_flow_data, None)
    except InvalidTag as exc:
        raise FlowDecryptionError("Couldn't decrypt the flow data") from exc

    try:
        payload = json.loads(decrypted)
    except ValueError as exc:
        raise FlowDecryptionError("Decrypted body wasn't valid JSON") from exc

    return payload, aes_key, iv


def encrypt_response(response_dict, aes_key, iv):
    """Encrypts one outgoing Flow data-exchange response — the IV is
    bit-flipped (XOR 0xFF every byte) per Meta's spec, never reused as-is
    from the request. Returns the base64 string to send as the raw
    text/plain response body."""
    flipped_iv = bytes(b ^ 0xFF for b in iv)
    plaintext = json.dumps(response_dict).encode()
    ciphertext = AESGCM(aes_key).encrypt(flipped_iv, plaintext, None)
    return base64.b64encode(ciphertext).decode()
