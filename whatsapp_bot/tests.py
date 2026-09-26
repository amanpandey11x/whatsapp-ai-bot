import base64
import json
import os
import tempfile
from datetime import timedelta
from pathlib import Path

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from django.test import Client, TestCase, override_settings
from django.utils import timezone
from unittest.mock import patch

from .models import Contact, Message, Order, ServiceCategory, ServicePricing
from .tasks import RESUME_NUDGE_AFTER_TURNS
from .services import (
    ADDON_LIST_DONE_ID,
    CALLBACK_SLOT_BUTTON_OPTIONS,
    CONFIRMATION_LIST_OPTIONS,
    CSAT_BUTTON_OPTIONS,
    build_quantity_reply,
    flattened_menu_fits,
    notify_human_alert,
    resolve_service_choice,
)


@override_settings(WHATSAPP_ADDON_FLOW_ID="", WHATSAPP_SERVICE_FLOW_ID="")
class ItrAddonListTests(TestCase):
    """Covers the tappable add-on list that lets a customer pick several of
    ITR Salary/Business/NRI Filing's optional income types (toggle add/remove
    per row), followed by a Confirm/Edit review screen — see
    tasks._enter_addon_list / tasks._addon_list_reply. Pinned to list mode
    (WHATSAPP_ADDON_FLOW_ID="") regardless of the local .env's real value —
    see ItrAddonFlowTests for the Flow-mode equivalents."""

    def setUp(self):
        self.category = ServiceCategory.objects.get(category_code="itr")
        self.salary_service = ServicePricing.objects.get(service_code="itr_salary")

    def test_itr_catalogue_has_all_requested_income_heads(self):
        names = list(
            self.salary_service.subservices.filter(is_active=True).values_list("display_name", flat=True)
        )
        self.assertEqual(
            names,
            [
                "Basic Salary", "Interest", "Indian Dividend", "Capital Gains",
                "House Property", "Foreign Dividend", "Other Income",
                "Sale of Property (Capital Gains)", "F&O", "Virtual Digital Asset",
                "Partnership",
            ],
        )

    def test_interactive_service_code_selects_the_exact_income_head(self):
        service = resolve_service_choice("itr_fo_income", category=self.category, parent_service=self.salary_service)
        self.assertEqual(service.display_name, "F&O")

    def test_quantity_pricing_uses_selected_property_count(self):
        service = ServicePricing.objects.get(service_code="itr_house_property")
        self.assertEqual(
            build_quantity_reply(service, 3),
            "For 3 properties of House Property, that's ₹99 per property — ₹297 total.",
        )

    @patch("whatsapp_bot.tasks.send_whatsapp_list")
    def test_selecting_a_wizard_parent_bundles_included_items_and_shows_the_addon_list(self, mock_send_list):
        from .tasks import process_incoming_message

        contact = Contact.objects.create(
            phone_number="918888888888",
            language="en",
            conversation_step=Contact.STEP_SUBSERVICE_SHOWN,
            pending_category_code="itr",
        )
        process_incoming_message.run(contact.id, "itr_business")

        contact.refresh_from_db()
        self.assertEqual(contact.conversation_step, Contact.STEP_AWAITING_INCOME_ITEM)
        self.assertEqual(contact.pending_parent_service_code, "itr_business")
        self.assertEqual(
            {item["service_code"] for item in contact.selected_services},
            {"itr_business_income", "itr_business_interest", "itr_business_indian_dividend"},
        )
        options = mock_send_list.call_args.args[3]
        self.assertEqual(options[-1][0], ADDON_LIST_DONE_ID)
        self.assertEqual(len(options), 9)  # 8 optional items + the Done row
        self.assertIn("Business Income", mock_send_list.call_args.args[1])

    @patch("whatsapp_bot.tasks.send_whatsapp_list")
    def test_business_selection_overrides_a_stale_salary_parent(self, mock_send_list):
        """A previous Salary flow must not hijack a Business Filing tap."""
        from .tasks import process_incoming_message

        contact = Contact.objects.create(
            phone_number="917777777777",
            language="en",
            conversation_step=Contact.STEP_SUBSERVICE_SHOWN,
            pending_category_code="itr",
            pending_parent_service_code="itr_salary",
        )
        process_incoming_message.run(contact.id, "itr_business")

        contact.refresh_from_db()
        self.assertEqual(contact.pending_parent_service_code, "itr_business")

    @patch("whatsapp_bot.tasks.send_whatsapp_list")
    def test_tapping_a_flat_priced_item_adds_it_then_tapping_again_removes_it(self, mock_send_list):
        from .tasks import process_incoming_message

        contact = Contact.objects.create(
            phone_number="916666666666",
            language="en",
            conversation_step=Contact.STEP_SUBSERVICE_SHOWN,
            pending_category_code="itr",
        )
        process_incoming_message.run(contact.id, "itr_salary")
        process_incoming_message.run(contact.id, "itr_capital_gains")

        contact.refresh_from_db()
        self.assertIn("itr_capital_gains", {item["service_code"] for item in contact.selected_services})
        options = {row[0]: row[2] for row in mock_send_list.call_args.args[3]}
        self.assertEqual(options["itr_capital_gains"], "Added ✓ — tap to remove")

        process_incoming_message.run(contact.id, "itr_capital_gains")  # tap again -> remove

        contact.refresh_from_db()
        self.assertNotIn("itr_capital_gains", {item["service_code"] for item in contact.selected_services})
        self.assertEqual(contact.conversation_step, Contact.STEP_AWAITING_INCOME_ITEM)

    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    @patch("whatsapp_bot.tasks.send_whatsapp_list")
    def test_quantity_item_detours_then_returns_to_the_addon_list(self, mock_send_list, mock_send_message):
        from .tasks import process_incoming_message

        contact = Contact.objects.create(
            phone_number="914444444444",
            language="en",
            conversation_step=Contact.STEP_SUBSERVICE_SHOWN,
            pending_category_code="itr",
        )
        process_incoming_message.run(contact.id, "itr_salary")
        process_incoming_message.run(contact.id, "itr_house_property")  # not yet selected -> asks quantity

        contact.refresh_from_db()
        self.assertEqual(contact.conversation_step, Contact.STEP_AWAITING_QUANTITY)
        self.assertEqual(contact.pending_service_code, "itr_house_property")
        self.assertEqual(contact.pending_parent_service_code, "itr_salary")

        process_incoming_message.run(contact.id, "2")

        contact.refresh_from_db()
        entries = {item["service_code"]: item for item in contact.selected_services}
        self.assertEqual(entries["itr_house_property"]["quantity"], 2)
        # Back in the add-on list, not free-chat and not the confirmation screen.
        self.assertEqual(contact.conversation_step, Contact.STEP_AWAITING_INCOME_ITEM)
        options = {row[0]: row[2] for row in mock_send_list.call_args.args[3]}
        self.assertEqual(options["itr_house_property"], "Added ✓ — tap to remove")

        # Tapping it again removes it directly (no re-asking the quantity).
        process_incoming_message.run(contact.id, "itr_house_property")
        contact.refresh_from_db()
        self.assertNotIn("itr_house_property", {item["service_code"] for item in contact.selected_services})

    @patch("whatsapp_bot.tasks.send_whatsapp_list")
    def test_confirm_and_done_moves_to_the_review_screen(self, mock_send_list):
        from .tasks import process_incoming_message

        contact = Contact.objects.create(
            phone_number="913333333333",
            language="en",
            conversation_step=Contact.STEP_SUBSERVICE_SHOWN,
            pending_category_code="itr",
        )
        process_incoming_message.run(contact.id, "itr_salary")
        process_incoming_message.run(contact.id, "itr_capital_gains")
        process_incoming_message.run(contact.id, "__addon_done__")

        contact.refresh_from_db()
        self.assertEqual(contact.conversation_step, Contact.STEP_CONFIRMING_SELECTION)
        self.assertEqual(contact.pending_parent_service_code, "itr_salary")  # kept, so Edit can find its way back
        # The review screen is a list too (Confirm / Edit more as rows), not
        # reply buttons — same call site (send_whatsapp_list) as the add-on
        # picker above it, so this is the last call recorded.
        self.assertEqual(mock_send_list.call_args.args[3], CONFIRMATION_LIST_OPTIONS)
        self.assertIn("Capital Gains", mock_send_list.call_args.args[1])

    @override_settings(RAZORPAY_KEY_ID="", RAZORPAY_KEY_SECRET="")
    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    @patch("whatsapp_bot.tasks.send_whatsapp_buttons")
    @patch("whatsapp_bot.tasks.send_whatsapp_list")
    def test_confirming_finalizes_and_sends_just_the_summary(self, mock_send_list, mock_send_buttons, mock_send_message):
        from .tasks import process_incoming_message

        contact = Contact.objects.create(
            phone_number="912222222222",
            language="en",
            conversation_step=Contact.STEP_SUBSERVICE_SHOWN,
            pending_category_code="itr",
        )
        process_incoming_message.run(contact.id, "itr_salary")
        process_incoming_message.run(contact.id, "itr_capital_gains")
        process_incoming_message.run(contact.id, "done")
        process_incoming_message.run(contact.id, "confirm")

        contact.refresh_from_db()
        self.assertEqual(contact.conversation_step, Contact.STEP_PRICING_SHOWN)
        self.assertEqual(contact.pending_parent_service_code, "")
        self.assertIsNone(contact.review_reached_at)  # cleared on confirm
        self.assertIn("Capital Gains", mock_send_message.call_args.args[1])
        # No callback/support ask fires here anymore — trigger_conversion_cta
        # only ever merges into the Pay Now CTA (see PaymentCtaGatingTests
        # and test_confirm_merges_the_summary_into_the_pay_now_cta) or, with
        # no payment link available (Razorpay unconfigured here), falls
        # through to the plain summary above and nothing else.
        mock_send_buttons.assert_not_called()

    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    @patch("whatsapp_bot.tasks.send_whatsapp_buttons")
    @patch("whatsapp_bot.tasks.send_whatsapp_list")
    def test_booking_a_callback_after_confirming_captures_the_slot_and_then_asks_csat(
        self, mock_send_list, mock_send_buttons, mock_send_message,
    ):
        from .tasks import process_incoming_message

        contact = Contact.objects.create(
            phone_number="912222222223",
            language="en",
            conversation_step=Contact.STEP_SUBSERVICE_SHOWN,
            pending_category_code="itr",
        )
        process_incoming_message.run(contact.id, "itr_salary")
        process_incoming_message.run(contact.id, "done")
        process_incoming_message.run(contact.id, "confirm")
        process_incoming_message.run(contact.id, "book_callback")
        self.assertEqual(mock_send_buttons.call_args.args[2], CALLBACK_SLOT_BUTTON_OPTIONS)

        process_incoming_message.run(contact.id, "slot_tomorrow")

        contact.refresh_from_db()
        self.assertEqual(contact.callback_slot, Contact.CALLBACK_SLOT_TOMORROW)
        self.assertIsNotNone(contact.callback_requested_at)
        self.assertTrue(contact.needs_human_review)
        self.assertEqual(contact.needs_human_review_reason, "callback_requested")
        self.assertEqual(mock_send_buttons.call_args.args[2], CSAT_BUTTON_OPTIONS)

    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    @patch("whatsapp_bot.tasks.send_whatsapp_buttons")
    @patch("whatsapp_bot.tasks.send_whatsapp_list")
    def test_keep_chatting_skips_the_callback_and_goes_straight_to_csat(
        self, mock_send_list, mock_send_buttons, mock_send_message,
    ):
        from .tasks import process_incoming_message

        contact = Contact.objects.create(
            phone_number="912222222224",
            language="en",
            conversation_step=Contact.STEP_SUBSERVICE_SHOWN,
            pending_category_code="itr",
        )
        process_incoming_message.run(contact.id, "itr_salary")
        process_incoming_message.run(contact.id, "done")
        process_incoming_message.run(contact.id, "confirm")
        process_incoming_message.run(contact.id, "keep_chatting")

        contact.refresh_from_db()
        self.assertEqual(contact.callback_slot, "")
        self.assertEqual(mock_send_buttons.call_args.args[2], CSAT_BUTTON_OPTIONS)

    @patch("whatsapp_bot.tasks.send_whatsapp_buttons")
    @patch("whatsapp_bot.tasks.send_whatsapp_list")
    def test_editing_returns_to_the_addon_list_with_current_selection_intact(self, mock_send_list, mock_send_buttons):
        from .tasks import process_incoming_message

        contact = Contact.objects.create(
            phone_number="911111111111",
            language="en",
            conversation_step=Contact.STEP_SUBSERVICE_SHOWN,
            pending_category_code="itr",
        )
        process_incoming_message.run(contact.id, "itr_salary")
        process_incoming_message.run(contact.id, "itr_capital_gains")
        process_incoming_message.run(contact.id, "done")
        process_incoming_message.run(contact.id, "edit")

        contact.refresh_from_db()
        self.assertEqual(contact.conversation_step, Contact.STEP_AWAITING_INCOME_ITEM)
        self.assertIn("itr_capital_gains", {item["service_code"] for item in contact.selected_services})
        options = {row[0]: row[2] for row in mock_send_list.call_args.args[3]}
        self.assertEqual(options["itr_capital_gains"], "Added ✓ — tap to remove")

    @patch("whatsapp_bot.tasks.send_whatsapp_list")
    def test_unrecognised_reply_re_shows_the_same_list_instead_of_free_chat(self, mock_send_list):
        from .tasks import process_incoming_message

        contact = Contact.objects.create(
            phone_number="910000000001",
            language="en",
            conversation_step=Contact.STEP_SUBSERVICE_SHOWN,
            pending_category_code="itr",
        )
        process_incoming_message.run(contact.id, "itr_salary")
        process_incoming_message.run(contact.id, "asdkfjasdkfj")

        contact.refresh_from_db()
        self.assertEqual(contact.conversation_step, Contact.STEP_AWAITING_INCOME_ITEM)
        self.assertIn("Didn't catch that", mock_send_list.call_args.args[1])


@override_settings(WHATSAPP_ADDON_FLOW_ID="test-flow-id", WHATSAPP_SERVICE_FLOW_ID="")
class ItrAddonFlowTests(TestCase):
    """Covers the two-screen WhatsApp Flow (ITR_TYPE -> ADDONS) that replaces
    the tap-list whenever WHATSAPP_ADDON_FLOW_ID is configured — either
    entry point (a specific type tap, or the ITR category itself) ends up
    applying the whole submission in one backend pass. See
    docs/whatsapp_flows/README.md."""

    def setUp(self):
        self.salary_service = ServicePricing.objects.get(service_code="itr_salary")

    @patch("whatsapp_bot.tasks.send_whatsapp_flow")
    def test_a_specific_type_tap_launches_straight_at_the_addons_screen(self, mock_send_flow):
        from .tasks import process_incoming_message

        contact = Contact.objects.create(
            phone_number="915555500001",
            language="en",
            conversation_step=Contact.STEP_SUBSERVICE_SHOWN,
            pending_category_code="itr",
        )
        process_incoming_message.run(contact.id, "itr_salary")

        mock_send_flow.assert_called_once()
        payload = mock_send_flow.call_args.args[3]
        self.assertEqual(mock_send_flow.call_args.kwargs["screen_id"], "ADDONS")
        self.assertEqual(payload["itr_type"], "itr_salary")
        options_by_id = {opt["id"]: opt for opt in payload["salary_addon_options"]}
        self.assertIn("itr_capital_gains", options_by_id)
        # Nothing on the cart yet, so no option should be flagged as added.
        self.assertNotIn("Already added", options_by_id["itr_capital_gains"]["description"])
        # Quantity items (House Property, Partnership) aren't checkboxes —
        # the ADDONS screen collects those via its own number fields.
        self.assertNotIn("itr_house_property", options_by_id)

    @patch("whatsapp_bot.tasks.send_whatsapp_flow")
    def test_the_itr_category_itself_launches_at_the_type_screen(self, mock_send_flow):
        """Tapping "ITR Filing" with no type named yet opens the Flow's
        first screen instead of the old sub-service list."""
        from .tasks import process_incoming_message

        contact = Contact.objects.create(
            phone_number="915555500002", language="en",
            conversation_step=Contact.STEP_CATEGORY_SHOWN,
        )
        process_incoming_message.run(contact.id, "itr")

        contact.refresh_from_db()
        self.assertEqual(contact.conversation_step, Contact.STEP_AWAITING_INCOME_ITEM)
        self.assertEqual(contact.pending_parent_service_code, "")
        mock_send_flow.assert_called_once()
        self.assertEqual(mock_send_flow.call_args.kwargs["screen_id"], "ITR_TYPE")
        payload = mock_send_flow.call_args.args[3]
        type_ids = {opt["id"] for opt in payload["itr_type_options"]}
        self.assertEqual(type_ids, {"itr_salary", "itr_business", "itr_nri"})
        # All three wizard parents' add-on sets are forwarded up front, so
        # the ADDONS screen can render whichever one gets picked with no
        # second server round trip.
        self.assertTrue(payload["business_addon_options"])
        self.assertTrue(payload["nri_addon_options"])

    @patch("whatsapp_bot.tasks.send_whatsapp_list")
    @patch("whatsapp_bot.tasks.send_whatsapp_flow")
    def test_flow_submission_applies_every_selection_in_one_pass(self, mock_send_flow, mock_send_list):
        from .tasks import process_incoming_message

        contact = Contact.objects.create(
            phone_number="915555500003",
            language="en",
            conversation_step=Contact.STEP_SUBSERVICE_SHOWN,
            pending_category_code="itr",
        )
        process_incoming_message.run(contact.id, "itr_salary")

        flow_reply = json.dumps({
            "itr_type": "itr_salary",
            "addons_salary": ["itr_capital_gains", "itr_other_income"],
            "house_property_qty": "",
            "partnership_qty": "",
        })
        process_incoming_message.run(contact.id, flow_reply)

        contact.refresh_from_db()
        self.assertEqual(
            {"itr_capital_gains", "itr_other_income"}
            & {item["service_code"] for item in contact.selected_services},
            {"itr_capital_gains", "itr_other_income"},
        )
        # The review is a list (Confirm / Edit more as rows), same as the
        # tap-list mode.
        self.assertEqual(contact.conversation_step, Contact.STEP_CONFIRMING_SELECTION)
        self.assertEqual(mock_send_list.call_args.args[3], CONFIRMATION_LIST_OPTIONS)

    @patch("whatsapp_bot.tasks.send_whatsapp_list")
    @patch("whatsapp_bot.tasks.send_whatsapp_flow")
    def test_flow_submission_applies_quantity_fields_with_no_further_round_trip(self, mock_send_flow, mock_send_list):
        """Unlike the tap-list, a quantity item ticked via the Flow needs no
        follow-up question — the ADDONS screen's dedicated number fields
        already collected it in the same submit."""
        from .tasks import process_incoming_message

        contact = Contact.objects.create(
            phone_number="915555500004",
            language="en",
            conversation_step=Contact.STEP_SUBSERVICE_SHOWN,
            pending_category_code="itr",
        )
        process_incoming_message.run(contact.id, "itr_salary")

        flow_reply = json.dumps({
            "itr_type": "itr_salary",
            "addons_salary": [],
            "house_property_qty": "3",
            "partnership_qty": "",
        })
        process_incoming_message.run(contact.id, flow_reply)

        contact.refresh_from_db()
        entries = {item["service_code"]: item for item in contact.selected_services}
        self.assertEqual(entries["itr_house_property"]["quantity"], 3)
        self.assertEqual(contact.conversation_step, Contact.STEP_CONFIRMING_SELECTION)

    @patch("whatsapp_bot.tasks.send_whatsapp_flow")
    def test_flow_submission_with_an_unknown_type_falls_through_to_free_chat(self, mock_send_flow):
        from .tasks import process_incoming_message

        contact = Contact.objects.create(
            phone_number="915555500005",
            language="en",
            conversation_step=Contact.STEP_AWAITING_INCOME_ITEM,
        )
        bad_reply = json.dumps({"itr_type": "not_a_real_type", "addons_salary": ["itr_capital_gains"]})
        process_incoming_message.run(contact.id, bad_reply)

        contact.refresh_from_db()
        self.assertEqual(contact.conversation_step, Contact.STEP_PRICING_SHOWN)
        self.assertEqual(contact.selected_services, [])

    @patch("whatsapp_bot.tasks.send_whatsapp_list")
    @patch("whatsapp_bot.tasks.send_whatsapp_flow")
    def test_flow_submission_only_reads_the_slot_matching_itr_type(self, mock_send_flow, mock_send_list):
        """The ADDONS screen's `If` branches mean only one of
        addons_salary/addons_business/addons_nri was ever actually rendered
        — even if another slot is present in the payload, it must be
        ignored rather than applied to the wrong parent's cart."""
        from .tasks import process_incoming_message

        contact = Contact.objects.create(
            phone_number="915555500006",
            language="en",
            conversation_step=Contact.STEP_AWAITING_INCOME_ITEM,
        )
        flow_reply = json.dumps({
            "itr_type": "itr_business",
            "addons_business": ["itr_business_capital_gains"],
            "addons_salary": ["itr_capital_gains"],  # should be ignored
            "house_property_qty": "",
            "partnership_qty": "",
        })
        process_incoming_message.run(contact.id, flow_reply)

        contact.refresh_from_db()
        codes = {item["service_code"] for item in contact.selected_services}
        self.assertIn("itr_business_capital_gains", codes)
        self.assertNotIn("itr_capital_gains", codes)


@override_settings(WHATSAPP_ADDON_FLOW_ID="", WHATSAPP_SERVICE_FLOW_ID="")
class FlattenedTopLevelMenuTests(TestCase):
    """Covers services.flattened_menu_fits — the very first menu shows every
    top-level service directly (skipping the category-then-subservice round
    trip) only while everything fits WhatsApp's 10-row list cap; once a
    catalogue grows past that it must fall back to the two-step category
    menu automatically rather than silently truncating anything. Pinned to
    list mode (WHATSAPP_ADDON_FLOW_ID="") — the ITR category tap behaves
    differently once the Flow is enabled, see ItrAddonFlowTests."""

    def setUp(self):
        # Scope this class to just ITR's top-level services — migration
        # 0020 also seeds GST's 8 top-level filings (and 3 more empty
        # categories), which alone pushes the real catalogue past the
        # 10-row cap. That's correct production behavior (falls back to the
        # two-step category menu automatically — see flattened_menu_fits'
        # docstring), but this class specifically covers the "small enough
        # to flatten" case, so it deactivates everything outside ITR rather
        # than assuming the whole catalogue happens to stay small.
        ServicePricing.objects.filter(parent_service__isnull=True).exclude(
            category__category_code="itr"
        ).update(is_active=False)

    def test_fits_with_only_itrs_top_level_services_active(self):
        # ITR's own top-level services (migrations 0012-0016) comfortably
        # fit under the 10-row cap on their own.
        self.assertTrue(flattened_menu_fits())

    def test_stops_fitting_once_the_catalogue_grows_past_ten(self):
        category = ServiceCategory.objects.get(category_code="itr")
        for i in range(10):
            ServicePricing.objects.create(
                category=category,
                service_code=f"extra_service_{i}",
                display_name=f"Extra Service {i}",
                price_display="₹99",
            )
        self.assertFalse(flattened_menu_fits())

    @patch("whatsapp_bot.tasks.send_whatsapp_list")
    def test_bare_category_code_still_falls_through_to_the_category_menu(self, mock_send_list):
        """A short generic word like "itr" is the category's own code — it
        must not get swallowed by a service-name substring match just
        because the flattened menu happens to fit (see
        services.resolve_flattened_service_choice)."""
        from .tasks import process_incoming_message

        contact = Contact.objects.create(
            phone_number="915555500005", language="en",
            conversation_step=Contact.STEP_CATEGORY_SHOWN,
        )
        process_incoming_message.run(contact.id, "itr")

        contact.refresh_from_db()
        self.assertEqual(contact.conversation_step, Contact.STEP_SUBSERVICE_SHOWN)
        self.assertEqual(contact.pending_category_code, "itr")

    @patch("whatsapp_bot.tasks.send_whatsapp_list")
    def test_flattened_menu_tap_on_a_wizard_parent_skips_the_category_step(self, mock_send_list):
        """Tapping a service_code row directly from the very first menu (see
        _top_level_menu_replies) goes straight to the add-on picker — no
        separate "which category?" reply needed."""
        from .tasks import process_incoming_message

        contact = Contact.objects.create(
            phone_number="915555500006", language="en",
            conversation_step=Contact.STEP_CATEGORY_SHOWN,
        )
        process_incoming_message.run(contact.id, "itr_salary")

        contact.refresh_from_db()
        self.assertEqual(contact.conversation_step, Contact.STEP_AWAITING_INCOME_ITEM)
        self.assertEqual(contact.pending_parent_service_code, "itr_salary")


@override_settings(WHATSAPP_SERVICE_FLOW_ID="")
class SatisfactionTests(TestCase):
    def setUp(self):
        self.contact = Contact.objects.create(
            phone_number="919000000000", language="en",
            conversation_step=Contact.STEP_PRICING_SHOWN,
        )

    @override_settings(RAZORPAY_KEY_ID="", RAZORPAY_KEY_SECRET="")
    @patch("whatsapp_bot.tasks.send_whatsapp_buttons")
    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    @patch("whatsapp_bot.tasks.send_whatsapp_list")
    def test_pricing_reply_is_followed_by_just_the_confirm_summary(self, mock_send_list, mock_send_message, mock_send_buttons):
        from .tasks import process_incoming_message

        self.contact.conversation_step = Contact.STEP_SUBSERVICE_SHOWN
        self.contact.pending_category_code = "itr"
        self.contact.save(update_fields=["conversation_step", "pending_category_code"])

        # A flat-priced quote now stops at SUMMARY (Confirm/Cancel list) —
        # see Contact.STEP_CONFIRMING_QUOTE — rather than firing the
        # payment link in the same turn as the quote.
        process_incoming_message.run(self.contact.id, "itr_futures_options")
        self.assertIn("Futures & Options ITR", mock_send_list.call_args.args[1])
        mock_send_buttons.assert_not_called()

        # Once the customer taps Confirm, with no payment link available
        # (Razorpay unconfigured here — see PaymentCtaGatingTests for that
        # merged path), the summary is just sent as a single plain message.
        # No callback/support ask fires here anymore — see
        # test_keep_chatting_then_gets_the_csat_prompt below for how CSAT
        # still gets triggered independently.
        process_incoming_message.run(self.contact.id, "confirm")

        self.assertIn("Futures & Options ITR", mock_send_message.call_args.args[1])
        mock_send_buttons.assert_not_called()

    @patch("whatsapp_bot.tasks.send_whatsapp_buttons")
    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    def test_keep_chatting_then_gets_the_csat_prompt(self, mock_send_message, mock_send_buttons):
        from .tasks import process_incoming_message

        self.contact.conversation_step = Contact.STEP_SUBSERVICE_SHOWN
        self.contact.pending_category_code = "itr"
        self.contact.save(update_fields=["conversation_step", "pending_category_code"])

        process_incoming_message.run(self.contact.id, "itr_futures_options")
        process_incoming_message.run(self.contact.id, "keep_chatting")

        self.assertEqual(mock_send_buttons.call_args.args[1], "Quick one — how was that?")
        self.assertEqual(mock_send_buttons.call_args.args[2], CSAT_BUTTON_OPTIONS)

    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    def test_tapping_a_csat_button_records_the_rating(self, mock_send_message):
        from .tasks import process_incoming_message

        process_incoming_message.run(self.contact.id, "great")

        self.contact.refresh_from_db()
        self.assertEqual(self.contact.satisfaction_rating, Contact.SATISFACTION_GREAT)
        self.assertIsNotNone(self.contact.satisfaction_rating_at)
        self.assertEqual(mock_send_message.call_args.args[1], "Thanks for letting us know! 🙌")

    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    def test_a_bad_rating_proactively_offers_help_and_flags_for_human_review(self, mock_send_message):
        from .tasks import process_incoming_message

        process_incoming_message.run(self.contact.id, "bad")

        self.assertIn("HELP", mock_send_message.call_args.args[1])
        self.contact.refresh_from_db()
        self.assertTrue(self.contact.needs_human_review)
        self.assertEqual(self.contact.needs_human_review_reason, "bad_csat")


def _make_test_keypair(tmp_dir, filename="test_flow_private_key.pem"):
    """Generates a throwaway RSA keypair and writes the private key (PEM,
    unencrypted) to tmp_dir — mirrors what `setup_flow_encryption` does,
    minus the passphrase and the call to Meta. Returns (private_key,
    public_key, key_path). `filename` must be unique per keypair within a
    test — reusing one would silently overwrite an earlier keypair's file
    on disk (bit us once already: see test_wrong_key_is_rejected_with_421,
    which generates a *second* keypair and must not clobber the server's)."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    key_path = Path(tmp_dir) / filename
    key_path.write_bytes(private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ))
    return private_key, private_key.public_key(), key_path


def _encrypt_like_meta(public_key, payload_dict):
    """Encrypts a payload the same way Meta's servers would when calling our
    Flow endpoint: a random AES-128 key (RSA-OAEP/SHA-256 encrypted with our
    public key) and a random 12-byte IV, AES-GCM encrypting the JSON body.
    Returns (request_body_dict, aes_key, iv) — the body is what the view
    receives; aes_key/iv are kept so the test can decrypt the view's
    response and confirm it round-trips correctly."""
    aes_key = AESGCM.generate_key(bit_length=128)
    iv = os.urandom(12)
    ciphertext = AESGCM(aes_key).encrypt(iv, json.dumps(payload_dict).encode(), None)
    encrypted_aes_key = public_key.encrypt(
        aes_key,
        padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None),
    )
    body = {
        "encrypted_flow_data": base64.b64encode(ciphertext).decode(),
        "encrypted_aes_key": base64.b64encode(encrypted_aes_key).decode(),
        "initial_vector": base64.b64encode(iv).decode(),
    }
    return body, aes_key, iv


def _decrypt_like_a_client(encrypted_body_text, aes_key, iv):
    """Decrypts a response the same way a real client would: the response
    is AES-GCM encrypted under the *flipped* IV (see flow_crypto.encrypt_response)."""
    flipped_iv = bytes(b ^ 0xFF for b in iv)
    ciphertext = base64.b64decode(encrypted_body_text)
    plaintext = AESGCM(aes_key).decrypt(flipped_iv, ciphertext, None)
    return json.loads(plaintext)


class FlowEndpointCryptoTests(TestCase):
    """End-to-end crypto round-trip for the live Flow data-exchange endpoint
    (whatsapp_bot.flow_crypto / views.whatsapp_flow_endpoint) — encrypts a
    request the same way Meta's servers do (see _encrypt_like_meta), posts
    it to the real URL, and decrypts the real response, without ever
    touching Meta's servers. This is what proves the encryption protocol
    itself is implemented correctly, independent of Flow screen logic."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._tmp_dir = tempfile.TemporaryDirectory()
        cls.private_key, cls.public_key, cls.key_path = _make_test_keypair(cls._tmp_dir.name)

    @classmethod
    def tearDownClass(cls):
        cls._tmp_dir.cleanup()
        super().tearDownClass()

    def _post_encrypted(self, payload):
        with override_settings(
            WHATSAPP_FLOW_PRIVATE_KEY_PATH=str(self.key_path),
            WHATSAPP_FLOW_PRIVATE_KEY_PASSPHRASE="",
        ):
            # Cached private key in flow_crypto must not leak across tests
            # using different throwaway keys.
            from . import flow_crypto
            flow_crypto._private_key_cache = None
            body, aes_key, iv = _encrypt_like_meta(self.public_key, payload)
            response = Client().post(
                "/api/whatsapp/flow/", data=json.dumps(body), content_type="application/json",
            )
            flow_crypto._private_key_cache = None
        return response, aes_key, iv

    def test_ping_health_check_round_trips_correctly(self):
        response, aes_key, iv = self._post_encrypted({"version": "3.0", "action": "ping"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "text/plain")
        decrypted = _decrypt_like_a_client(response.content.decode(), aes_key, iv)
        self.assertEqual(decrypted, {"data": {"status": "active"}})

    def test_a_non_ping_action_still_gets_a_valid_encrypted_response(self):
        """Unhandled actions (Phase 2 territory) must still respond with
        something the client can decrypt, never a plaintext/garbage body."""
        response, aes_key, iv = self._post_encrypted({
            "version": "3.0", "action": "data_exchange", "screen": "SOMETHING", "data": {},
        })

        self.assertEqual(response.status_code, 200)
        decrypted = _decrypt_like_a_client(response.content.decode(), aes_key, iv)
        self.assertIn("data", decrypted)

    def test_garbage_body_is_rejected_with_421_not_a_crash(self):
        response = Client().post(
            "/api/whatsapp/flow/",
            data=json.dumps({"encrypted_flow_data": "not-valid-base64!!", "encrypted_aes_key": "x", "initial_vector": "x"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 421)

    def test_wrong_key_is_rejected_with_421(self):
        """A request encrypted with a DIFFERENT keypair than the one the
        server has registered must fail cleanly, not decrypt into garbage."""
        _, other_public_key, _ = _make_test_keypair(self._tmp_dir.name, filename="other_key.pem")
        body, _, _ = _encrypt_like_meta(other_public_key, {"version": "3.0", "action": "ping"})

        with override_settings(WHATSAPP_FLOW_PRIVATE_KEY_PATH=str(self.key_path), WHATSAPP_FLOW_PRIVATE_KEY_PASSPHRASE=""):
            from . import flow_crypto
            flow_crypto._private_key_cache = None
            response = Client().post("/api/whatsapp/flow/", data=json.dumps(body), content_type="application/json")
            flow_crypto._private_key_cache = None
        self.assertEqual(response.status_code, 421)


class ServiceFlowScreenTests(TestCase):
    """Covers flow_screens.handle_flow_action — the mega Flow's screen
    routing (SERVICE -> ITR_TYPE -> ITR_ADDONS -> REVIEW and
    SERVICE -> GST_FILING -> REVIEW) — exercised directly on the plain
    decrypted payload dict, since encryption (flow_crypto) and routing are
    deliberately independent concerns."""

    def setUp(self):
        from . import services
        self.services = services
        self.contact = Contact.objects.create(phone_number="915555500010", language="en")
        self.token = services.make_flow_token(self.contact)

    def _call(self, screen, data, action="data_exchange"):
        from .flow_screens import handle_flow_action
        return handle_flow_action({
            "version": "3.0", "action": action, "screen": screen,
            "data": data, "flow_token": self.token,
        })

    def test_init_returns_service_screen(self):
        # SERVICE's category list ships as a literal data-source in
        # service_flow.json (not ${data.service_options}) so the screen
        # renders without depending on this INIT round trip resolving first
        # — see _service_screen_response. There is no LANGUAGE screen —
        # SERVICE is the Flow's entry screen, always (see
        # test_init_returns_service_screen_even_without_a_language below for
        # the no-language-yet case). show_first_time_question is the one
        # thing INIT's data does carry, since self.contact's
        # first_time_filer is still unset (see setUp).
        result = self._call(None, {}, action="INIT")
        self.assertEqual(result["screen"], "SERVICE")
        self.assertEqual(result["data"], {"show_first_time_question": True})

    def test_init_returns_service_screen_even_without_a_language(self):
        # A contact with no language set yet (e.g. somehow reaching the Flow
        # before the chat-based language gate ever ran) still gets SERVICE —
        # the Flow has no LANGUAGE screen to fall back to at all now that
        # language is always collected via chat first (see tasks.py's
        # unconditional language gate). _entry_screen_response just logs a
        # warning rather than erroring the whole open.
        from . import services
        new_contact = Contact.objects.create(phone_number="915555500011")
        token = services.make_flow_token(new_contact)
        from .flow_screens import handle_flow_action
        result = handle_flow_action({
            "version": "3.0", "action": "INIT", "screen": None,
            "data": {}, "flow_token": token,
        })
        self.assertEqual(result["screen"], "SERVICE")
        self.assertEqual(result["data"], {"show_first_time_question": True})

    def test_service_screen_persists_the_first_time_filer_answer(self):
        # The "First time filing with us?" question used to live on the
        # (now-removed) LANGUAGE screen — it's asked on SERVICE now instead
        # (see _handle_service_screen), conditionally shown only while
        # first_time_filer is still unset (see show_first_time_question).
        new_contact = Contact.objects.create(phone_number="915555500013", language="en")
        from . import services
        token = services.make_flow_token(new_contact)
        from .flow_screens import handle_flow_action
        result = handle_flow_action({
            "version": "3.0", "action": "data_exchange", "screen": "SERVICE",
            "data": {"selected_service": "itr", "first_time": "yes"}, "flow_token": token,
        })
        self.assertEqual(result["screen"], "ITR_TYPE")
        new_contact.refresh_from_db()
        self.assertIs(new_contact.first_time_filer, True)

    def test_service_screen_stops_asking_first_time_once_answered(self):
        contact = Contact.objects.create(
            phone_number="915555500014", language="en", first_time_filer=True,
        )
        from . import services
        token = services.make_flow_token(contact)
        from .flow_screens import handle_flow_action
        result = handle_flow_action({
            "version": "3.0", "action": "INIT", "screen": None,
            "data": {}, "flow_token": token,
        })
        self.assertEqual(result["data"], {"show_first_time_question": False})

    def test_unknown_flow_token_returns_error_screen_not_a_crash(self):
        from .flow_screens import handle_flow_action
        result = handle_flow_action({
            "version": "3.0", "action": "INIT", "screen": None,
            "data": {}, "flow_token": "not-a-real-token",
        })
        self.assertEqual(result["screen"], "ERROR")

    def test_selecting_itr_routes_to_itr_type_with_live_options(self):
        result = self._call("SERVICE", {"selected_service": "itr"})
        self.assertEqual(result["screen"], "ITR_TYPE")
        codes = {row["id"] for row in result["data"]["itr_type_options"]}
        self.assertIn("itr_salary", codes)
        self.assertIn("itr_business", codes)
        self.assertIn("itr_nri", codes)

    def test_selecting_an_unknown_category_code_routes_to_coming_soon(self):
        """All 5 real categories have a live catalogue now — this only
        exercises the safety net for a stale/bogus category_code (e.g. a
        client on an older cached Flow definition)."""
        result = self._call("SERVICE", {"selected_service": "legal_advice"})
        self.assertEqual(result["screen"], "COMING_SOON")

    def test_selecting_business_reg_routes_with_live_options(self):
        result = self._call("SERVICE", {"selected_service": "business_reg"})
        self.assertEqual(result["screen"], "BUSINESS_REG")
        ids = {row["id"] for row in result["data"]["business_reg_options"]}
        self.assertIn("bizreg_pvt_ltd_registration", ids)

    def test_business_reg_multiselect_saves_selection_and_totals(self):
        result = self._call("BUSINESS_REG", {
            "business_reg_items": ["bizreg_pan_application", "bizreg_msme_registration"],
        })
        self.assertEqual(result["screen"], "REVIEW")
        self.assertEqual(result["data"]["total_display"], "₹1,498")  # ₹499 + ₹999

        self.contact.refresh_from_db()
        codes = {item["service_code"] for item in self.contact.selected_services}
        self.assertEqual(codes, {"bizreg_pan_application", "bizreg_msme_registration"})

    def test_selecting_notices_routes_with_live_options(self):
        result = self._call("SERVICE", {"selected_service": "notices"})
        self.assertEqual(result["screen"], "NOTICES")
        ids = {row["id"] for row in result["data"]["notice_options"]}
        self.assertIn("notice_gst_tribunal", ids)

    def test_notices_multiselect_saves_selection(self):
        result = self._call("NOTICES", {"notice_items": ["notice_itr_appeal"]})
        self.assertEqual(result["screen"], "REVIEW")
        self.assertEqual(result["data"]["total_display"], "₹49,999")

    def test_selecting_book_expert_routes_with_live_options(self):
        result = self._call("SERVICE", {"selected_service": "book_expert"})
        self.assertEqual(result["screen"], "BOOK_EXPERT")
        ids = {row["id"] for row in result["data"]["consult_options"]}
        self.assertEqual(ids, {"consult_tax", "consult_compliance", "consult_business_structuring", "consult_general"})

    def test_book_expert_multiselect_saves_multiple_topics(self):
        result = self._call("BOOK_EXPERT", {"consult_items": ["consult_tax", "consult_compliance"]})
        self.assertEqual(result["screen"], "REVIEW")
        self.assertEqual(result["data"]["total_display"], "₹1,998")  # ₹999 x 2

    def test_itr_salary_type_routes_to_addons_with_bundled_children_included(self):
        result = self._call("ITR_TYPE", {"itr_type": "itr_salary"})
        self.assertEqual(result["screen"], "ITR_ADDONS")
        addon_ids = {row["id"] for row in result["data"]["addon_options"]}
        self.assertIn("itr_capital_gains", addon_ids)
        self.assertTrue(result["data"]["show_house_property"])
        self.assertTrue(result["data"]["show_partnership"])

        self.contact.refresh_from_db()
        codes = {item["service_code"] for item in self.contact.selected_services}
        self.assertIn("itr_basic_salary", codes)  # always-bundled, never asked about

    def test_flat_priced_itr_type_skips_addons_straight_to_review(self):
        result = self._call("ITR_TYPE", {"itr_type": "itr_futures_options"})
        self.assertEqual(result["screen"], "REVIEW")
        self.assertIn("₹", result["data"]["total_display"])
        # Landing on REVIEW starts the abandoned-cart nudge clock — see
        # flow_screens.handle_flow_action / services.mark_review_reached.
        self.contact.refresh_from_db()
        self.assertIsNotNone(self.contact.review_reached_at)

    def test_itr_addons_submit_computes_total_and_saves_quantities(self):
        self._call("ITR_TYPE", {"itr_type": "itr_salary"})
        result = self._call("ITR_ADDONS", {
            "itr_type": "itr_salary",
            "addons": ["itr_capital_gains"],
            "house_property_qty": "2",
            "partnership_qty": "",
        })
        self.assertEqual(result["screen"], "REVIEW")
        self.assertIn("Capital Gains", result["data"]["summary_text"])
        self.assertIn("House Property", result["data"]["summary_text"])
        # ₹999 (capital gains) + ₹99 x 2 (house property) = ₹1,197
        self.assertEqual(result["data"]["total_display"], "₹1,197")

    def test_switching_itr_type_drops_the_previous_types_addons(self):
        self._call("ITR_TYPE", {"itr_type": "itr_salary"})
        self._call("ITR_ADDONS", {"itr_type": "itr_salary", "addons": ["itr_capital_gains"]})
        self._call("ITR_TYPE", {"itr_type": "itr_business"})

        self.contact.refresh_from_db()
        codes = {item["service_code"] for item in self.contact.selected_services}
        self.assertNotIn("itr_capital_gains", codes)
        # Wizard parents are never priced line items themselves (see
        # _handle_itr_type_screen) — the always-bundled child is the signal
        # that itr_business is now the active type.
        self.assertIn("itr_business_income", codes)

    def test_gst_multiselect_saves_every_ticked_filing_and_totals_them(self):
        result = self._call("GST_FILING", {"gst_filings": ["gst_nil_return", "gst_9_annual"]})
        self.assertEqual(result["screen"], "REVIEW")
        self.assertIn("₹6,999", result["data"]["total_display"])  # nil is ₹0, 9-annual is ₹6,999

        self.contact.refresh_from_db()
        codes = {item["service_code"] for item in self.contact.selected_services}
        self.assertEqual(codes, {"gst_nil_return", "gst_9_annual"})

    def test_resubmitting_gst_with_fewer_items_drops_the_unticked_one(self):
        self._call("GST_FILING", {"gst_filings": ["gst_nil_return", "gst_9_annual"]})
        self._call("GST_FILING", {"gst_filings": ["gst_nil_return"]})

        self.contact.refresh_from_db()
        codes = {item["service_code"] for item in self.contact.selected_services}
        self.assertEqual(codes, {"gst_nil_return"})

    def test_completion_reply_is_parsed_and_does_not_get_stuck(self):
        from .services import build_service_flow_completion_reply, parse_service_flow_reply

        reply = json.dumps({"summary_text": "• GSTR-9 — ₹6,999", "total_display": "₹6,999"})
        parsed = parse_service_flow_reply(reply)
        self.assertEqual(parsed["kind"], "review")
        text = build_service_flow_completion_reply(parsed, self.contact)
        self.assertIn("₹6,999", text)
        # The reference code is generated and logged for internal tracking
        # (see build_service_flow_reference) but deliberately kept out of
        # the customer-facing text — no reason for a customer to see it.
        self.assertNotIn(f"TP-{self.contact.id}-", text)


class FlowIntentLeadScoreTests(TestCase):
    """Covers services.record_flow_progress/SCREEN_INTENT_WEIGHTS and their
    effect on compute_lead_score — the per-screen "user intent" signal, as
    opposed to the old single reached-pricing-or-not flag."""

    def setUp(self):
        from . import services
        self.services = services
        self.contact = Contact.objects.create(phone_number="915555500030", language="en")
        self.token = services.make_flow_token(self.contact)

    def _call(self, screen, data, action="data_exchange"):
        from .flow_screens import handle_flow_action
        return handle_flow_action({
            "version": "3.0", "action": action, "screen": screen,
            "data": data, "flow_token": self.token,
        })

    def test_progress_is_monotonic_does_not_regress(self):
        self.services.record_flow_progress(self.contact, "ITR_ADDONS")
        self.services.record_flow_progress(self.contact, "SERVICE")  # shallower — must not overwrite
        self.contact.refresh_from_db()
        self.assertEqual(self.contact.flow_furthest_screen, "ITR_ADDONS")

    def test_error_screen_does_not_record_or_erase_progress(self):
        self.services.record_flow_progress(self.contact, "ITR_TYPE")
        self.services.record_flow_progress(self.contact, "ERROR")
        self.contact.refresh_from_db()
        self.assertEqual(self.contact.flow_furthest_screen, "ITR_TYPE")  # untouched, not "ERROR"

    def test_deeper_screen_scores_higher(self):
        service_only = Contact.objects.create(phone_number="915555500031", flow_furthest_screen="SERVICE")
        addons = Contact.objects.create(phone_number="915555500032", flow_furthest_screen="ITR_ADDONS")
        self.assertLess(self.services.compute_lead_score(service_only), self.services.compute_lead_score(addons))

    def test_review_with_real_cart_scores_higher_than_empty_review(self):
        empty_review = Contact.objects.create(phone_number="915555500033", flow_furthest_screen="REVIEW")
        priced_review = Contact.objects.create(
            phone_number="915555500034", flow_furthest_screen="REVIEW",
            selected_services=[{"service_code": "gst_9_annual", "quantity": 1}],  # ₹6,999
        )
        self.assertLess(
            self.services.compute_lead_score(empty_review),
            self.services.compute_lead_score(priced_review),
        )

    def test_legacy_wizard_contact_without_flow_furthest_screen_still_uses_old_flag(self):
        legacy = Contact.objects.create(
            phone_number="915555500035", conversation_step=Contact.STEP_PRICING_SHOWN,
        )
        self.assertGreaterEqual(self.services.compute_lead_score(legacy), 35)

    def test_tier_flip_to_hot_fires_a_human_alert_exactly_once(self):
        hot = Contact.objects.create(
            phone_number="915555500036", flow_furthest_screen="REVIEW",
            selected_services=[{"service_code": "gst_9_annual", "quantity": 1}],
            has_requested_help=True,
        )
        self.assertEqual(hot.lead_tier, Contact.LEAD_TIER_COLD)  # default, before update_lead_score runs

        self.services.update_lead_score(hot)
        hot.refresh_from_db()
        self.assertEqual(hot.lead_tier, Contact.LEAD_TIER_HOT)
        self.assertTrue(hot.needs_human_review)
        self.assertEqual(hot.needs_human_review_reason, "hot_lead")

        # Clear the flag and recompute again with no change in signals — must
        # NOT re-fire the alert every time an already-hot contact is touched.
        hot.needs_human_review = False
        hot.save(update_fields=["needs_human_review"])
        self.services.update_lead_score(hot)
        hot.refresh_from_db()
        self.assertFalse(hot.needs_human_review)

    def test_reaching_a_screen_via_the_real_endpoint_updates_lead_score_live(self):
        """Not just record_flow_progress in isolation — the actual
        handle_flow_action call path (see flow_screens.py) must refresh
        lead_score/lead_tier without waiting for a later chat message,
        since an abandoned mid-Flow contact may never send one."""
        self.assertEqual(self.contact.lead_score, 0)
        self._call("SERVICE", {"selected_service": "itr"}, action="data_exchange")
        self.contact.refresh_from_db()
        self.assertGreater(self.contact.lead_score, 0)
        self.assertEqual(self.contact.flow_furthest_screen, "ITR_TYPE")


@override_settings(WHATSAPP_SERVICE_FLOW_ID="test-service-flow-id", WHATSAPP_ACCESS_TOKEN="test-token")
class ServiceFlowCtaLabelTests(TestCase):
    """Covers services.send_service_flow's flow_cta — a deep-linked launch
    (natural-language intent confirmed, see FLOW_TYPE_CTA_LABELS) must show
    the actual service being opened, not the generic "Get started" every
    plain reopen shows — see send_service_flow's docstring for why this is
    the one user-visible signal that can catch a misclassified intent
    before the Flow itself opens."""

    def setUp(self):
        self.contact = Contact.objects.create(phone_number="915555599960", language="en")

    def _sent_flow_cta(self, mock_post):
        payload = mock_post.call_args.kwargs["json"]
        return payload["interactive"]["action"]["parameters"]["flow_cta"]

    @patch("whatsapp_bot.services.requests.post")
    def test_deep_linked_launch_shows_the_specific_service_name(self, mock_post):
        from .services import send_service_flow

        send_service_flow(self.contact, deep_link_screen="GST_FILING")

        self.assertEqual(self._sent_flow_cta(mock_post), "Start GST Filing")

    @patch("whatsapp_bot.services.requests.post")
    def test_plain_reopen_with_no_deep_link_shows_the_generic_label(self, mock_post):
        from .services import send_service_flow

        send_service_flow(self.contact)

        self.assertEqual(self._sent_flow_cta(mock_post), "Get started")

    @patch("whatsapp_bot.services.requests.post")
    def test_deep_link_is_ignored_for_a_contact_with_no_language_set_yet(self, mock_post):
        """Purely a CTA-label cosmetics guard (see send_service_flow's
        docstring) — contact.language should always be set by the time this
        runs in practice (tasks.py's chat-based language gate), but a
        language-less contact still must not see a specific "Start GST
        Filing"-style label on the button."""
        from .services import send_service_flow

        new_contact = Contact.objects.create(phone_number="915555599961")
        send_service_flow(new_contact, deep_link_screen="GST_FILING")

        self.assertEqual(self._sent_flow_cta(mock_post), "Get started")

    @patch("whatsapp_bot.services.requests.post")
    def test_every_flow_type_cta_label_fits_whatsapps_20_char_cap(self, mock_post):
        from .services import FLOW_TYPE_CHOICES, send_service_flow

        for flow_type in FLOW_TYPE_CHOICES:
            mock_post.reset_mock()
            send_service_flow(self.contact, deep_link_screen=flow_type)
            cta = self._sent_flow_cta(mock_post)
            self.assertLessEqual(len(cta), 20, f"{flow_type}'s CTA label {cta!r} exceeds WhatsApp's 20-char cap")


@override_settings(WHATSAPP_SERVICE_FLOW_ID="test-service-flow-id")
class FullCutoverTests(TestCase):
    """Covers the full-cutover wiring in tasks.py — once
    WHATSAPP_SERVICE_FLOW_ID is set, a contact with no language yet always
    sees the chat-native language list first (see tasks.py's language
    gate, unconditional regardless of cutover), and the mega Flow opens
    right after language is picked — for a brand new contact, one mid a
    legacy step from before the cutover, or one just RESTART/menu'd.
    Must NOT intercept HELP/CSAT or a contact who's already finished
    (STEP_PRICING_SHOWN), and — the one-shot behavior — must NOT keep
    re-intercepting every later message from a contact who ignores the Flow
    and just asks a real question instead (see
    test_ignoring_the_flow_and_asking_a_real_question_gets_a_real_reply)."""

    @patch("whatsapp_bot.tasks.send_service_flow")
    def test_brand_new_contact_gets_the_language_list_first_not_the_flow(self, mock_send_flow):
        """The Flow no longer opens on message 1 — language is asked via a
        plain chat list first (see tasks.py's language gate), regardless of
        full cutover, so a real customer's language is known even if they
        never touch the Flow itself (the bug this gate was added to close —
        see services.send_service_flow's docstring)."""
        from .tasks import process_incoming_message

        contact = Contact.objects.create(phone_number="915555500020")
        process_incoming_message.run(contact.id, "hi")

        mock_send_flow.assert_not_called()
        contact.refresh_from_db()
        self.assertEqual(contact.language, "")
        self.assertEqual(contact.conversation_step, "")  # not yet welcomed

    @patch("whatsapp_bot.tasks.send_service_flow")
    def test_picking_a_language_then_opens_the_flow(self, mock_send_flow):
        from .tasks import process_incoming_message

        contact = Contact.objects.create(phone_number="915555500020")
        process_incoming_message.run(contact.id, "hi")
        mock_send_flow.assert_not_called()

        process_incoming_message.run(contact.id, "English")

        mock_send_flow.assert_called_once()
        contact.refresh_from_db()
        self.assertEqual(contact.language, "en")
        self.assertEqual(contact.conversation_step, Contact.STEP_WELCOMED)  # stamped so this doesn't repeat

    @patch("whatsapp_bot.tasks.send_service_flow")
    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    def test_welcome_line_does_not_repeat_the_language_question_already_answered(
        self, mock_send_message, mock_send_flow,
    ):
        """The plain-text welcome line sent ahead of the Flow trigger (see
        tasks._send_welcome_then_flow) must not ask the customer to reply 1
        or 2 for a language — that's build_welcome_message's legacy
        chat-wizard text, and by this point (language just answered via the
        chat list one turn ago) it's already been asked and answered, so
        repeating it as inert plain text would just be confusing."""
        from .tasks import process_incoming_message

        contact = Contact.objects.create(phone_number="915555500023")
        process_incoming_message.run(contact.id, "hi")
        process_incoming_message.run(contact.id, "English")

        welcome_body = mock_send_message.call_args.args[1]
        self.assertNotIn("1", welcome_body)
        self.assertNotIn("Reply with", welcome_body)
        self.assertNotIn("choose your preferred language", welcome_body)
        self.assertIn("Welcome to TaxPlan Advisors", welcome_body)

    @patch(
        "whatsapp_bot.tasks.classify_and_respond",
        return_value={"reply": "Sure, happy to help with that!", "action": "continue_chat", "flow_type": None, "confidence": "low"},
    )
    @patch("whatsapp_bot.tasks.send_service_flow")
    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    def test_ignoring_the_flow_and_asking_a_real_question_gets_a_real_reply(
        self, mock_send_message, mock_send_flow, mock_ai,
    ):
        """The actual bug this class exists to prevent: a contact who never
        completes the Flow must still be able to just... talk to the bot.
        The Flow only auto-opens once, right after language is picked; a
        later message that isn't MENU/RESTART/HELP must fall through to
        free-chat, not get the Flow re-shoved at them again."""
        from .tasks import process_incoming_message

        contact = Contact.objects.create(phone_number="915555500025")
        process_incoming_message.run(contact.id, "hi")
        process_incoming_message.run(contact.id, "English")
        mock_send_flow.assert_called_once()
        # The language-pick turn also gets the personalized welcome line as
        # its own send, ahead of the Flow trigger — see
        # tasks._send_welcome_then_flow.
        mock_send_message.assert_called_once()

        process_incoming_message.run(contact.id, "what are your business hours?")

        mock_send_flow.assert_called_once()  # still just the one call, not a second
        mock_ai.assert_called_once()
        mock_send_message.assert_called_with("915555500025", "Sure, happy to help with that!")

    @patch(
        "whatsapp_bot.tasks.classify_and_respond",
        return_value={
            "reply": "The pricing for GST varies by case — reply HELP to connect with our team.",
            "action": "continue_chat", "flow_type": None, "confidence": "high",
        },
    )
    @patch("whatsapp_bot.tasks.send_service_flow")
    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    def test_mid_flow_cart_session_asking_a_real_question_gets_a_real_reply_not_the_welcome(
        self, mock_send_message, mock_send_flow, mock_ai,
    ):
        """Regression test for the 2026-09-11 trace: a contact actively mid
        the mega Flow's cart-confirmation steps (STEP_CONFIRMING_SELECTION,
        STEP_CONFIRMING_QUOTE — neither of them legacy, both still in active
        use post-cutover) is NOT a "brand new / stuck-in-a-dead-step"
        contact, so a real question from them ("I have 3 house property,
        want to know the pricing of gst") must reach free-chat like any
        other message, not get discarded for a brand-new "Welcome to
        TaxPlan Advisors!" + re-opened Flow — that was the bug: the old
        blocklist condition (conversation_step not in (STEP_WELCOMED,
        STEP_PRICING_SHOWN)) swept up every in-progress mega-Flow step too,
        not just the legacy ones it was meant for. STEP_AWAITING_INCOME_ITEM
        gets its own dedicated test below — it never reaches free-chat for
        unmatched text even correctly configured, so it doesn't fit this
        same "must call classify_and_respond" shape."""
        from .tasks import process_incoming_message

        for step in (Contact.STEP_CONFIRMING_SELECTION, Contact.STEP_CONFIRMING_QUOTE):
            mock_send_flow.reset_mock()
            mock_send_message.reset_mock()
            mock_ai.reset_mock()
            extra = {}
            if step == Contact.STEP_CONFIRMING_SELECTION:
                # Needs a resolvable wizard parent, or this step takes its
                # own separate "item no longer available — finalize now
                # regardless of what was said" path instead of deferring to
                # free-chat (see its "if not parent_service" branch).
                extra["pending_parent_service_code"] = "itr_salary"
            contact = Contact.objects.create(
                phone_number=f"91555550{hash(step) % 10000:04d}", language="en",
                conversation_step=step, selected_services=[{"service_code": "itr_basic_salary", "quantity": 1}],
                **extra,
            )

            process_incoming_message.run(contact.id, "I have 3 house property, want to know pricing of gst")

            mock_send_flow.assert_not_called()
            mock_ai.assert_called_once()
            mock_send_message.assert_called_once_with(
                contact.phone_number, "The pricing for GST varies by case — reply HELP to connect with our team.",
            )
            contact.refresh_from_db()
            self.assertEqual(contact.conversation_step, step, f"{step} must be left untouched")

    @patch("whatsapp_bot.tasks.send_service_flow")
    @patch("whatsapp_bot.tasks.send_whatsapp_flow")
    def test_mid_addon_picker_asking_a_real_question_gets_the_picker_not_the_welcome(
        self, mock_send_addon_flow, mock_send_service_flow,
    ):
        """Same regression as above, for STEP_AWAITING_INCOME_ITEM
        specifically — it doesn't defer to free-chat for unmatched text
        (see _addon_screen_replies' own "Didn't catch that" reprompt, sent
        via the static add-on Flow here since WHATSAPP_ADDON_FLOW_ID is set
        — see services.addon_flow_enabled), so the bug-relevant assertion
        here is narrower: it must never trigger FULL CUTOVER's brand-new
        welcome + mega-Flow re-open either, same as the confirmation steps
        above (a distinct send call, send_service_flow, from the add-on
        Flow's own send_whatsapp_flow — asserted separately so a regression
        back to the old behavior can't hide behind the wrong mock)."""
        from .tasks import process_incoming_message

        contact = Contact.objects.create(
            phone_number="915555500027", language="en",
            conversation_step=Contact.STEP_AWAITING_INCOME_ITEM,
            pending_parent_service_code="itr_salary",
            selected_services=[{"service_code": "itr_basic_salary", "quantity": 1}],
        )

        # Deliberately not the "3 house property" phrasing used elsewhere in
        # this class — it fuzzy-matches the House Property add-on item
        # itself (a real, separate, correct behavior: it asks a quantity
        # follow-up), which isn't what this test is checking. A phrase with
        # no catalog-item match at all is what actually exercises this
        # step's "Didn't catch that" fallback.
        process_incoming_message.run(contact.id, "what's the weather like today")

        mock_send_service_flow.assert_not_called()
        mock_send_addon_flow.assert_called_once()
        body_text = mock_send_addon_flow.call_args.args[1]
        self.assertIn("Didn't catch that", body_text)
        contact.refresh_from_db()
        self.assertEqual(contact.conversation_step, Contact.STEP_AWAITING_INCOME_ITEM)

    @patch("whatsapp_bot.tasks.send_service_flow")
    def test_contact_stuck_mid_legacy_step_from_before_cutover_gets_the_flow(self, mock_send_flow):
        """A contact who was mid old-wizard (e.g. STEP_CATEGORY_SHOWN) before
        WHATSAPP_SERVICE_FLOW_ID was ever set must not stay stuck in that
        dead branch once it is — full cutover means every non-finished
        contact gets the Flow, not just brand new ones."""
        from .tasks import process_incoming_message

        contact = Contact.objects.create(
            phone_number="915555500021", language="en",
            conversation_step=Contact.STEP_CATEGORY_SHOWN,
        )
        process_incoming_message.run(contact.id, "itr")

        mock_send_flow.assert_called_once()

    @patch("whatsapp_bot.tasks.send_service_flow")
    def test_help_keyword_still_works_and_does_not_open_the_flow(self, mock_send_flow):
        from .tasks import process_incoming_message

        contact = Contact.objects.create(phone_number="915555500022")
        process_incoming_message.run(contact.id, "HELP")

        mock_send_flow.assert_not_called()
        contact.refresh_from_db()
        self.assertTrue(contact.has_requested_help)

    @patch("whatsapp_bot.tasks.send_service_flow")
    def test_restart_keyword_shows_the_language_list_not_the_flow_directly(self, mock_send_flow):
        """RESTART clears contact.language, same as always — but no longer
        reopens the Flow itself on this same turn (that used to skip
        straight past language, the exact bug the language gate exists to
        close — see tasks.py). It shows the same chat-native language list
        a brand-new contact gets; the Flow opens only once that's answered
        (test_restart_then_picking_a_language_opens_the_flow below)."""
        from .tasks import process_incoming_message

        contact = Contact.objects.create(
            phone_number="915555500023", language="en",
            conversation_step=Contact.STEP_PRICING_SHOWN,
        )
        process_incoming_message.run(contact.id, "RESTART")

        mock_send_flow.assert_not_called()
        contact.refresh_from_db()
        self.assertEqual(contact.language, "")  # RESTART re-asks language too

    @patch("whatsapp_bot.tasks.send_service_flow")
    def test_restart_then_picking_a_language_opens_the_flow(self, mock_send_flow):
        from .tasks import process_incoming_message

        contact = Contact.objects.create(
            phone_number="915555500024", language="en",
            conversation_step=Contact.STEP_PRICING_SHOWN,
        )
        process_incoming_message.run(contact.id, "RESTART")
        mock_send_flow.assert_not_called()

        process_incoming_message.run(contact.id, "English")

        mock_send_flow.assert_called_once()
        contact.refresh_from_db()
        self.assertEqual(contact.language, "en")
        self.assertEqual(contact.conversation_step, Contact.STEP_WELCOMED)

    @patch("whatsapp_bot.tasks.send_service_flow")
    def test_menu_keyword_reopens_the_flow_without_resetting_the_contact(self, mock_send_flow):
        """Unlike RESTART, "menu" is meant as a lightweight reopen — see
        services.MENU_KEYWORD — so language/lead_score/satisfaction rating
        must all survive it."""
        from .tasks import process_incoming_message

        contact = Contact.objects.create(
            phone_number="915555500026", language="en",
            conversation_step=Contact.STEP_PRICING_SHOWN,
            satisfaction_rating=Contact.SATISFACTION_GREAT,
        )
        process_incoming_message.run(contact.id, "menu")

        mock_send_flow.assert_called_once()
        contact.refresh_from_db()
        # Unlike RESTART (which explicitly zeroes/clears these — see
        # tasks.py), "menu" never touches language or satisfaction_rating at
        # all. lead_score isn't checked here: it's recomputed deterministically
        # on every message regardless of branch (see compute_lead_score),
        # so preservation isn't the right thing to assert about it.
        self.assertEqual(contact.language, "en")
        self.assertEqual(contact.satisfaction_rating, Contact.SATISFACTION_GREAT)

    @patch(
        "whatsapp_bot.tasks.classify_and_respond",
        return_value={"reply": "Sure, happy to help with that!", "action": "continue_chat", "flow_type": None, "confidence": "low"},
    )
    @patch("whatsapp_bot.tasks.send_service_flow")
    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    def test_contact_who_already_finished_the_flow_free_chats_instead(self, mock_send_message, mock_send_flow, mock_ai):
        """STEP_PRICING_SHOWN (set once the Flow's REVIEW/COMING_SOON/ERROR
        reply lands — see the completion branch above) must not be
        re-intercepted on the very next unrelated message."""
        from .tasks import process_incoming_message

        contact = Contact.objects.create(
            phone_number="915555500024", language="en",
            conversation_step=Contact.STEP_PRICING_SHOWN,
        )
        process_incoming_message.run(contact.id, "what documents do I need?")

        mock_send_flow.assert_not_called()
        mock_ai.assert_called_once()

    @patch("whatsapp_bot.tasks.send_whatsapp_buttons")
    @patch("whatsapp_bot.tasks.send_service_flow")
    def test_disabled_flag_leaves_the_old_welcome_message_untouched(self, mock_send_flow, mock_send_buttons):
        from .tasks import process_incoming_message
        from .services import build_welcome_message

        with override_settings(WHATSAPP_SERVICE_FLOW_ID=""):
            contact = Contact.objects.create(phone_number="915555500025")
            process_incoming_message.run(contact.id, "hi")

        mock_send_flow.assert_not_called()
        mock_send_buttons.assert_called_once()
        self.assertTrue(Message.objects.filter(contact=contact, direction="out", body=build_welcome_message(contact)).exists())


@override_settings(WHATSAPP_SERVICE_FLOW_ID="test-service-flow-id")
class ResumeStepNudgeTests(TestCase):
    """Covers tasks.RESUME_NUDGE_AFTER_TURNS — a soft, in-conversation
    reminder appended to the bot's own reply once a contact has kept
    free-chatting for a few turns while STEP_WELCOMED (the service Flow was
    offered but never opened, or opened and abandoned) without ever
    resolving it, instead of only the separate hour-based cart/dropoff
    nudges catching this later. Every test mocks classify_and_respond (never
    calls Gemini), same convention as NaturalLanguageFlowIntentTests below."""

    def _welcomed_contact(self, phone, **extra):
        return Contact.objects.create(
            phone_number=phone, language="en", conversation_step=Contact.STEP_WELCOMED, **extra,
        )

    def _free_chat_turn(self, contact, text="what's the ITR deadline?"):
        from .tasks import process_incoming_message
        with patch("whatsapp_bot.tasks.classify_and_respond") as mock_classify, \
                patch("whatsapp_bot.tasks.send_whatsapp_message") as mock_send_message:
            mock_classify.return_value = {
                "reply": "ITR filing is usually due by July 31st.",
                "action": "continue_chat", "flow_type": None, "confidence": "low",
            }
            process_incoming_message.run(contact.id, text)
            return mock_send_message.call_args.args[1]

    def test_no_nudge_before_the_threshold(self):
        contact = self._welcomed_contact("915555500060")

        for _ in range(RESUME_NUDGE_AFTER_TURNS - 1):
            body = self._free_chat_turn(contact)
            self.assertNotIn("MENU", body)

        contact.refresh_from_db()
        self.assertEqual(contact.free_chat_turns_since_welcomed, RESUME_NUDGE_AFTER_TURNS - 1)
        self.assertIsNone(contact.resume_nudge_sent_at)

    def test_nudge_appended_once_the_threshold_is_reached(self):
        contact = self._welcomed_contact("915555500061")

        body = None
        for _ in range(RESUME_NUDGE_AFTER_TURNS):
            body = self._free_chat_turn(contact)

        self.assertIn("ITR filing is usually due by July 31st.", body)  # original reply still present
        self.assertIn("MENU", body)
        contact.refresh_from_db()
        self.assertIsNotNone(contact.resume_nudge_sent_at)

    def test_nudge_is_one_shot_not_repeated_every_turn(self):
        contact = self._welcomed_contact("915555500062")

        for _ in range(RESUME_NUDGE_AFTER_TURNS):
            self._free_chat_turn(contact)
        contact.refresh_from_db()
        first_sent_at = contact.resume_nudge_sent_at

        body = self._free_chat_turn(contact)  # one more turn past the threshold

        self.assertNotIn("MENU", body)
        contact.refresh_from_db()
        self.assertEqual(contact.resume_nudge_sent_at, first_sent_at)  # untouched, not re-stamped

    def test_nudge_names_the_saved_cart_when_there_is_one(self):
        from .services import add_service_selection

        contact = self._welcomed_contact("915555500063")
        capital_gains = ServicePricing.objects.get(service_code="itr_capital_gains")
        add_service_selection(contact, capital_gains)

        body = None
        for _ in range(RESUME_NUDGE_AFTER_TURNS):
            body = self._free_chat_turn(contact)

        self.assertIn("Capital Gains", body)
        self.assertIn("MENU", body)

    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    def test_menu_keyword_resets_the_counter(self, mock_send_message):
        from .tasks import process_incoming_message

        contact = self._welcomed_contact("915555500064")
        for _ in range(RESUME_NUDGE_AFTER_TURNS):
            self._free_chat_turn(contact)
        contact.refresh_from_db()
        self.assertGreater(contact.free_chat_turns_since_welcomed, 0)

        process_incoming_message.run(contact.id, "menu")

        contact.refresh_from_db()
        self.assertEqual(contact.free_chat_turns_since_welcomed, 0)
        self.assertIsNone(contact.resume_nudge_sent_at)

    @patch("whatsapp_bot.tasks.send_service_flow")
    def test_confirming_a_natural_language_launch_resets_the_counter(self, mock_send_flow):
        from .tasks import process_incoming_message

        contact = self._welcomed_contact("915555500065", pending_flow_type_candidate="GST_FILING")
        contact.free_chat_turns_since_welcomed = RESUME_NUDGE_AFTER_TURNS
        contact.resume_nudge_sent_at = timezone.now()
        contact.save(update_fields=["free_chat_turns_since_welcomed", "resume_nudge_sent_at"])

        process_incoming_message.run(contact.id, "yes")

        contact.refresh_from_db()
        self.assertEqual(contact.free_chat_turns_since_welcomed, 0)
        self.assertIsNone(contact.resume_nudge_sent_at)

    def test_restart_resets_the_counter(self):
        contact = self._welcomed_contact("915555500066")
        for _ in range(RESUME_NUDGE_AFTER_TURNS):
            self._free_chat_turn(contact)
        contact.refresh_from_db()
        self.assertGreater(contact.free_chat_turns_since_welcomed, 0)

        with patch("whatsapp_bot.tasks.send_service_flow"), patch("whatsapp_bot.tasks.send_whatsapp_message"):
            from .tasks import process_incoming_message
            process_incoming_message.run(contact.id, "restart")

        contact.refresh_from_db()
        self.assertEqual(contact.free_chat_turns_since_welcomed, 0)
        self.assertIsNone(contact.resume_nudge_sent_at)


class NaturalLanguageFlowIntentTests(TestCase):
    """Covers services.classify_and_respond's structured {reply, action,
    flow_type, confidence} result and how tasks.process_incoming_message
    branches on it — the natural-language counterpart to tapping a menu/Flow
    option (e.g. "I want to file my ITR" with no button involved). Every
    test here mocks classify_and_respond itself (never calls Gemini) — same
    "don't hit the network in manage.py test" convention as the rest of this
    suite; classify_and_respond's actual prompt quality is covered by the
    offline eval_flow_intent management command instead, the same split
    eval_ai_persona already uses for get_ai_response's persona rules.

    Every contact here starts at STEP_PRICING_SHOWN — same "already past the
    Flow, now in free-chat" setup as FullCutoverTests' equivalent tests —
    so every message reaches the classify_and_respond call site regardless
    of whether WHATSAPP_SERVICE_FLOW_ID happens to be set in this
    environment's .env."""

    def _chatting_contact(self, phone, **extra):
        return Contact.objects.create(
            phone_number=phone, language="en", conversation_step=Contact.STEP_PRICING_SHOWN, **extra,
        )

    @patch("whatsapp_bot.tasks.classify_and_respond")
    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    def test_continue_chat_just_sends_the_reply(self, mock_send_message, mock_classify):
        """A purely informational question (e.g. 'what's the ITR deadline?')
        must never trigger a Flow launch or the confirm prompt."""
        from .tasks import process_incoming_message

        mock_classify.return_value = {
            "reply": "ITR filing is usually due by July 31st.",
            "action": "continue_chat", "flow_type": None, "confidence": "low",
        }
        contact = self._chatting_contact("915555500050")

        process_incoming_message.run(contact.id, "what's the ITR deadline?")

        mock_send_message.assert_called_once_with("915555500050", "ITR filing is usually due by July 31st.")
        contact.refresh_from_db()
        self.assertEqual(contact.pending_flow_type_candidate, "")

    @patch("whatsapp_bot.tasks.classify_and_respond")
    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    def test_clarify_first_sends_only_the_clarifying_question(self, mock_send_message, mock_classify):
        """clarify_first must not show a confirm screen yet — just the
        question, waiting for the next message to re-classify."""
        from .tasks import process_incoming_message

        mock_classify.return_value = {
            "reply": "Happy to help — is that for GST registration or a GST return filing?",
            "action": "clarify_first", "flow_type": None, "confidence": "low",
        }
        contact = self._chatting_contact("915555500051")

        process_incoming_message.run(contact.id, "help me with GST")

        mock_send_message.assert_called_once_with(
            "915555500051", "Happy to help — is that for GST registration or a GST return filing?",
        )
        contact.refresh_from_db()
        self.assertEqual(contact.pending_flow_type_candidate, "")

    @patch("whatsapp_bot.tasks.send_whatsapp_list")
    @patch("whatsapp_bot.tasks.send_service_flow")
    @patch("whatsapp_bot.tasks.classify_and_respond")
    def test_launch_flow_candidate_auto_launches_the_deep_linked_flow(
        self, mock_classify, mock_send_flow, mock_send_list,
    ):
        """launch_flow_candidate now opens the deep-linked Flow immediately
        (see agents.sales_cart_node) instead of showing a Confirm/Edit list
        first — one tap (the Flow's own CTA button) instead of two. Gemini's
        own reply travels as the Flow message's body_text rather than a
        separate confirm message. No pending_flow_type_candidate is ever set
        for this path anymore — there's nothing left pending to confirm."""
        from .tasks import process_incoming_message

        mock_classify.return_value = {
            "reply": "Got it — let's get your GST registration sorted.",
            "action": "launch_flow_candidate", "flow_type": "GST_FILING", "confidence": "high",
        }
        contact = self._chatting_contact("915555500052")

        process_incoming_message.run(contact.id, "help me register for GST")

        mock_send_list.assert_not_called()
        mock_send_flow.assert_called_once_with(
            contact, deep_link_screen="GST_FILING",
            body_text="Got it — let's get your GST registration sorted.",
        )
        contact.refresh_from_db()
        self.assertEqual(contact.pending_flow_type_candidate, "")

    @patch("whatsapp_bot.tasks.send_service_flow")
    @patch("whatsapp_bot.tasks.classify_and_respond")
    def test_launch_flow_candidate_creates_an_auto_launched_classification_row(self, mock_classify, mock_send_flow):
        """Durable counterpart to the [flow-intent] log line — see
        FlowIntentClassification's docstring for why a log line alone can't
        answer whether Gemini's confidence is trustworthy. Recorded (and
        resolved) as auto_launched immediately — there's no separate
        Confirm/Edit answer to wait on anymore."""
        from .models import FlowIntentClassification
        from .tasks import process_incoming_message

        mock_classify.return_value = {
            "reply": "Got it — let's get your GST registration sorted.",
            "action": "launch_flow_candidate", "flow_type": "GST_FILING", "confidence": "high",
        }
        contact = self._chatting_contact("915555500052")

        process_incoming_message.run(contact.id, "help me register for GST")

        mock_send_flow.assert_called_once_with(
            contact, deep_link_screen="GST_FILING",
            body_text="Got it — let's get your GST registration sorted.",
        )
        row = FlowIntentClassification.objects.get(contact=contact)
        self.assertEqual(row.flow_type, "GST_FILING")
        self.assertEqual(row.confidence, "high")
        self.assertEqual(row.message_text, "help me register for GST")
        self.assertEqual(row.decision, FlowIntentClassification.DECISION_AUTO_LAUNCHED)
        self.assertIsNotNone(row.resolved_at)

    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    @patch("whatsapp_bot.tasks.send_whatsapp_list")
    @patch("whatsapp_bot.tasks.send_service_flow")
    @patch("whatsapp_bot.tasks.classify_and_respond")
    def test_high_and_medium_confidence_behave_identically_low_gets_cross_questioned(
        self, mock_classify, mock_send_flow, mock_send_list, mock_send_message,
    ):
        """Two invariants in one test:

        1. High and medium confidence must behave byte-identically — both
        auto-launch the deep-linked Flow immediately (see
        agents.sales_cart_node), with no Confirm/Edit prompt in between.
        Splitting them further is a decision that can only ever come from
        real evidence (report_flow_intent_confidence's numbers over
        FlowIntentClassification rows), never from trusting Gemini's own
        self-reported `confidence` field directly for finer-grained
        treatment — see FlowIntentClassification's docstring.

        2. Low confidence is the one deliberate, narrow exception — added
        after the 2026-09 GST-misread-as-ITR incident, where a guess Gemini
        itself was unsure about must never reach the customer looking
        authoritative. Low confidence doesn't auto-launch — it cross-
        questions instead of even showing the guess, via agents.
        low_confidence_clarify_node. This is the one remaining checkpoint
        before a Flow opens, now that high/medium skip confirmation."""
        from .tasks import process_incoming_message

        outcomes = {}
        for confidence in ("high", "medium", "low"):
            mock_send_flow.reset_mock()
            mock_send_list.reset_mock()
            mock_send_message.reset_mock()
            mock_classify.return_value = {
                "reply": "Got it — let's get your GST registration sorted.",
                "action": "launch_flow_candidate", "flow_type": "GST_FILING", "confidence": confidence,
            }
            contact = self._chatting_contact(f"91555550{confidence[:3]}99")

            process_incoming_message.run(contact.id, "help me register for GST")

            contact.refresh_from_db()
            outcomes[confidence] = {
                "flow_opened": mock_send_flow.called,
                "confirm_list_sent": mock_send_list.called,
                "pending_flow_type_candidate": contact.pending_flow_type_candidate,
            }

        # High and medium auto-launch, and are indistinguishable — the actual
        # invariant. Neither ever sets pending_flow_type_candidate anymore
        # (nothing is left pending once the Flow's already opened) or shows
        # the old confirm list.
        self.assertEqual(outcomes["high"], outcomes["medium"])
        self.assertTrue(outcomes["high"]["flow_opened"])
        self.assertFalse(outcomes["high"]["confirm_list_sent"])
        self.assertEqual(outcomes["high"]["pending_flow_type_candidate"], "")
        # Low is the deliberate exception: no Flow opened, no confirm list, no
        # pending candidate — a real cross-question instead.
        self.assertFalse(outcomes["low"]["flow_opened"], "low confidence must never auto-launch the Flow")
        self.assertFalse(outcomes["low"]["confirm_list_sent"], "low confidence must not show the confirm list")
        self.assertEqual(outcomes["low"]["pending_flow_type_candidate"], "")
        mock_send_message.assert_called_once()
        clarifying_reply = mock_send_message.call_args.args[1]
        self.assertNotEqual(clarifying_reply, "Got it — let's get your GST registration sorted.")
        self.assertIn("registration", clarifying_reply.lower())

    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    @patch("whatsapp_bot.tasks.send_service_flow")
    @patch("whatsapp_bot.tasks.classify_and_respond")
    def test_low_confidence_direct_service_request_also_gets_cross_questioned(
        self, mock_classify, mock_send_flow, mock_send_message,
    ):
        """Same low-confidence gate, for direct_service_request — a specific-
        sounding item name Gemini itself wasn't confident about (e.g. a
        half-heard/ambiguous voice-to-text message) shouldn't start guided
        intake on the wrong flow_type either."""
        from .tasks import process_incoming_message

        mock_classify.return_value = {
            "reply": "Sure thing.", "action": "direct_service_request", "flow_type": "TDS_FILING",
            "direct_service_query": "the quarterly one", "confidence": "low",
        }
        contact = self._chatting_contact("915555500056")

        process_incoming_message.run(contact.id, "something about the quarterly thing")

        mock_send_flow.assert_not_called()
        mock_send_message.assert_called_once()
        clarifying_reply = mock_send_message.call_args.args[1]
        self.assertNotEqual(clarifying_reply, "Sure thing.")
        contact.refresh_from_db()
        self.assertEqual(contact.active_intake_flow_type, "")

    @patch("whatsapp_bot.tasks.send_service_flow")
    def test_confirming_the_launch_opens_the_deep_linked_flow(self, mock_send_flow):
        from .tasks import process_incoming_message

        contact = self._chatting_contact("915555500053", pending_flow_type_candidate="ITR_TYPE")

        process_incoming_message.run(contact.id, "yes")

        mock_send_flow.assert_called_once_with(contact, deep_link_screen="ITR_TYPE", body_text=None)
        contact.refresh_from_db()
        self.assertEqual(contact.pending_flow_type_candidate, "")

    @patch("whatsapp_bot.tasks.send_service_flow")
    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    def test_rejecting_the_launch_opens_nothing(self, mock_send_message, mock_send_flow):
        from .tasks import process_incoming_message

        contact = self._chatting_contact("915555500054", pending_flow_type_candidate="GST_FILING")

        process_incoming_message.run(contact.id, "no")

        mock_send_flow.assert_not_called()
        mock_send_message.assert_called_once()
        contact.refresh_from_db()
        self.assertEqual(contact.pending_flow_type_candidate, "")

    @patch("whatsapp_bot.tasks.send_service_flow")
    def test_legacy_pending_confirm_still_resolves_the_classification_row_as_confirmed(self, mock_send_flow):
        """Legacy path only (see FlowIntentClassification's docstring) — a
        contact that already had pending_flow_type_candidate set before the
        auto-launch change still gets its Confirm/Edit answer resolved
        correctly. New turns never take this path anymore (launch_flow_
        candidate auto-launches instead — see the row above)."""
        from .models import FlowIntentClassification
        from .tasks import process_incoming_message

        contact = self._chatting_contact("915555500052", pending_flow_type_candidate="GST_FILING")
        row = FlowIntentClassification.objects.create(
            contact=contact, flow_type="GST_FILING", confidence="high", message_text="help me register for GST",
        )

        process_incoming_message.run(contact.id, "yes")

        mock_send_flow.assert_called_once_with(contact, deep_link_screen="GST_FILING", body_text=None)
        row.refresh_from_db()
        self.assertEqual(row.decision, FlowIntentClassification.DECISION_CONFIRMED)
        self.assertIsNotNone(row.resolved_at)

    @patch("whatsapp_bot.tasks.send_service_flow")
    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    def test_legacy_pending_reject_still_resolves_the_classification_row_as_rejected(
        self, mock_send_message, mock_send_flow,
    ):
        """Legacy path only — see the docstring above."""
        from .models import FlowIntentClassification
        from .tasks import process_incoming_message

        contact = self._chatting_contact("915555500052", pending_flow_type_candidate="GST_FILING")
        row = FlowIntentClassification.objects.create(
            contact=contact, flow_type="GST_FILING", confidence="medium", message_text="help me register for GST",
        )

        process_incoming_message.run(contact.id, "no")

        mock_send_flow.assert_not_called()
        mock_send_message.assert_called_once()
        row.refresh_from_db()
        self.assertEqual(row.decision, FlowIntentClassification.DECISION_REJECTED)
        self.assertIsNotNone(row.resolved_at)

    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    @patch("whatsapp_bot.tasks.send_whatsapp_list")
    @patch("whatsapp_bot.tasks.send_service_flow")
    @patch("whatsapp_bot.tasks.classify_and_respond")
    def test_unparseable_reply_while_pending_defers_to_free_chat(
        self, mock_classify, mock_send_flow, mock_send_list, mock_send_message,
    ):
        """Regression test for the 2026-09-10 stuck-in-TDS-filing trace: a
        reply that doesn't match yes/no (a real question, a name, an
        unrelated request — 'banana' stands in for all of those) must NOT
        just re-show the same confirm prompt forever. It's treated as an
        implicit decline — pending_flow_type_candidate clears and the
        classification resolves rejected — and the SAME message gets a real
        answer from free-chat this turn, same as active_intake_flow_type's
        interruption handling."""
        from .models import FlowIntentClassification
        from .tasks import process_incoming_message

        contact = self._chatting_contact("915555500055", pending_flow_type_candidate="NOTICES")
        classification = FlowIntentClassification.objects.create(
            contact=contact, flow_type="NOTICES", confidence="high", message_text="I got a notice",
        )
        mock_classify.return_value = {
            "reply": "Bananas aren't really my department!",
            "action": "continue_chat", "flow_type": None, "confidence": "low",
        }

        process_incoming_message.run(contact.id, "banana")

        mock_send_flow.assert_not_called()
        mock_send_list.assert_not_called()
        mock_send_message.assert_called_once_with("915555500055", "Bananas aren't really my department!")
        contact.refresh_from_db()
        self.assertEqual(contact.pending_flow_type_candidate, "")
        classification.refresh_from_db()
        self.assertEqual(classification.decision, FlowIntentClassification.DECISION_REJECTED)

    @override_settings(WHATSAPP_SERVICE_FLOW_ID="test-mega-flow-id")
    @patch("whatsapp_bot.tasks.send_service_flow")
    def test_pending_confirm_resolves_before_full_cutover_would_intercept(self, mock_send_flow):
        """The exact ordering bug flagged during scan: a pending confirm
        must resolve BEFORE the FULL CUTOVER branch, or a contact whose
        conversation_step isn't STEP_WELCOMED/STEP_PRICING_SHOWN would get
        the whole mega Flow re-sent from SERVICE instead of the deep-linked
        confirm reaching its target."""
        from .tasks import process_incoming_message

        contact = Contact.objects.create(
            phone_number="915555500056", language="en",
            conversation_step=Contact.STEP_CATEGORY_SHOWN,  # not exempt from FULL CUTOVER
            pending_flow_type_candidate="BUSINESS_REG",
        )

        process_incoming_message.run(contact.id, "yes")

        mock_send_flow.assert_called_once_with(contact, deep_link_screen="BUSINESS_REG", body_text=None)
        contact.refresh_from_db()
        self.assertEqual(contact.pending_flow_type_candidate, "")

    def _cart_contact(self, phone):
        """A contact already sitting on a confirmed cart — the state
        edit_cart requires (see services._build_system_instruction's cart
        context note)."""
        contact = self._chatting_contact(phone)
        capital_gains = ServicePricing.objects.get(service_code="itr_capital_gains")
        from .services import add_service_selection
        add_service_selection(contact, capital_gains)
        return contact

    @override_settings(RAZORPAY_KEY_ID="", RAZORPAY_KEY_SECRET="")
    @patch("whatsapp_bot.tasks.classify_and_respond")
    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    def test_edit_cart_remove_updates_the_cart_and_shows_it(self, mock_send_message, mock_classify):
        """No payment link available (Razorpay unconfigured here) — falls
        through to a plain updated-cart message, same fallback as a fresh
        checkout-confirm with no link (see PaymentCtaGatingTests)."""
        from .tasks import process_incoming_message

        contact = self._cart_contact("915555500060")
        mock_classify.return_value = {
            "reply": "Sure, one sec.", "action": "edit_cart", "flow_type": None,
            "cart_edit_operation": "remove", "cart_edit_query": "capital gains", "confidence": "high",
        }

        process_incoming_message.run(contact.id, "actually remove the capital gains one")

        contact.refresh_from_db()
        self.assertNotIn(
            "itr_capital_gains", {item["service_code"] for item in contact.selected_services},
        )
        self.assertEqual(contact.conversation_step, Contact.STEP_CONFIRMING_QUOTE)
        body = mock_send_message.call_args.args[1]
        self.assertIn("Removed", body)
        self.assertIn("Capital Gains", body)

    @override_settings(RAZORPAY_KEY_ID="", RAZORPAY_KEY_SECRET="")
    @patch("whatsapp_bot.tasks.classify_and_respond")
    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    def test_edit_cart_add_updates_the_cart_and_shows_it(self, mock_send_message, mock_classify):
        from .tasks import process_incoming_message

        contact = self._cart_contact("915555500061")
        mock_classify.return_value = {
            "reply": "Sure, one sec.", "action": "edit_cart", "flow_type": None,
            "cart_edit_operation": "add", "cart_edit_query": "house property", "confidence": "high",
        }

        process_incoming_message.run(contact.id, "can you add house property too")

        contact.refresh_from_db()
        self.assertIn(
            "itr_house_property", {item["service_code"] for item in contact.selected_services},
        )
        body = mock_send_message.call_args.args[1]
        self.assertIn("Added", body)
        self.assertIn("House Property", body)

    @override_settings(RAZORPAY_KEY_ID="test-key", RAZORPAY_KEY_SECRET="test-secret")
    @patch("whatsapp_bot.tasks.classify_and_respond")
    @patch("whatsapp_bot.services.send_whatsapp_cta_url")
    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    @patch("whatsapp_bot.services.payments.create_payment_link", return_value=True)
    def test_edit_cart_sends_a_fresh_link_and_invalidates_the_old_one(
        self, mock_create_link, mock_send_message, mock_send_cta, mock_classify,
    ):
        """The concrete ask: after a cart edit, the customer gets an updated
        payment link (folded into the updated-cart summary, one message —
        see services.maybe_send_payment_cta) and any payment link already
        issued against the pre-edit cart is invalidated, both locally
        (STATUS_EXPIRED) and on Razorpay's side (payments.cancel_payment_link)
        — see services.invalidate_stale_orders."""
        from .services import maybe_send_payment_cta
        from .tasks import process_incoming_message

        # A cart edit that empties the cart entirely has no amount to quote
        # (correctly no link, per maybe_send_payment_cta's gate 1) — use
        # "add" here so the post-edit cart is still non-empty, isolating
        # what this test actually covers: the old link being invalidated
        # and a fresh one going out for the new total.
        contact = self._cart_contact("915555500064")
        contact.last_quote_amount = 999
        contact.save(update_fields=["last_quote_amount"])
        # An already-issued, still-open link against the pre-edit cart.
        maybe_send_payment_cta(contact)
        old_order = Order.objects.get(contact=contact, status=Order.STATUS_CREATED)
        mock_create_link.reset_mock()
        mock_send_cta.reset_mock()

        mock_classify.return_value = {
            "reply": "Sure, one sec.", "action": "edit_cart", "flow_type": None,
            "cart_edit_operation": "add", "cart_edit_query": "house property", "confidence": "high",
        }
        process_incoming_message.run(contact.id, "can you add house property too")

        old_order.refresh_from_db()
        self.assertEqual(old_order.status, Order.STATUS_EXPIRED)
        self.assertEqual(old_order.failure_reason, "cart_edited")

        mock_send_cta.assert_called_once()
        self.assertIn("Added", mock_send_cta.call_args.args[1])
        mock_send_message.assert_not_called()

        new_order = Order.objects.filter(contact=contact, status=Order.STATUS_CREATED).exclude(id=old_order.id).first()
        self.assertIsNotNone(new_order)

    @patch("whatsapp_bot.tasks.classify_and_respond")
    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    def test_edit_cart_with_no_confident_match_asks_for_clarification(self, mock_send_message, mock_classify):
        from .tasks import process_incoming_message

        contact = self._cart_contact("915555500062")
        mock_classify.return_value = {
            "reply": "Sure, one sec.", "action": "edit_cart", "flow_type": None,
            "cart_edit_operation": "remove", "cart_edit_query": "asdkfjasdkfj nonsense",
            "confidence": "medium",
        }

        process_incoming_message.run(contact.id, "remove the asdkfjasdkfj thing")

        contact.refresh_from_db()
        # Cart untouched — no confident match, so nothing was mutated.
        self.assertIn(
            "itr_capital_gains", {item["service_code"] for item in contact.selected_services},
        )
        self.assertIn("couldn't find", mock_send_message.call_args.args[1])

    @patch("whatsapp_bot.tasks.classify_and_respond")
    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    def test_edit_cart_is_ignored_with_an_empty_cart(self, mock_send_message, mock_classify):
        """Defensive guard: even if classify_and_respond somehow returns
        edit_cart for a contact with nothing in their cart (it's told not
        to), the reply falls through unmodified rather than crashing on an
        empty selection."""
        from .tasks import process_incoming_message

        contact = self._chatting_contact("915555500063")
        mock_classify.return_value = {
            "reply": "I don't think you have anything selected yet.", "action": "edit_cart",
            "flow_type": None, "cart_edit_operation": "remove", "cart_edit_query": "capital gains",
            "confidence": "low",
        }

        process_incoming_message.run(contact.id, "remove the capital gains one")

        mock_send_message.assert_called_once_with(
            "915555500063", "I don't think you have anything selected yet.",
        )

    def test_malformed_classification_output_degrades_to_safe_continue_chat(self):
        """classify_and_respond must never raise or launch anything on
        malformed/schema-violating model output — see
        services._parse_classification_response."""
        from .services import _parse_classification_response

        result = _parse_classification_response("not valid json at all", "fallback reply")
        self.assertEqual(result, {
            "reply": "fallback reply", "action": "continue_chat", "flow_type": None,
            "direct_service_query": "", "cart_edit_operation": None, "cart_edit_query": "", "confidence": "low",
        })

    def test_classification_output_with_bad_flow_type_falls_back_to_continue_chat(self):
        """An enum-violating flow_type (or one paired with the wrong action)
        must never reach the launch path — see
        services._parse_classification_response."""
        from .services import _parse_classification_response

        result = _parse_classification_response(
            json.dumps({
                "reply": "Sure!", "action": "launch_flow_candidate",
                "flow_type": "NOT_A_REAL_SCREEN", "confidence": "high",
            }),
            "fallback reply",
        )
        self.assertIsNone(result["flow_type"])

    def test_classification_flow_type_is_dropped_when_action_is_not_launch(self):
        """Belt-and-braces case from _parse_classification_response: even a
        schema-valid flow_type must be discarded if action isn't
        launch_flow_candidate, so a stray field can never cause a launch."""
        from .services import _parse_classification_response

        result = _parse_classification_response(
            json.dumps({
                "reply": "That's outside what we handle.", "action": "continue_chat",
                "flow_type": "GST_FILING", "confidence": "medium",
            }),
            "fallback reply",
        )
        self.assertIsNone(result["flow_type"])
        self.assertEqual(result["action"], "continue_chat")


class MultiAgentGraphTests(TestCase):
    """Covers agents.py's Supervisor/Router + specialist nodes — the two new
    classify_and_respond actions (request_handoff, check_order_status) this
    layer adds, plus the router's defensive fallthrough. Same "mock
    classify_and_respond itself, never call Gemini" convention as
    NaturalLanguageFlowIntentTests; classify_and_respond's call site in
    tasks.py is unchanged by this refactor, so it's still patched at
    whatsapp_bot.tasks.classify_and_respond."""

    def _chatting_contact(self, phone, **extra):
        return Contact.objects.create(
            phone_number=phone, language="en", conversation_step=Contact.STEP_PRICING_SHOWN, **extra,
        )

    @patch("whatsapp_bot.tasks.classify_and_respond")
    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    def test_request_handoff_flags_for_review_without_pausing_the_bot(self, mock_send_message, mock_classify):
        """The human-handoff pause feature was removed: a handoff still
        flags the contact for staff triage, but must no longer silence the
        bot's own auto-replies (see create_handoff_ticket in agents.py)."""
        from .tasks import process_incoming_message

        mock_classify.return_value = {
            "reply": "Sure, let me connect you.", "action": "request_handoff", "flow_type": None,
            "cart_edit_operation": None, "cart_edit_query": "", "confidence": "high",
        }
        contact = self._chatting_contact("915555500080")

        process_incoming_message.run(contact.id, "can I talk to a real person")

        contact.refresh_from_db()
        self.assertIsNone(contact.bot_paused_until)
        self.assertTrue(contact.needs_human_review)
        self.assertEqual(contact.needs_human_review_reason, "chat_requested_human")
        # has_requested_help is specifically the HELP-keyword flag — a
        # free-text handoff ask must not redefine its meaning.
        self.assertFalse(contact.has_requested_help)

    @patch("whatsapp_bot.tasks.classify_and_respond")
    @patch("whatsapp_bot.tasks.send_whatsapp_buttons")
    def test_check_order_status_reaches_the_same_reply_the_status_keyword_gives(
        self, mock_send_buttons, mock_classify,
    ):
        from .services import ORDER_STATUS_RESEND_PREFIX
        from .tasks import process_incoming_message

        mock_classify.return_value = {
            "reply": "Let me check that for you.", "action": "check_order_status", "flow_type": None,
            "cart_edit_operation": None, "cart_edit_query": "", "confidence": "high",
        }
        contact = self._chatting_contact("915555500081")
        order = Order.objects.create(contact=contact, amount=1500, status=Order.STATUS_CREATED)

        process_incoming_message.run(contact.id, "what's happening with my order")

        options = mock_send_buttons.call_args.args[2]
        self.assertIn(f"{ORDER_STATUS_RESEND_PREFIX}{order.id}", [opt[0] for opt in options])

    @patch("whatsapp_bot.tasks.classify_and_respond")
    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    def test_router_falls_through_to_general_chat_on_launch_with_no_flow_type(
        self, mock_send_message, mock_classify,
    ):
        """Same defensive case as NaturalLanguageFlowIntentTests' bad-flow_type
        coverage, exercised end to end: launch_flow_candidate with a falsy
        flow_type must never reach the Sales & Cart node — Gemini's `reply`
        goes out verbatim instead, exactly like continue_chat."""
        from .tasks import process_incoming_message

        mock_classify.return_value = {
            "reply": "Not sure which service you mean.", "action": "launch_flow_candidate",
            "flow_type": None, "cart_edit_operation": None, "cart_edit_query": "", "confidence": "low",
        }
        contact = self._chatting_contact("915555500082")

        process_incoming_message.run(contact.id, "I want to start something")

        mock_send_message.assert_called_once_with("915555500082", "Not sure which service you mean.")
        contact.refresh_from_db()
        self.assertEqual(contact.pending_flow_type_candidate, "")


class GuidedIntakeMissingSlotTests(TestCase):
    """Covers intake.compute_missing_slot in isolation — the pure function
    everything else in intake.py is built around. No Gemini/classifier
    involved; just catalogue + contact.selected_services in, next slot (or
    None) out."""

    def _contact(self, **kwargs):
        return Contact.objects.create(phone_number="915555500090", language="en", **kwargs)

    def test_flat_category_missing_until_one_service_is_selected(self):
        from .intake import compute_missing_slot

        contact = self._contact()
        slot = compute_missing_slot(contact, "TDS_FILING")
        self.assertEqual(slot["kind"], "flat")
        self.assertEqual(slot["category_code"], "tds_filing")
        option_ids = [opt[0] for opt in slot["options"]]
        self.assertIn("tds_quarterly_filing", option_ids)

        service = ServicePricing.objects.get(service_code="tds_quarterly_filing")
        contact.selected_services = [{"service_code": service.service_code, "quantity": 1}]
        contact.save(update_fields=["selected_services"])

        self.assertIsNone(compute_missing_slot(contact, "TDS_FILING"))

    def test_itr_wizard_asks_type_then_addons_then_nothing(self):
        from .intake import compute_missing_slot

        contact = self._contact()
        slot = compute_missing_slot(contact, "ITR_TYPE")
        self.assertEqual(slot["kind"], "itr_type")

        contact.pending_parent_service_code = "itr_salary"
        contact.save(update_fields=["pending_parent_service_code"])
        slot = compute_missing_slot(contact, "ITR_TYPE")
        self.assertEqual(slot["kind"], "itr_addons")
        self.assertEqual(slot["parent"].service_code, "itr_salary")

        contact.pending_parent_service_code = ""
        contact.selected_services = [{"service_code": "itr_basic_salary", "quantity": 1}]
        contact.save(update_fields=["pending_parent_service_code", "selected_services"])
        self.assertIsNone(compute_missing_slot(contact, "ITR_TYPE"))

    def test_itr_flat_type_needs_no_addon_step(self):
        """A flat-priced ITR type (not a wizard parent) goes straight to
        'nothing missing' the moment its own code is on the cart — no
        itr_addons slot, mirroring flow_screens._handle_itr_type_screen's
        flat branch."""
        from .intake import compute_missing_slot

        contact = self._contact()
        flat_type = ServicePricing.objects.filter(
            category__category_code="itr", parent_service__isnull=True,
        ).exclude(service_code__in=["itr_salary", "itr_business", "itr_nri"]).first()
        self.assertIsNotNone(flat_type, "expected at least one flat-priced ITR type to be seeded")

        contact.selected_services = [{"service_code": flat_type.service_code, "quantity": 1}]
        contact.save(update_fields=["selected_services"])
        self.assertIsNone(compute_missing_slot(contact, "ITR_TYPE"))


class GuidedIntakeEndToEndTests(TestCase):
    """direct_service_request now auto-launches the deep-linked Flow via
    agents.sales_cart_node, same as launch_flow_candidate (see its
    docstring) — agents.guided_intake_node and the chat-only wizard it drove
    are dead code for any fresh turn now (kept only so a contact already
    mid-wizard from before this change can finish it — see
    tasks.process_incoming_message's active_intake_flow_type branch and
    GuidedIntakeMissingSlotTests for that mechanism's own unit coverage,
    still exercised directly against intake.py). Same 'mock
    classify_and_respond itself, never call Gemini' convention as
    NaturalLanguageFlowIntentTests/MultiAgentGraphTests."""

    def _contact(self, phone, **extra):
        return Contact.objects.create(
            phone_number=phone, language="en", conversation_step=Contact.STEP_PRICING_SHOWN, **extra,
        )

    @patch("whatsapp_bot.tasks.send_service_flow")
    @patch("whatsapp_bot.tasks.classify_and_respond")
    def test_specific_flat_request_auto_launches_the_deep_linked_flow(self, mock_classify, mock_send_flow):
        """A specific-enough ask (e.g. 'TDS quarterly filing', naming the
        item not just the category) auto-launches the same as a vague one —
        no chat-only resolution/add-on questions anymore, no
        active_intake_flow_type ever set."""
        from .tasks import process_incoming_message

        mock_classify.return_value = {
            "reply": "Sure thing — let's get your TDS quarterly filing sorted.",
            "action": "direct_service_request", "flow_type": "TDS_FILING",
            "direct_service_query": "TDS quarterly filing", "cart_edit_operation": None,
            "cart_edit_query": "", "confidence": "high",
        }
        contact = self._contact("915555500091")

        process_incoming_message.run(contact.id, "I need TDS quarterly filing")

        mock_send_flow.assert_called_once_with(
            contact, deep_link_screen="TDS_FILING",
            body_text="Sure thing — let's get your TDS quarterly filing sorted.",
        )
        contact.refresh_from_db()
        self.assertEqual(contact.active_intake_flow_type, "")
        self.assertEqual(contact.selected_services, [])

    @patch("whatsapp_bot.tasks.send_service_flow")
    @patch("whatsapp_bot.tasks.classify_and_respond")
    def test_vague_category_direct_service_request_also_auto_launches(self, mock_classify, mock_send_flow):
        """Medium confidence behaves the same as high — see
        NaturalLanguageFlowIntentTests' equivalent invariant for
        launch_flow_candidate; this is the same gate, same node, just a
        different action value reaching it."""
        from .tasks import process_incoming_message

        mock_classify.return_value = {
            "reply": "Let's sort your TDS.", "action": "direct_service_request", "flow_type": "TDS_FILING",
            "direct_service_query": "TDS filing", "cart_edit_operation": None,
            "cart_edit_query": "", "confidence": "medium",
        }
        contact = self._contact("915555500092")

        process_incoming_message.run(contact.id, "help me with TDS filing")

        mock_send_flow.assert_called_once_with(
            contact, deep_link_screen="TDS_FILING", body_text="Let's sort your TDS.",
        )
        contact.refresh_from_db()
        self.assertEqual(contact.active_intake_flow_type, "")

    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    @patch("whatsapp_bot.tasks.classify_and_respond")
    def test_low_confidence_direct_service_request_still_cross_questions(self, mock_classify, mock_send_message):
        """The one remaining checkpoint (agents.low_confidence_clarify_node)
        still applies to direct_service_request too — a low-confidence
        specific-sounding guess doesn't auto-launch either."""
        from .tasks import process_incoming_message

        mock_classify.return_value = {
            "reply": "Let's do your ITR.", "action": "direct_service_request", "flow_type": "ITR_TYPE",
            "direct_service_query": "ITR", "cart_edit_operation": None,
            "cart_edit_query": "", "confidence": "low",
        }
        contact = self._contact("915555500093")

        with patch("whatsapp_bot.tasks.send_service_flow") as mock_send_flow:
            process_incoming_message.run(contact.id, "I need to file my ITR")

        mock_send_flow.assert_not_called()
        mock_send_message.assert_called_once()
        contact.refresh_from_db()
        self.assertEqual(contact.active_intake_flow_type, "")

    def test_restart_menu_help_all_clear_active_intake_flow_type(self):
        from .tasks import process_incoming_message

        for keyword in ("restart", "menu", "help"):
            contact = self._contact(f"91555550{hash(keyword) % 10000:04d}", active_intake_flow_type="TDS_FILING")
            with patch("whatsapp_bot.tasks.send_whatsapp_message"), \
                 patch("whatsapp_bot.tasks.send_whatsapp_buttons"), \
                 patch("whatsapp_bot.tasks.notify_human_alert"):
                process_incoming_message.run(contact.id, keyword)
            contact.refresh_from_db()
            self.assertEqual(contact.active_intake_flow_type, "", f"{keyword} should have cleared it")

    def test_restart_and_menu_escape_a_stale_flow_launch_prompt(self):
        """Regression test: a contact sitting on a launch-confirm prompt
        (pending_flow_type_candidate set) must be able to escape it with
        RESTART or MENU rather than getting the exact same "Want me to open
        X for you now?" prompt re-shown forever — see the keyword_command
        carve-out on the pending_flow_type_candidate branch in tasks.py.
        Before that fix, neither RESTART nor MENU could ever reach their own
        branches: this exact elif came first and matched unconditionally."""
        from .tasks import process_incoming_message

        for keyword in ("restart", "menu"):
            contact = self._contact(
                f"91555551{hash(keyword) % 10000:04d}",
                pending_flow_type_candidate="ITR_TYPE",
            )
            with patch("whatsapp_bot.tasks.send_whatsapp_message"), \
                 patch("whatsapp_bot.tasks.send_whatsapp_buttons"), \
                 patch("whatsapp_bot.tasks.send_service_flow"), \
                 patch("whatsapp_bot.tasks.service_flow_enabled", return_value=True):
                process_incoming_message.run(contact.id, keyword)
            contact.refresh_from_db()
            self.assertEqual(
                contact.pending_flow_type_candidate, "",
                f"{keyword} should have cleared the stale flow-launch prompt",
            )
            last_out = contact.messages.filter(direction="out").latest("created_at")
            self.assertNotIn(
                "Want me to open", last_out.body,
                f"{keyword} should not just re-show the stale ITR launch prompt",
            )


class AbandonedCartNudgeTests(TestCase):
    """Covers tasks.send_abandoned_cart_nudges — the Celery Beat periodic
    task that reminds a contact sitting on an unconfirmed review screen
    (see services.mark_review_reached) at +2h and +24h."""

    def _stale_contact(self, phone, hours_ago, **extra):
        contact = Contact.objects.create(phone_number=phone, language="en", **extra)
        contact.review_reached_at = timezone.now() - timezone.timedelta(hours=hours_ago, minutes=5)
        contact.save(update_fields=["review_reached_at"])
        return contact

    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    def test_sends_and_stamps_the_2h_nudge(self, mock_send_message):
        from .tasks import send_abandoned_cart_nudges

        contact = self._stale_contact("915555500040", hours_ago=2)
        mock_send_message.return_value = {"messages": [{"id": "wamid.1"}]}

        send_abandoned_cart_nudges.run()

        mock_send_message.assert_called_once()
        contact.refresh_from_db()
        self.assertIsNotNone(contact.cart_nudge_2h_sent_at)
        self.assertIsNone(contact.cart_nudge_24h_sent_at)  # not due yet

    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    def test_does_not_resend_the_2h_nudge_once_already_sent(self, mock_send_message):
        from .tasks import send_abandoned_cart_nudges

        contact = self._stale_contact("915555500041", hours_ago=2)
        contact.cart_nudge_2h_sent_at = timezone.now()
        contact.save(update_fields=["cart_nudge_2h_sent_at"])

        send_abandoned_cart_nudges.run()

        mock_send_message.assert_not_called()

    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    def test_sends_the_24h_nudge_independently_of_the_2h_one(self, mock_send_message):
        from .tasks import send_abandoned_cart_nudges

        contact = self._stale_contact("915555500042", hours_ago=24)
        contact.cart_nudge_2h_sent_at = timezone.now() - timezone.timedelta(hours=22)  # already handled earlier
        contact.save(update_fields=["cart_nudge_2h_sent_at"])
        mock_send_message.return_value = {"messages": [{"id": "wamid.2"}]}

        send_abandoned_cart_nudges.run()

        mock_send_message.assert_called_once()
        contact.refresh_from_db()
        self.assertIsNotNone(contact.cart_nudge_24h_sent_at)

    @override_settings(WHATSAPP_CART_NUDGE_TEMPLATE_NAME="")
    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    def test_a_failed_send_with_no_template_configured_leaves_it_unstamped_to_retry(self, mock_send_message):
        """A free-form send outside WhatsApp's 24h window fails (returns
        None, same as any other WhatsApp API failure). With no fallback
        template configured, the timestamp must stay unset so the next
        Beat run retries — stamping it here would falsely record the nudge
        as sent when nobody actually received anything."""
        from .tasks import send_abandoned_cart_nudges

        contact = self._stale_contact("915555500043", hours_ago=24)
        mock_send_message.return_value = None

        send_abandoned_cart_nudges.run()

        contact.refresh_from_db()
        self.assertIsNone(contact.cart_nudge_24h_sent_at)

    @override_settings(WHATSAPP_CART_NUDGE_TEMPLATE_NAME="cart_abandonment_nudge", WHATSAPP_CART_NUDGE_TEMPLATE_LANG="en")
    @patch("whatsapp_bot.tasks.send_whatsapp_template")
    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    def test_a_failed_send_falls_back_to_the_template_and_stamps_on_success(
        self, mock_send_message, mock_send_template,
    ):
        """Once WHATSAPP_CART_NUDGE_TEMPLATE_NAME is configured, a failed
        free-form send (outside the 24h window) falls back to the approved
        template — see services.send_whatsapp_template — and the timestamp
        is stamped once THAT succeeds."""
        from .tasks import send_abandoned_cart_nudges

        contact = self._stale_contact("915555500047", hours_ago=24)
        contact.cart_nudge_2h_sent_at = timezone.now() - timezone.timedelta(hours=22)  # already handled earlier
        contact.save(update_fields=["cart_nudge_2h_sent_at"])
        mock_send_message.return_value = None
        mock_send_template.return_value = {"messages": [{"id": "wamid.3"}]}

        send_abandoned_cart_nudges.run()

        mock_send_template.assert_called_once()
        contact.refresh_from_db()
        self.assertIsNotNone(contact.cart_nudge_24h_sent_at)

    @override_settings(WHATSAPP_CART_NUDGE_TEMPLATE_NAME="cart_abandonment_nudge", WHATSAPP_CART_NUDGE_TEMPLATE_LANG="en")
    @patch("whatsapp_bot.tasks.send_whatsapp_template")
    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    def test_the_template_fallback_also_fails_leaves_it_unstamped(
        self, mock_send_message, mock_send_template,
    ):
        """Belt and braces: if even the template send fails, still don't
        stamp — same "only stamp on real success" rule either way."""
        from .tasks import send_abandoned_cart_nudges

        contact = self._stale_contact("915555500048", hours_ago=24)
        contact.cart_nudge_2h_sent_at = timezone.now() - timezone.timedelta(hours=22)  # already handled earlier
        contact.save(update_fields=["cart_nudge_2h_sent_at"])
        mock_send_message.return_value = None
        mock_send_template.return_value = None

        send_abandoned_cart_nudges.run()

        contact.refresh_from_db()
        self.assertIsNone(contact.cart_nudge_24h_sent_at)

    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    def test_a_contact_who_confirmed_is_never_nudged(self, mock_send_message):
        from .tasks import send_abandoned_cart_nudges

        contact = Contact.objects.create(phone_number="915555500044", language="en")  # review_reached_at blank

        send_abandoned_cart_nudges.run()

        mock_send_message.assert_not_called()


class DropoffNudgeTests(TestCase):
    """Covers tasks.send_dropoff_nudges — the 30m/20h/48h/7d checkpoint
    schedule, additional to send_abandoned_cart_nudges' 2h/24h, gated per
    stage by lead_tier: 30m and 7d are hot-only, 20h and 48h are hot+warm."""

    def _stale_contact(self, phone, hours_ago, lead_tier, **extra):
        contact = Contact.objects.create(phone_number=phone, language="en", lead_tier=lead_tier, **extra)
        contact.review_reached_at = timezone.now() - timezone.timedelta(hours=hours_ago, minutes=5)
        contact.save(update_fields=["review_reached_at"])
        return contact

    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    def test_hot_lead_gets_the_30m_nudge(self, mock_send_message):
        from .tasks import send_dropoff_nudges

        contact = self._stale_contact("915555500070", hours_ago=0.6, lead_tier=Contact.LEAD_TIER_HOT)
        mock_send_message.return_value = {"messages": [{"id": "wamid.1"}]}

        send_dropoff_nudges.run()

        mock_send_message.assert_called_once()
        contact.refresh_from_db()
        self.assertIsNotNone(contact.dropoff_nudge_30m_sent_at)

    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    def test_cold_lead_never_gets_the_30m_nudge(self, mock_send_message):
        """The whole point of this schedule: skip low-intent noise (a cold
        contact who bounced at the welcome menu) even though they'd qualify
        on time alone."""
        from .tasks import send_dropoff_nudges

        self._stale_contact("915555500071", hours_ago=0.6, lead_tier=Contact.LEAD_TIER_COLD)

        send_dropoff_nudges.run()

        mock_send_message.assert_not_called()

    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    def test_warm_lead_does_not_get_the_30m_nudge_but_does_get_20h(self, mock_send_message):
        from .tasks import send_dropoff_nudges

        contact = self._stale_contact("915555500072", hours_ago=0.6, lead_tier=Contact.LEAD_TIER_WARM)

        send_dropoff_nudges.run()
        mock_send_message.assert_not_called()  # 30m is hot-only

        contact.review_reached_at = timezone.now() - timezone.timedelta(hours=20, minutes=5)
        contact.save(update_fields=["review_reached_at"])
        mock_send_message.return_value = {"messages": [{"id": "wamid.2"}]}

        send_dropoff_nudges.run()

        mock_send_message.assert_called_once()
        contact.refresh_from_db()
        self.assertIsNotNone(contact.dropoff_nudge_20h_sent_at)

    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    def test_warm_lead_never_gets_the_7d_nudge(self, mock_send_message):
        """7d is hot-only, same as 30m — warm qualifies for 20h/48h only.
        Those two stages are pre-marked sent so only 7d's own eligibility
        is under test here."""
        from .tasks import send_dropoff_nudges

        contact = self._stale_contact("915555500073", hours_ago=24 * 7 + 1, lead_tier=Contact.LEAD_TIER_WARM)
        contact.dropoff_nudge_20h_sent_at = timezone.now()
        contact.dropoff_nudge_48h_sent_at = timezone.now()
        contact.save(update_fields=["dropoff_nudge_20h_sent_at", "dropoff_nudge_48h_sent_at"])

        send_dropoff_nudges.run()

        mock_send_message.assert_not_called()

    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    def test_does_not_resend_a_stage_once_already_sent(self, mock_send_message):
        from .tasks import send_dropoff_nudges

        contact = self._stale_contact("915555500074", hours_ago=0.6, lead_tier=Contact.LEAD_TIER_HOT)
        contact.dropoff_nudge_30m_sent_at = timezone.now()
        contact.save(update_fields=["dropoff_nudge_30m_sent_at"])

        send_dropoff_nudges.run()

        mock_send_message.assert_not_called()

    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    def test_runs_independently_of_the_2h_24h_cart_nudge_schedule(self, mock_send_message):
        """Both schedules key off the same review_reached_at signal but
        track separate sent-at fields — a contact already nudged by
        send_abandoned_cart_nudges must still be eligible here."""
        from .tasks import send_abandoned_cart_nudges, send_dropoff_nudges

        contact = self._stale_contact("915555500075", hours_ago=24, lead_tier=Contact.LEAD_TIER_HOT)
        mock_send_message.return_value = {"messages": [{"id": "wamid.3"}]}

        send_abandoned_cart_nudges.run()  # stamps cart_nudge_2h/24h_sent_at
        mock_send_message.reset_mock()
        mock_send_message.return_value = {"messages": [{"id": "wamid.4"}]}

        send_dropoff_nudges.run()  # 30m and 20h are both due at +24h

        mock_send_message.assert_called()
        contact.refresh_from_db()
        self.assertIsNotNone(contact.dropoff_nudge_30m_sent_at)
        self.assertIsNotNone(contact.dropoff_nudge_20h_sent_at)


class HotLeadNudgeTests(TestCase):
    """Covers tasks.send_hot_lead_nudges — HOT_LEAD_NUDGE_STAGES' 2h/24h/72h
    schedule after a cold/warm -> hot flip, but only for a contact who never
    reached the review screen (review_reached_at is None); a hot lead who
    did is left to DROPOFF_NUDGE_STAGES' hot-only 30m/7d stages instead (see
    DropoffNudgeTests elsewhere in this file)."""

    def _hot_contact(self, phone, hours_ago, needs_human_review=False, **extra):
        contact = Contact.objects.create(
            phone_number=phone, language="en", lead_tier=Contact.LEAD_TIER_HOT,
            needs_human_review=needs_human_review, **extra,
        )
        contact.hot_lead_flipped_at = timezone.now() - timezone.timedelta(hours=hours_ago, minutes=5)
        contact.save(update_fields=["hot_lead_flipped_at"])
        return contact

    @patch("whatsapp_bot.tasks.send_whatsapp_buttons")
    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    def test_sends_the_2h_stage_with_cta_buttons(self, mock_send_message, mock_send_buttons):
        """Unlike the warm-lead schedule, every hot stage DOES attach the
        callback CTA buttons — a hot lead has shown real intent."""
        from .tasks import send_hot_lead_nudges

        contact = self._hot_contact("915555500080", hours_ago=2)
        mock_send_message.return_value = {"messages": [{"id": "wamid.30"}]}

        send_hot_lead_nudges.run()

        mock_send_message.assert_called_once()
        mock_send_buttons.assert_called_once()
        contact.refresh_from_db()
        self.assertIsNotNone(contact.hot_lead_nudge_2h_sent_at)
        self.assertIsNone(contact.hot_lead_nudge_24h_sent_at)

    @patch("whatsapp_bot.tasks.send_whatsapp_buttons")
    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    def test_progresses_through_all_three_stages(self, mock_send_message, mock_send_buttons):
        from .tasks import send_hot_lead_nudges

        contact = self._hot_contact("915555500081", hours_ago=2)
        mock_send_message.return_value = {"messages": [{"id": "wamid.a"}]}

        send_hot_lead_nudges.run()  # fires 2h
        contact.refresh_from_db()
        self.assertIsNotNone(contact.hot_lead_nudge_2h_sent_at)
        self.assertIsNone(contact.hot_lead_nudge_24h_sent_at)

        contact.hot_lead_flipped_at = timezone.now() - timezone.timedelta(hours=24, minutes=5)
        contact.save(update_fields=["hot_lead_flipped_at"])
        mock_send_message.reset_mock()
        mock_send_message.return_value = {"messages": [{"id": "wamid.b"}]}

        send_hot_lead_nudges.run()  # 2h already sent, fires 24h only
        mock_send_message.assert_called_once()
        contact.refresh_from_db()
        self.assertIsNotNone(contact.hot_lead_nudge_24h_sent_at)
        self.assertIsNone(contact.hot_lead_nudge_72h_sent_at)

        contact.hot_lead_flipped_at = timezone.now() - timezone.timedelta(hours=72, minutes=5)
        contact.save(update_fields=["hot_lead_flipped_at"])
        mock_send_message.reset_mock()
        mock_send_message.return_value = {"messages": [{"id": "wamid.c"}]}

        send_hot_lead_nudges.run()  # fires 72h only
        mock_send_message.assert_called_once()
        contact.refresh_from_db()
        self.assertIsNotNone(contact.hot_lead_nudge_72h_sent_at)

    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    def test_not_yet_due_before_the_2h_stage(self, mock_send_message):
        from .tasks import send_hot_lead_nudges

        contact = self._hot_contact("915555500082", hours_ago=0.5)  # well under 2h

        send_hot_lead_nudges.run()

        mock_send_message.assert_not_called()
        contact.refresh_from_db()
        self.assertIsNone(contact.hot_lead_nudge_2h_sent_at)

    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    def test_does_not_resend_a_stage_once_already_sent(self, mock_send_message):
        from .tasks import send_hot_lead_nudges

        contact = self._hot_contact("915555500083", hours_ago=2)
        contact.hot_lead_nudge_2h_sent_at = timezone.now()
        contact.save(update_fields=["hot_lead_nudge_2h_sent_at"])

        send_hot_lead_nudges.run()

        mock_send_message.assert_not_called()

    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    def test_skips_a_contact_who_already_reached_review(self, mock_send_message):
        """Left to DROPOFF_NUDGE_STAGES' hot-only 30m/7d stages instead, so
        the two schedules never double up on the same contact."""
        from .tasks import send_hot_lead_nudges

        self._hot_contact("915555500084", hours_ago=72, review_reached_at=timezone.now())

        send_hot_lead_nudges.run()

        mock_send_message.assert_not_called()

    @patch("whatsapp_bot.tasks.send_whatsapp_buttons")
    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    def test_does_not_skip_a_contact_flagged_for_staff_attention(
        self, mock_send_message, mock_send_buttons,
    ):
        """Opposite of the warm-lead schedule's gate: a hot flip sets
        needs_human_review itself (see notify_human_alert), so skipping on
        it being True would mean this task never fires at all."""
        from .tasks import send_hot_lead_nudges

        contact = self._hot_contact(
            "915555500085", hours_ago=2, needs_human_review=True,
        )
        mock_send_message.return_value = {"messages": [{"id": "wamid.31"}]}

        send_hot_lead_nudges.run()

        mock_send_message.assert_called_once()
        mock_send_buttons.assert_called_once()
        contact.refresh_from_db()
        self.assertIsNotNone(contact.hot_lead_nudge_2h_sent_at)

    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    def test_skips_a_contact_who_is_no_longer_hot(self, mock_send_message):
        from .tasks import send_hot_lead_nudges

        contact = self._hot_contact("915555500086", hours_ago=2)
        contact.lead_tier = Contact.LEAD_TIER_WARM
        contact.save(update_fields=["lead_tier"])

        send_hot_lead_nudges.run()

        mock_send_message.assert_not_called()

    @override_settings(WHATSAPP_HOT_LEAD_TEMPLATE_NAME="")
    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    def test_a_failed_send_with_no_template_configured_leaves_it_unstamped_to_retry(self, mock_send_message):
        from .tasks import send_hot_lead_nudges

        contact = self._hot_contact("915555500087", hours_ago=2)
        mock_send_message.return_value = None

        send_hot_lead_nudges.run()

        contact.refresh_from_db()
        self.assertIsNone(contact.hot_lead_nudge_2h_sent_at)

    @override_settings(WHATSAPP_HOT_LEAD_TEMPLATE_NAME="hot_lead_nudge", WHATSAPP_HOT_LEAD_TEMPLATE_LANG="en")
    @patch("whatsapp_bot.tasks.send_whatsapp_buttons")
    @patch("whatsapp_bot.tasks.send_whatsapp_template")
    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    def test_a_failed_send_falls_back_to_the_template_without_cta_buttons(
        self, mock_send_message, mock_send_template, mock_send_buttons,
    ):
        """No CTA buttons on the template-fallback path — an interactive
        message can't be sent outside the 24h window either, so there's
        nothing to attach it to."""
        from .tasks import send_hot_lead_nudges

        contact = self._hot_contact("915555500088", hours_ago=2)
        mock_send_message.return_value = None
        mock_send_template.return_value = {"messages": [{"id": "wamid.32"}]}

        send_hot_lead_nudges.run()

        mock_send_template.assert_called_once()
        mock_send_buttons.assert_not_called()
        contact.refresh_from_db()
        self.assertIsNotNone(contact.hot_lead_nudge_2h_sent_at)


class WarmLeadNudgeTests(TestCase):
    """Covers tasks.send_warm_lead_nudges — WARM_LEAD_NUDGE_STAGES' 12h/3d/7d
    schedule, one tier down from HotLeadNudgeTests: much longer fuse,
    lower-pressure copy at every stage, and the OPPOSITE needs_human_review
    gate direction (see that task's docstring)."""

    def _warm_contact(self, phone, hours_ago, needs_human_review=False, **extra):
        contact = Contact.objects.create(
            phone_number=phone, language="en", lead_tier=Contact.LEAD_TIER_WARM,
            needs_human_review=needs_human_review, **extra,
        )
        contact.warm_lead_flipped_at = timezone.now() - timezone.timedelta(hours=hours_ago, minutes=5)
        contact.save(update_fields=["warm_lead_flipped_at"])
        return contact

    @patch("whatsapp_bot.tasks.send_whatsapp_buttons")
    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    def test_sends_the_12h_stage_with_no_cta_buttons(self, mock_send_message, mock_send_buttons):
        """Unlike the hot-lead schedule, no stage attaches the callback CTA
        buttons — a 'book a call' push doesn't match the low-pressure tone."""
        from .tasks import send_warm_lead_nudges

        contact = self._warm_contact("915555500070", hours_ago=12)
        mock_send_message.return_value = {"messages": [{"id": "wamid.20"}]}

        send_warm_lead_nudges.run()

        mock_send_message.assert_called_once()
        mock_send_buttons.assert_not_called()
        contact.refresh_from_db()
        self.assertIsNotNone(contact.warm_lead_nudge_12h_sent_at)
        self.assertIsNone(contact.warm_lead_nudge_3d_sent_at)

    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    def test_progresses_through_all_three_stages(self, mock_send_message):
        from .tasks import send_warm_lead_nudges

        contact = self._warm_contact("915555500071", hours_ago=12)
        mock_send_message.return_value = {"messages": [{"id": "wamid.a"}]}

        send_warm_lead_nudges.run()  # fires 12h
        contact.refresh_from_db()
        self.assertIsNotNone(contact.warm_lead_nudge_12h_sent_at)

        contact.warm_lead_flipped_at = timezone.now() - timezone.timedelta(hours=24 * 3, minutes=5)
        contact.save(update_fields=["warm_lead_flipped_at"])
        mock_send_message.reset_mock()
        mock_send_message.return_value = {"messages": [{"id": "wamid.b"}]}

        send_warm_lead_nudges.run()  # fires 3d only
        mock_send_message.assert_called_once()
        contact.refresh_from_db()
        self.assertIsNotNone(contact.warm_lead_nudge_3d_sent_at)
        self.assertIsNone(contact.warm_lead_nudge_7d_sent_at)

        contact.warm_lead_flipped_at = timezone.now() - timezone.timedelta(hours=24 * 7, minutes=5)
        contact.save(update_fields=["warm_lead_flipped_at"])
        mock_send_message.reset_mock()
        mock_send_message.return_value = {"messages": [{"id": "wamid.c"}]}

        send_warm_lead_nudges.run()  # fires 7d only
        mock_send_message.assert_called_once()
        contact.refresh_from_db()
        self.assertIsNotNone(contact.warm_lead_nudge_7d_sent_at)

    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    def test_not_yet_due_before_the_12h_stage(self, mock_send_message):
        from .tasks import send_warm_lead_nudges

        contact = self._warm_contact("915555500072", hours_ago=1)  # well under 12h

        send_warm_lead_nudges.run()

        mock_send_message.assert_not_called()
        contact.refresh_from_db()
        self.assertIsNone(contact.warm_lead_nudge_12h_sent_at)

    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    def test_does_not_resend_a_stage_once_already_sent(self, mock_send_message):
        from .tasks import send_warm_lead_nudges

        contact = self._warm_contact("915555500073", hours_ago=12)
        contact.warm_lead_nudge_12h_sent_at = timezone.now()
        contact.save(update_fields=["warm_lead_nudge_12h_sent_at"])

        send_warm_lead_nudges.run()

        mock_send_message.assert_not_called()

    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    def test_skips_a_contact_currently_flagged_for_staff_attention(self, mock_send_message):
        """Opposite gate direction from the hot-lead schedule:
        needs_human_review True here means something else already needs a
        human's attention (bad CSAT, HELP request, an unresolved hot-lead
        alert) — piling a bot nudge on top would be the wrong move."""
        from .tasks import send_warm_lead_nudges

        self._warm_contact("915555500074", hours_ago=12, needs_human_review=True)

        send_warm_lead_nudges.run()

        mock_send_message.assert_not_called()

    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    def test_skips_a_contact_who_is_no_longer_warm(self, mock_send_message):
        from .tasks import send_warm_lead_nudges

        contact = self._warm_contact("915555500075", hours_ago=12)
        contact.lead_tier = Contact.LEAD_TIER_COLD
        contact.save(update_fields=["lead_tier"])

        send_warm_lead_nudges.run()

        mock_send_message.assert_not_called()

    @override_settings(WHATSAPP_WARM_LEAD_TEMPLATE_NAME="")
    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    def test_a_failed_send_with_no_template_configured_leaves_it_unstamped_to_retry(self, mock_send_message):
        from .tasks import send_warm_lead_nudges

        contact = self._warm_contact("915555500076", hours_ago=12)
        mock_send_message.return_value = None

        send_warm_lead_nudges.run()

        contact.refresh_from_db()
        self.assertIsNone(contact.warm_lead_nudge_12h_sent_at)

    @override_settings(WHATSAPP_WARM_LEAD_TEMPLATE_NAME="warm_lead_nudge", WHATSAPP_WARM_LEAD_TEMPLATE_LANG="en")
    @patch("whatsapp_bot.tasks.send_whatsapp_template")
    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    def test_a_failed_send_falls_back_to_the_template_and_stamps_on_success(
        self, mock_send_message, mock_send_template,
    ):
        from .tasks import send_warm_lead_nudges

        contact = self._warm_contact("915555500077", hours_ago=12)
        mock_send_message.return_value = None
        mock_send_template.return_value = {"messages": [{"id": "wamid.21"}]}

        send_warm_lead_nudges.run()

        mock_send_template.assert_called_once()
        contact.refresh_from_db()
        self.assertIsNotNone(contact.warm_lead_nudge_12h_sent_at)


class HumanAlertSlackTests(TestCase):
    """Covers services.notify_human_alert's Slack Incoming Webhook post
    (_post_slack_alert) — the admin-flag/log-line behavior is unconditional
    and unchanged; the Slack post is the new, optional part."""

    def _contact(self, phone, **extra):
        return Contact.objects.create(phone_number=phone, language="en", **extra)

    @override_settings(SLACK_HOT_LEAD_WEBHOOK_URL="")
    @patch("whatsapp_bot.services.requests.post")
    def test_no_webhook_configured_does_not_post(self, mock_post):
        contact = self._contact("915555500090")

        notify_human_alert(contact, "hot_lead", detail="score=65")

        mock_post.assert_not_called()
        contact.refresh_from_db()
        self.assertTrue(contact.needs_human_review)
        self.assertEqual(contact.needs_human_review_reason, "hot_lead")

    @override_settings(SLACK_HOT_LEAD_WEBHOOK_URL="https://hooks.slack.com/services/test")
    @patch("whatsapp_bot.services.requests.post")
    def test_webhook_configured_posts_with_contact_context(self, mock_post):
        mock_post.return_value.raise_for_status.return_value = None
        contact = self._contact(
            "915555500091", name="Priya Sharma",
            lead_tier=Contact.LEAD_TIER_HOT, lead_score=72,
        )

        notify_human_alert(contact, "hot_lead", detail="score=72")

        mock_post.assert_called_once()
        call_args, call_kwargs = mock_post.call_args
        self.assertEqual(call_args[0], "https://hooks.slack.com/services/test")
        text = call_kwargs["json"]["text"]
        self.assertIn("Priya Sharma", text)
        self.assertIn("915555500091", text)
        self.assertIn("hot", text)
        self.assertIn("score=72", text)

    @override_settings(SLACK_HOT_LEAD_WEBHOOK_URL="https://hooks.slack.com/services/test")
    @patch("whatsapp_bot.services.requests.post")
    def test_a_failed_post_does_not_raise_or_block_the_admin_flag(self, mock_post):
        mock_post.side_effect = requests.RequestException("boom")
        contact = self._contact("915555500092")

        notify_human_alert(contact, "bad_csat")  # must not raise

        contact.refresh_from_db()
        self.assertTrue(contact.needs_human_review)


class LeadTierFlipTimestampTests(TestCase):
    """Covers services.update_lead_score stamping hot_lead_flipped_at /
    warm_lead_flipped_at on the respective tier transitions — the trigger
    side of both nudge tasks above."""

    def test_flip_to_warm_stamps_warm_timestamp_only(self):
        from .services import update_lead_score

        # STEP_PRICING_SHOWN alone is worth 35 (see compute_lead_score's
        # legacy-wizard fallback branch) — comfortably in the 30-59 warm
        # band, short of hot's 60+ threshold.
        contact = Contact.objects.create(
            phone_number="915555500077", conversation_step=Contact.STEP_PRICING_SHOWN,
        )

        update_lead_score(contact)

        contact.refresh_from_db()
        self.assertEqual(contact.lead_tier, Contact.LEAD_TIER_WARM)
        self.assertIsNotNone(contact.warm_lead_flipped_at)
        self.assertIsNone(contact.hot_lead_flipped_at)

    def test_flip_to_hot_stamps_hot_timestamp_only(self):
        from .services import update_lead_score

        # 35 (STEP_PRICING_SHOWN) + 25 (has_requested_help) = 60, exactly
        # hot's threshold.
        contact = Contact.objects.create(
            phone_number="915555500078", conversation_step=Contact.STEP_PRICING_SHOWN,
            has_requested_help=True,
        )

        update_lead_score(contact)

        contact.refresh_from_db()
        self.assertEqual(contact.lead_tier, Contact.LEAD_TIER_HOT)
        self.assertIsNotNone(contact.hot_lead_flipped_at)
        self.assertIsNone(contact.warm_lead_flipped_at)

    def test_staying_in_the_same_tier_does_not_restamp(self):
        """A later message that keeps a contact hot (score changes but tier
        doesn't) must not reset hot_lead_flipped_at — that would let a
        contact who never actually goes quiet keep dodging the nudge."""
        from .services import update_lead_score

        contact = Contact.objects.create(
            phone_number="915555500079", conversation_step=Contact.STEP_PRICING_SHOWN,
            has_requested_help=True,
        )
        update_lead_score(contact)
        contact.refresh_from_db()
        first_flip = contact.hot_lead_flipped_at
        self.assertIsNotNone(first_flip)

        Message.objects.create(contact=contact, direction="in", body="another message")
        update_lead_score(contact)

        contact.refresh_from_db()
        self.assertEqual(contact.lead_tier, Contact.LEAD_TIER_HOT)
        self.assertEqual(contact.hot_lead_flipped_at, first_flip)


class ReferralAskTests(TestCase):
    """Covers tasks.send_referral_ask / _maybe_schedule_referral_ask — the
    post-booking referral message, sent once per REFERRAL_ASK_COOLDOWN_DAYS."""

    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    def test_sends_and_stamps_when_never_sent_before(self, mock_send_message):
        from .tasks import send_referral_ask

        contact = Contact.objects.create(phone_number="915555500050", language="en")

        send_referral_ask.run(contact.id)

        mock_send_message.assert_called_once()
        contact.refresh_from_db()
        self.assertIsNotNone(contact.referral_ask_sent_at)

    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    def test_skips_within_the_cooldown_window(self, mock_send_message):
        from .tasks import send_referral_ask

        contact = Contact.objects.create(
            phone_number="915555500051", language="en",
            referral_ask_sent_at=timezone.now() - timezone.timedelta(days=5),
        )

        send_referral_ask.run(contact.id)

        mock_send_message.assert_not_called()

    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    @patch("whatsapp_bot.tasks.send_whatsapp_buttons")
    @patch("whatsapp_bot.tasks.send_whatsapp_list")
    @patch("whatsapp_bot.tasks.send_referral_ask.apply_async")
    def test_confirming_the_legacy_wizard_schedules_the_referral_ask(
        self, mock_apply_async, mock_send_list, mock_send_buttons, mock_send_message,
    ):
        from .tasks import process_incoming_message

        contact = Contact.objects.create(
            phone_number="915555500052", language="en",
            conversation_step=Contact.STEP_SUBSERVICE_SHOWN, pending_category_code="itr",
        )
        with override_settings(WHATSAPP_SERVICE_FLOW_ID=""):
            process_incoming_message.run(contact.id, "itr_salary")
            process_incoming_message.run(contact.id, "done")
            process_incoming_message.run(contact.id, "confirm")

        mock_apply_async.assert_called_once_with((contact.id,), countdown=3600)


@override_settings(RAZORPAY_KEY_ID="test-key", RAZORPAY_KEY_SECRET="test-secret")
class PaymentCtaGatingTests(TestCase):
    """Covers services.maybe_send_payment_cta — a customer who just checked
    out and was quoted a total always gets the pay link, regardless of
    needs_human_review (bad CSAT / HELP request / hot-lead flip — none of
    them hold the link back, the human path just runs alongside it); only
    the hard gates (no amount, Razorpay not configured) skip it."""

    def _contact(self, **kwargs):
        kwargs.setdefault("last_quote_amount", 1000)
        kwargs.setdefault("last_quote_is_variable", False)
        return Contact.objects.create(phone_number="915555599900", language="en", **kwargs)

    @patch("whatsapp_bot.services.send_whatsapp_cta_url")
    @patch("whatsapp_bot.services.payments.create_payment_link", return_value=True)
    def test_hot_lead_gets_the_link(self, mock_create_link, mock_send_cta):
        from .services import maybe_send_payment_cta

        contact = self._contact(needs_human_review=True, needs_human_review_reason="hot_lead")

        maybe_send_payment_cta(contact)

        mock_create_link.assert_called_once()
        mock_send_cta.assert_called_once()

    @patch("whatsapp_bot.services.send_whatsapp_cta_url")
    @patch("whatsapp_bot.services.payments.create_payment_link", return_value=True)
    def test_bad_csat_still_gets_the_link(self, mock_create_link, mock_send_cta):
        from .services import maybe_send_payment_cta

        contact = self._contact(needs_human_review=True, needs_human_review_reason="bad_csat")

        maybe_send_payment_cta(contact)

        mock_create_link.assert_called_once()
        mock_send_cta.assert_called_once()

    @patch("whatsapp_bot.services.send_whatsapp_cta_url")
    @patch("whatsapp_bot.services.payments.create_payment_link", return_value=True)
    def test_help_requested_still_gets_the_link(self, mock_create_link, mock_send_cta):
        from .services import maybe_send_payment_cta

        contact = self._contact(needs_human_review=True, needs_human_review_reason="help_requested")

        maybe_send_payment_cta(contact)

        mock_create_link.assert_called_once()
        mock_send_cta.assert_called_once()

    @patch("whatsapp_bot.services.send_whatsapp_cta_url")
    @patch("whatsapp_bot.services.payments.create_payment_link", return_value=True)
    def test_no_review_flag_sends_the_link_as_before(self, mock_create_link, mock_send_cta):
        from .services import maybe_send_payment_cta

        contact = self._contact()

        maybe_send_payment_cta(contact)

        mock_create_link.assert_called_once()
        mock_send_cta.assert_called_once()

    @patch("whatsapp_bot.services.send_whatsapp_cta_url")
    @patch("whatsapp_bot.services.payments.create_payment_link", return_value=True)
    def test_no_amount_skips_the_link(self, mock_create_link, mock_send_cta):
        from .services import maybe_send_payment_cta

        contact = self._contact(last_quote_amount=None)

        maybe_send_payment_cta(contact)

        mock_create_link.assert_not_called()
        mock_send_cta.assert_not_called()

    @override_settings(RAZORPAY_KEY_ID="", RAZORPAY_KEY_SECRET="")
    @patch("whatsapp_bot.services.send_whatsapp_cta_url")
    @patch("whatsapp_bot.services.payments.create_payment_link", return_value=True)
    def test_razorpay_not_configured_skips_the_link(self, mock_create_link, mock_send_cta):
        from .services import maybe_send_payment_cta

        contact = self._contact()

        maybe_send_payment_cta(contact)

        mock_create_link.assert_not_called()
        mock_send_cta.assert_not_called()

    @patch("whatsapp_bot.services.send_whatsapp_cta_url")
    @patch("whatsapp_bot.services.payments.create_payment_link", return_value=True)
    def test_summary_is_folded_into_the_cta_body_not_sent_separately(self, mock_create_link, mock_send_cta):
        """checkout-confirm should produce exactly one outbound WhatsApp
        message, not a plain-text summary followed by a separate CTA — see
        tasks.process_incoming_message's trigger_conversion_cta handling."""
        from .services import maybe_send_payment_cta

        contact = self._contact()

        sent = maybe_send_payment_cta(contact, summary="Your selected income details:\n• ITR Salary Filing")

        self.assertTrue(sent)
        mock_send_cta.assert_called_once()
        body = mock_send_cta.call_args.args[1]
        self.assertIn("ITR Salary Filing", body)
        self.assertIn("Pay", body)

    @override_settings(WHATSAPP_SERVICE_FLOW_ID="")
    @patch("whatsapp_bot.services.send_whatsapp_cta_url")
    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    @patch("whatsapp_bot.tasks.send_whatsapp_buttons")
    @patch("whatsapp_bot.services.payments.create_payment_link", return_value=True)
    def test_confirm_sends_one_merged_message_when_a_payment_link_is_available(
        self, mock_create_link, mock_send_buttons, mock_send_message, mock_send_cta,
    ):
        """Full confirm flow, Razorpay configured (class-level override):
        exactly one outbound message (the merged summary+CTA), no separate
        plain-text summary and no callback/support ask."""
        from .tasks import process_incoming_message

        contact = Contact.objects.create(
            phone_number="915555599901",
            language="en",
            conversation_step=Contact.STEP_SUBSERVICE_SHOWN,
            pending_category_code="itr",
        )
        process_incoming_message.run(contact.id, "itr_futures_options")
        process_incoming_message.run(contact.id, "confirm")

        mock_send_cta.assert_called_once()
        self.assertIn("Futures & Options ITR", mock_send_cta.call_args.args[1])
        mock_send_message.assert_not_called()


@override_settings(RAZORPAY_KEY_ID="test-key", RAZORPAY_KEY_SECRET="test-secret")
class InvalidateStaleOrdersTests(TestCase):
    """Covers services.invalidate_stale_orders directly — called after a
    cart edit (see tasks.py's edit_cart handling) so an already-issued
    payment link can never be paid at a now-wrong amount."""

    def _contact_with_open_order(self, phone):
        contact = Contact.objects.create(
            phone_number=phone, language="en", last_quote_amount=999, last_quote_is_variable=False,
        )
        order = Order.objects.create(contact=contact, amount=999, payment_link_id="plink_abc123")
        return contact, order

    @patch("whatsapp_bot.services.payments.cancel_payment_link", return_value=True)
    def test_expires_locally_and_cancels_on_the_gateway(self, mock_cancel):
        from .services import invalidate_stale_orders

        contact, order = self._contact_with_open_order("915555599910")

        invalidate_stale_orders(contact)

        mock_cancel.assert_called_once_with(order)
        order.refresh_from_db()
        self.assertEqual(order.status, Order.STATUS_EXPIRED)
        self.assertEqual(order.failure_reason, "cart_edited")

    @patch("whatsapp_bot.services.payments.cancel_payment_link", return_value=False)
    def test_still_expires_locally_even_if_the_gateway_call_fails(self, mock_cancel):
        """payments.cancel_payment_link never raises, but a False return
        (network/gateway failure) must still leave the order unpayable
        locally — see maybe_send_payment_cta's exact-snapshot reuse check,
        which must never pick this order back up."""
        from .services import invalidate_stale_orders

        contact, order = self._contact_with_open_order("915555599911")

        invalidate_stale_orders(contact)

        order.refresh_from_db()
        self.assertEqual(order.status, Order.STATUS_EXPIRED)

    @patch("whatsapp_bot.services.payments.cancel_payment_link", return_value=True)
    def test_leaves_paid_and_other_contacts_orders_alone(self, mock_cancel):
        from .services import invalidate_stale_orders

        contact, open_order = self._contact_with_open_order("915555599912")
        paid_order = Order.objects.create(
            contact=contact, amount=500, status=Order.STATUS_PAID, payment_link_id="plink_paid",
        )
        other_contact, other_order = self._contact_with_open_order("915555599913")

        invalidate_stale_orders(contact)

        mock_cancel.assert_called_once_with(open_order)
        paid_order.refresh_from_db()
        self.assertEqual(paid_order.status, Order.STATUS_PAID)
        other_order.refresh_from_db()
        self.assertEqual(other_order.status, Order.STATUS_CREATED)


class PaymentWebhookStaleOrderTests(TestCase):
    """Covers tasks.process_payment_webhook's handling of a payment landing
    on an order services.invalidate_stale_orders already expired — the
    money is still real (recorded as paid), but flagged for a human rather
    than silently treated as an ordinary payment."""

    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    @patch("whatsapp_bot.tasks.notify_human_alert")
    def test_payment_on_a_cart_edited_order_is_recorded_and_flagged(self, mock_alert, mock_send_message):
        from .tasks import process_payment_webhook

        contact = Contact.objects.create(phone_number="915555599920", language="en")
        order = Order.objects.create(
            contact=contact, amount=999, payment_link_id="plink_stale",
            status=Order.STATUS_EXPIRED, failure_reason="cart_edited",
        )

        process_payment_webhook.run("payment_link.paid", "plink_stale", "pay_xyz", {})

        order.refresh_from_db()
        self.assertEqual(order.status, Order.STATUS_PAID)
        mock_send_message.assert_called_once()  # customer still gets the thank-you
        mock_alert.assert_called_once()
        self.assertEqual(mock_alert.call_args.args[1], "stale_link_paid")

    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    @patch("whatsapp_bot.tasks.notify_human_alert")
    def test_ordinary_payment_is_not_flagged(self, mock_alert, mock_send_message):
        from .tasks import process_payment_webhook

        contact = Contact.objects.create(phone_number="915555599921", language="en")
        Order.objects.create(contact=contact, amount=999, payment_link_id="plink_ok")

        process_payment_webhook.run("payment_link.paid", "plink_ok", "pay_abc", {})

        mock_send_message.assert_called_once()
        mock_alert.assert_not_called()


@override_settings(
    WHATSAPP_ADDON_FLOW_ID="", WHATSAPP_SERVICE_FLOW_ID="",
    RAZORPAY_KEY_ID="", RAZORPAY_KEY_SECRET="",
)
class ConfirmScreenGapFixesTests(TestCase):
    """Covers three gaps found in a review of the confirm-screen state
    machine: STEP_CONFIRMING_QUOTE's Cancel used to wipe the WHOLE cart
    (correct only back when this screen could never hold more than one
    just-quoted item); both confirm screens looped the identical prompt on
    any unrecognized reply instead of falling through to free-chat/edit_cart;
    and STEP_CONFIRMING_SELECTION's Edit tap silently became Confirm if the
    parent service had since been deactivated."""

    @patch("whatsapp_bot.tasks.send_whatsapp_list")
    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    def test_cancel_on_a_standalone_quote_drops_just_that_item(self, mock_send_message, mock_send_list):
        from .tasks import process_incoming_message

        contact = Contact.objects.create(
            phone_number="915555599930", language="en",
            conversation_step=Contact.STEP_SUBSERVICE_SHOWN, pending_category_code="itr",
        )
        process_incoming_message.run(contact.id, "itr_futures_options")
        process_incoming_message.run(contact.id, "cancel")

        contact.refresh_from_db()
        self.assertEqual(contact.selected_services, [])
        self.assertEqual(contact.conversation_step, Contact.STEP_PRICING_SHOWN)

    @patch("whatsapp_bot.tasks.classify_and_respond")
    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    def test_cancel_after_a_cart_edit_leaves_the_rest_of_the_cart_alone(self, mock_send_message, mock_classify):
        """The regression this session's cart-edit feature introduced: Cancel
        must undo only the just-made edit, not the whole multi-item cart."""
        from .services import add_service_selection
        from .tasks import process_incoming_message

        contact = Contact.objects.create(
            phone_number="915555599931", language="en", conversation_step=Contact.STEP_PRICING_SHOWN,
        )
        salary = ServicePricing.objects.get(service_code="itr_salary")
        capital_gains = ServicePricing.objects.get(service_code="itr_capital_gains")
        add_service_selection(contact, salary)
        add_service_selection(contact, capital_gains)

        mock_classify.return_value = {
            "reply": "Sure, one sec.", "action": "edit_cart", "flow_type": None,
            "cart_edit_operation": "add", "cart_edit_query": "house property", "confidence": "high",
        }
        process_incoming_message.run(contact.id, "can you add house property too")

        process_incoming_message.run(contact.id, "cancel")

        contact.refresh_from_db()
        codes = {item["service_code"] for item in contact.selected_services}
        self.assertEqual(codes, {"itr_salary", "itr_capital_gains", "itr_house_property"})
        self.assertEqual(contact.conversation_step, Contact.STEP_PRICING_SHOWN)

    @patch("whatsapp_bot.tasks.classify_and_respond")
    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    def test_unparsed_text_at_confirm_quote_reaches_edit_cart_instead_of_looping(
        self, mock_send_message, mock_classify,
    ):
        from .tasks import process_incoming_message

        contact = Contact.objects.create(
            phone_number="915555599932", language="en",
            conversation_step=Contact.STEP_SUBSERVICE_SHOWN, pending_category_code="itr",
        )
        process_incoming_message.run(contact.id, "itr_futures_options")

        mock_classify.return_value = {
            "reply": "Sure, one sec.", "action": "edit_cart", "flow_type": None,
            "cart_edit_operation": "add", "cart_edit_query": "house property", "confidence": "high",
        }
        process_incoming_message.run(contact.id, "oh wait can you also add house property")

        mock_classify.assert_called_once()
        contact.refresh_from_db()
        codes = {item["service_code"] for item in contact.selected_services}
        self.assertIn("itr_house_property", codes)
        body = mock_send_message.call_args.args[1]
        self.assertNotIn("Ready to go?", body)  # not the re-shown Confirm/Cancel prompt

    @patch("whatsapp_bot.tasks.classify_and_respond")
    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    def test_unparsed_text_at_confirm_selection_answers_instead_of_looping(
        self, mock_send_message, mock_classify,
    ):
        from .tasks import process_incoming_message

        contact = Contact.objects.create(
            phone_number="915555599933", language="en",
            conversation_step=Contact.STEP_SUBSERVICE_SHOWN, pending_category_code="itr",
        )
        process_incoming_message.run(contact.id, "itr_salary")
        process_incoming_message.run(contact.id, "done")  # -> STEP_CONFIRMING_SELECTION

        mock_classify.return_value = {
            "reply": "ITR filing is usually due by July 31st.", "action": "continue_chat",
            "flow_type": None, "cart_edit_operation": None, "cart_edit_query": "", "confidence": "low",
        }
        process_incoming_message.run(contact.id, "what's the ITR deadline?")

        mock_classify.assert_called_once()
        mock_send_message.assert_called_once_with("915555599933", "ITR filing is usually due by July 31st.")
        contact.refresh_from_db()
        # Still sitting on the same confirm screen — a later "confirm" must
        # still resolve correctly.
        self.assertEqual(contact.conversation_step, Contact.STEP_CONFIRMING_SELECTION)

    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    def test_edit_tap_with_a_deactivated_parent_finalizes_with_an_explanation(self, mock_send_message):
        """Past behavior: an Edit tap here silently became Confirm with no
        explanation once parent_service stopped resolving. Now it must still
        finalize (nothing left to edit back into) but say so explicitly."""
        from .tasks import process_incoming_message

        contact = Contact.objects.create(
            phone_number="915555599934", language="en",
            conversation_step=Contact.STEP_CONFIRMING_SELECTION,
            pending_parent_service_code="itr_salary",
        )
        from .services import add_service_selection
        add_service_selection(contact, ServicePricing.objects.get(service_code="itr_capital_gains"))
        # Simulate the parent service being deactivated mid-conversation.
        ServicePricing.objects.filter(service_code="itr_salary").update(is_active=False)

        process_incoming_message.run(contact.id, "edit")

        contact.refresh_from_db()
        self.assertEqual(contact.conversation_step, Contact.STEP_PRICING_SHOWN)
        body = mock_send_message.call_args.args[1]
        self.assertIn("no longer available", body)
        ServicePricing.objects.filter(service_code="itr_salary").update(is_active=True)


class WhatsappWebhookDedupeTests(TestCase):
    """Covers views.whatsapp_webhook's wa_message_id dedup check — without
    it, a retried/duplicate Meta webhook delivery would silently re-run the
    whole turn: a second real Gemini API call, a second reply sent to the
    customer, and (for an edit_cart/payment turn) a second cart mutation or
    Order. This is the actual reckless-spend vector in the app, not the
    classify_and_respond call itself (that's already exactly one call per
    genuine inbound message)."""

    def _payload(self, wa_message_id, text="hello", phone="915555599940"):
        return {
            "entry": [{"changes": [{"value": {"messages": [{
                "from": phone, "id": wa_message_id, "type": "text",
                "text": {"body": text},
            }]}}]}],
        }

    @patch("whatsapp_bot.views.process_incoming_message")
    def test_first_delivery_creates_the_message_and_enqueues_the_task(self, mock_task):
        response = Client().post(
            "/api/whatsapp/webhook/",
            data=json.dumps(self._payload("wamid.first")),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(Message.objects.filter(whatsapp_message_id="wamid.first").count(), 1)
        mock_task.delay.assert_called_once()

    @patch("whatsapp_bot.views.process_incoming_message")
    def test_duplicate_delivery_is_skipped_entirely(self, mock_task):
        Client().post(
            "/api/whatsapp/webhook/",
            data=json.dumps(self._payload("wamid.dupe")),
            content_type="application/json",
        )
        mock_task.delay.reset_mock()

        response = Client().post(
            "/api/whatsapp/webhook/",
            data=json.dumps(self._payload("wamid.dupe")),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        # Still exactly one Message row for this wa_message_id — the retry
        # created nothing new.
        self.assertEqual(Message.objects.filter(whatsapp_message_id="wamid.dupe").count(), 1)
        # And, the actual point: no second Gemini-call-bearing task enqueued.
        mock_task.delay.assert_not_called()

    @patch("whatsapp_bot.views.process_incoming_message")
    def test_different_messages_are_not_treated_as_duplicates(self, mock_task):
        Client().post(
            "/api/whatsapp/webhook/",
            data=json.dumps(self._payload("wamid.a", text="first")),
            content_type="application/json",
        )
        Client().post(
            "/api/whatsapp/webhook/",
            data=json.dumps(self._payload("wamid.b", text="second")),
            content_type="application/json",
        )

        self.assertEqual(mock_task.delay.call_count, 2)


class WhatsappWebhookProfileNameTests(TestCase):
    """Covers views.whatsapp_webhook's contacts[0].profile.name extraction —
    Contact.whatsapp_display_name capture, opportunistic resync on rename,
    tolerance of a missing/absent contacts array, and that the captured name
    actually reaches a customer-facing nudge (see services._named_opener /
    tasks.build_cart_nudge_message)."""

    def _payload(self, wa_message_id, phone="915555599950", text="hi", profile_name=None):
        value = {"messages": [{
            "from": phone, "id": wa_message_id, "type": "text",
            "text": {"body": text},
        }]}
        if profile_name is not None:
            value["contacts"] = [{"profile": {"name": profile_name}}]
        return {"entry": [{"changes": [{"value": value}]}]}

    @patch("whatsapp_bot.views.process_incoming_message")
    def test_profile_name_is_captured_onto_the_contact(self, mock_task):
        Client().post(
            "/api/whatsapp/webhook/",
            data=json.dumps(self._payload("wamid.name1", profile_name="Priya Sharma")),
            content_type="application/json",
        )

        contact = Contact.objects.get(phone_number="915555599950")
        self.assertEqual(contact.whatsapp_display_name, "Priya Sharma")

    @patch("whatsapp_bot.views.process_incoming_message")
    def test_a_later_delivery_with_a_changed_name_resyncs_it(self, mock_task):
        Client().post(
            "/api/whatsapp/webhook/",
            data=json.dumps(self._payload("wamid.name2", profile_name="Priya Sharma")),
            content_type="application/json",
        )
        Client().post(
            "/api/whatsapp/webhook/",
            data=json.dumps(self._payload("wamid.name3", profile_name="Priya S.")),
            content_type="application/json",
        )

        contact = Contact.objects.get(phone_number="915555599950")
        self.assertEqual(contact.whatsapp_display_name, "Priya S.")

    @patch("whatsapp_bot.views.process_incoming_message")
    def test_missing_contacts_array_leaves_an_existing_name_untouched(self, mock_task):
        Client().post(
            "/api/whatsapp/webhook/",
            data=json.dumps(self._payload("wamid.name4", profile_name="Priya Sharma")),
            content_type="application/json",
        )
        # Second delivery has no `contacts` at all — a real payload-shape
        # surprise this must survive without wiping the name already on file.
        Client().post(
            "/api/whatsapp/webhook/",
            data=json.dumps(self._payload("wamid.name5")),
            content_type="application/json",
        )

        contact = Contact.objects.get(phone_number="915555599950")
        self.assertEqual(contact.whatsapp_display_name, "Priya Sharma")

    def test_captured_name_personalizes_the_abandoned_cart_nudge(self):
        from .services import build_cart_nudge_message

        contact = Contact.objects.create(
            phone_number="915555599951", whatsapp_display_name="Priya Sharma",
        )
        message = build_cart_nudge_message(contact, 2)

        self.assertTrue(message.startswith("Hi Priya,"))

    def test_no_name_on_file_falls_back_to_the_unnamed_opener(self):
        from .services import build_cart_nudge_message

        contact = Contact.objects.create(phone_number="915555599952")
        message = build_cart_nudge_message(contact, 2)

        self.assertFalse(message.startswith("Hi "))

    def test_the_welcome_message_never_uses_the_captured_name_even_when_known(self):
        """Deliberately different from the nudge builders above — see
        _greeting_prefix's docstring for why the very first message a
        contact sees stays name-free (plain company greeting) even for a
        returning contact whose whatsapp_display_name is already on file,
        while later follow-ups (nudges) do personalize by name."""
        from .services import build_welcome_buttons_body, build_welcome_message

        contact = Contact.objects.create(
            phone_number="915555599953", whatsapp_display_name="Priya Sharma",
        )

        self.assertTrue(build_welcome_message(contact).startswith("👋 Welcome to TaxPlan Advisors!"))
        self.assertTrue(build_welcome_buttons_body(contact).startswith("👋 Welcome to TaxPlan Advisors!"))
        self.assertNotIn("Priya", build_welcome_message(contact))
        self.assertNotIn("Priya", build_welcome_buttons_body(contact))


class HumanHandoffPauseTests(TestCase):
    """The human-handoff pause feature (Contact.bot_paused_until silencing
    the bot after HELP or a completed callback booking) was removed by
    explicit request — tasks.process_incoming_message no longer gates
    auto-replies on it, and nothing sets it anymore. These tests now cover
    that removal directly: a handoff must still flag the contact for staff
    (see notify_human_alert) but must never again leave the customer
    without a reply. The remaining bot_paused_until-clearing behavior
    (RESTART, the "Resume bot" admin action) is still exercised below since
    it's still correct cleanup for any contact carrying a legacy value from
    before this change. See also FullCutoverTests'
    test_help_keyword_still_works_and_does_not_open_the_flow, which covers
    the HELP reply itself and isn't touched by this."""

    @patch("whatsapp_bot.tasks.send_service_flow")
    def test_help_no_longer_pauses_the_bot_and_the_next_message_still_gets_a_reply(self, mock_send_flow):
        from .tasks import process_incoming_message

        contact = Contact.objects.create(phone_number="915555599960", language="en")
        process_incoming_message.run(contact.id, "HELP")

        contact.refresh_from_db()
        self.assertIsNone(contact.bot_paused_until)

        message_count_before = contact.messages.count()
        process_incoming_message.run(contact.id, "what's the ITR deadline?")
        self.assertGreater(contact.messages.count(), message_count_before)  # still replied

    @patch("whatsapp_bot.tasks.send_whatsapp_buttons")
    @patch("whatsapp_bot.tasks.send_whatsapp_message")
    @patch("whatsapp_bot.tasks.send_whatsapp_list")
    def test_completing_a_callback_booking_no_longer_pauses_the_bot(
        self, mock_send_list, mock_send_message, mock_send_buttons,
    ):
        from .tasks import process_incoming_message

        contact = Contact.objects.create(
            phone_number="915555599961", language="en",
            conversation_step=Contact.STEP_SUBSERVICE_SHOWN, pending_category_code="itr",
        )
        process_incoming_message.run(contact.id, "itr_salary")
        process_incoming_message.run(contact.id, "done")
        process_incoming_message.run(contact.id, "confirm")
        process_incoming_message.run(contact.id, "book_callback")
        contact.refresh_from_db()
        self.assertIsNone(contact.bot_paused_until)

        process_incoming_message.run(contact.id, "slot_tomorrow")

        contact.refresh_from_db()
        self.assertIsNone(contact.bot_paused_until)

    def test_restart_still_clears_a_legacy_pause_value(self):
        from .tasks import process_incoming_message

        contact = Contact.objects.create(
            phone_number="915555599962", language="en",
            bot_paused_until=timezone.now() + timedelta(hours=4),
        )
        # RESTART re-asks language via the chat list (see tasks.py's
        # language gate) rather than reopening the Flow directly now — not
        # what this test is about, so no send_service_flow mock/assertion
        # here; just that the legacy pause value is still cleared.
        process_incoming_message.run(contact.id, "RESTART")

        contact.refresh_from_db()
        self.assertIsNone(contact.bot_paused_until)

    def test_a_legacy_pause_value_no_longer_blocks_replies(self):
        """Even an unexpired bot_paused_until (set before this feature was
        removed) must not block replies anymore — the gate itself is gone,
        not just relaxed to check expiry."""
        from .tasks import process_incoming_message

        contact = Contact.objects.create(
            phone_number="915555599963", language="en",
            bot_paused_until=timezone.now() + timedelta(hours=4),
        )
        message_count_before = contact.messages.count()
        process_incoming_message.run(contact.id, "what's the ITR deadline?")
        self.assertGreater(contact.messages.count(), message_count_before)

    def test_resume_bot_admin_action_clears_pause_and_review_flag(self):
        from django.contrib.admin.sites import AdminSite
        from django.contrib.messages.storage.fallback import FallbackStorage
        from django.test import RequestFactory

        from .admin import ContactAdmin

        contact = Contact.objects.create(
            phone_number="915555599964", language="en",
            bot_paused_until=timezone.now() + timedelta(hours=4),
            needs_human_review=True, needs_human_review_reason="help_requested",
        )
        request = RequestFactory().post("/admin/whatsapp_bot/contact/")
        request.session = {}  # FallbackStorage's session backend just needs a dict-like session
        request._messages = FallbackStorage(request)  # message_user needs somewhere to write to
        admin_instance = ContactAdmin(Contact, AdminSite())
        admin_instance.resume_bot(request, Contact.objects.filter(id=contact.id))

        contact.refresh_from_db()
        self.assertIsNone(contact.bot_paused_until)
        self.assertFalse(contact.needs_human_review)
        self.assertEqual(contact.needs_human_review_reason, "")


class OrderStatusChatTests(TestCase):
    """Covers the STATUS keyword (services.build_order_status_reply and
    friends) — a pure DB read against Order.status, no LLM involved."""

    @patch("whatsapp_bot.tasks.send_whatsapp_buttons")
    def test_no_orders_offers_a_human_handoff_button(self, mock_send_buttons):
        from .tasks import process_incoming_message
        from .services import HELP_KEYWORD

        contact = Contact.objects.create(phone_number="915555599970", language="en")
        process_incoming_message.run(contact.id, "status")

        options = mock_send_buttons.call_args.args[2]
        self.assertEqual([opt[0] for opt in options], [HELP_KEYWORD])

    @patch("whatsapp_bot.tasks.send_whatsapp_buttons")
    def test_one_order_shows_status_and_a_resend_button_when_unpaid(self, mock_send_buttons):
        from .tasks import process_incoming_message
        from .services import ORDER_STATUS_RESEND_PREFIX

        contact = Contact.objects.create(phone_number="915555599971", language="en")
        order = Order.objects.create(contact=contact, amount=1500, status=Order.STATUS_CREATED)
        process_incoming_message.run(contact.id, "status")

        options = mock_send_buttons.call_args.args[2]
        self.assertIn(f"{ORDER_STATUS_RESEND_PREFIX}{order.id}", [opt[0] for opt in options])

    @patch("whatsapp_bot.tasks.send_whatsapp_buttons")
    def test_a_paid_order_never_offers_the_resend_button(self, mock_send_buttons):
        from .tasks import process_incoming_message

        contact = Contact.objects.create(phone_number="915555599972", language="en")
        Order.objects.create(contact=contact, amount=1500, status=Order.STATUS_PAID)
        process_incoming_message.run(contact.id, "status")

        options = mock_send_buttons.call_args.args[2]
        self.assertTrue(all("resend_payment_link" not in opt[0] for opt in options))

    @patch("whatsapp_bot.tasks.send_whatsapp_list")
    def test_multiple_orders_show_a_selectable_list(self, mock_send_list):
        from .tasks import process_incoming_message

        contact = Contact.objects.create(phone_number="915555599973", language="en")
        Order.objects.create(contact=contact, amount=1000, status=Order.STATUS_PAID)
        Order.objects.create(contact=contact, amount=2000, status=Order.STATUS_CREATED)
        process_incoming_message.run(contact.id, "status")

        mock_send_list.assert_called_once()

    @patch("whatsapp_bot.tasks.send_whatsapp_buttons")
    def test_selecting_an_order_from_the_list_shows_its_status(self, mock_send_buttons):
        from .tasks import process_incoming_message
        from .services import ORDER_STATUS_SELECT_PREFIX

        contact = Contact.objects.create(phone_number="915555599974", language="en")
        order = Order.objects.create(contact=contact, amount=2000, status=Order.STATUS_FAILED)
        process_incoming_message.run(contact.id, f"{ORDER_STATUS_SELECT_PREFIX}{order.id}")

        body_text = mock_send_buttons.call_args.args[1]
        self.assertIn(f"Order #{order.id}", body_text)

    def test_cannot_select_another_contacts_order(self):
        from .tasks import process_incoming_message
        from .services import ORDER_STATUS_SELECT_PREFIX

        other_contact = Contact.objects.create(phone_number="915555599975", language="en")
        other_order = Order.objects.create(contact=other_contact, amount=2000, status=Order.STATUS_CREATED)
        contact = Contact.objects.create(phone_number="915555599976", language="en")

        process_incoming_message.run(contact.id, f"{ORDER_STATUS_SELECT_PREFIX}{other_order.id}")

        reply = contact.messages.filter(direction="out").last()
        self.assertIn("Couldn't find that order", reply.body)

    @patch("whatsapp_bot.services.send_whatsapp_cta_url")
    @patch("whatsapp_bot.services.payments.create_payment_link", return_value=True)
    def test_resend_on_an_expired_order_creates_a_fresh_order_and_link(self, mock_create_link, mock_send_cta):
        from .tasks import process_incoming_message
        from .services import ORDER_STATUS_RESEND_PREFIX

        contact = Contact.objects.create(phone_number="915555599977", language="en")
        old_order = Order.objects.create(
            contact=contact, amount=3000, status=Order.STATUS_EXPIRED, quote_snapshot=[{"code": "itr_salary"}],
        )
        process_incoming_message.run(contact.id, f"{ORDER_STATUS_RESEND_PREFIX}{old_order.id}")

        mock_create_link.assert_called_once()
        mock_send_cta.assert_called_once()
        new_order = Order.objects.exclude(id=old_order.id).get(contact=contact)
        self.assertEqual(new_order.amount, old_order.amount)
        self.assertEqual(new_order.quote_snapshot, old_order.quote_snapshot)
        old_order.refresh_from_db()
        self.assertEqual(old_order.status, Order.STATUS_EXPIRED)  # never mutated in place
