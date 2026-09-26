"""
Offline eval for services.classify_and_respond's {action, flow_type,
confidence} classification — run by hand after editing
services._CLASSIFICATION_INSTRUCTION or FLOW_TYPE_LABELS, to catch a prompt
regression before a real customer's message gets mis-routed. Same split as
eval_ai_persona.py (get_ai_response's persona rules): this is NOT part of
`manage.py test` — it makes real Gemini API calls (costs money, takes a few
seconds per case, needs GEMINI_API_KEY configured), so it's a manual
pre-deploy check.

Unlike eval_ai_persona's mostly-soft "read the reply and judge" checks, this
one is fully deterministic: classify_and_respond returns discrete
action/flow_type values, so every case below is a genuine PASS/FAIL, not a
human-eyeball REVIEW — this is exactly the labeled sample set the
natural-language flow-intent feature's spec calls for: one clearly-launching
message per FLOW_TYPE_CHOICES, plus messages that must NEVER produce
launch_flow_candidate (purely informational questions, off-topic chat, and a
couple of genuinely ambiguous ones that should clarify rather than launch or
guess).

Exits non-zero if any case fails, so this can still be wired into a
pre-deploy step even though it's not part of the automated test suite.
"""
import sys

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from whatsapp_bot.models import Contact
from whatsapp_bot.services import FLOW_TYPE_CHOICES, classify_and_respond

# Each case: name, prompt, and either:
#   - expect_action + (expect_flow_type when expect_action is
#     "launch_flow_candidate") — an exact match required, or
#   - forbid_actions — a set of actions that must NEVER come back for this
#     prompt (used for the ambiguous cases, where clarify_first vs
#     continue_chat is a legitimate judgment call, but launch_flow_candidate
#     never is).
CASES = [
    # --- One clearly-launching message per flow_type -----------------------
    {"name": "launch_itr", "prompt": "I want to file my ITR this year",
     "expect_action": "launch_flow_candidate", "expect_flow_type": "ITR_TYPE"},
    {"name": "launch_gst", "prompt": "help me register for GST",
     "expect_action": "launch_flow_candidate", "expect_flow_type": "GST_FILING"},
    {"name": "launch_tds", "prompt": "I need to file my TDS returns for this quarter",
     "expect_action": "launch_flow_candidate", "expect_flow_type": "TDS_FILING"},
    {"name": "launch_business_reg", "prompt": "I want to register my new company",
     "expect_action": "launch_flow_candidate", "expect_flow_type": "BUSINESS_REG"},
    {"name": "launch_notices", "prompt": "I got a GST notice and need help responding to it",
     "expect_action": "launch_flow_candidate", "expect_flow_type": "NOTICES"},
    {"name": "launch_book_expert", "prompt": "Can I just talk to someone directly about my taxes?",
     "expect_action": "launch_flow_candidate", "expect_flow_type": "BOOK_EXPERT"},

    # --- Same six, in Hinglish/Romanized Hindi — added after a real customer's
    # "gst ka filling chaiye" got mis-classified as ITR_TYPE (see the
    # flow_type keyword-matching rules + "never default to ITR_TYPE" line in
    # _CLASSIFICATION_INSTRUCTION). Every customer-facing phrase here is a
    # real style this business's customers actually write in, not a
    # constructed edge case — this block exists specifically to catch a
    # regression back toward the old ITR-biased behavior. ---------------
    {"name": "launch_gst_hinglish", "prompt": "gst ka filling chaiye",
     "expect_action": "launch_flow_candidate", "expect_flow_type": "GST_FILING"},
    {"name": "launch_tds_hinglish", "prompt": "tds file karna hai",
     "expect_action": "launch_flow_candidate", "expect_flow_type": "TDS_FILING"},
    {"name": "launch_itr_hinglish", "prompt": "itr filing karna hai",
     "expect_action": "launch_flow_candidate", "expect_flow_type": "ITR_TYPE"},
    {"name": "launch_business_reg_hinglish", "prompt": "mujhe company register karwani hai",
     "expect_action": "launch_flow_candidate", "expect_flow_type": "BUSINESS_REG"},
    {"name": "launch_notices_hinglish", "prompt": "mujhe notice aaya hai gst ka",
     "expect_action": "launch_flow_candidate", "expect_flow_type": "NOTICES"},
    {"name": "launch_book_expert_hinglish", "prompt": "expert se baat karni hai",
     "expect_action": "launch_flow_candidate", "expect_flow_type": "BOOK_EXPERT"},

    # --- Informational — must never launch ----------------------------------
    {"name": "info_itr_deadline", "prompt": "what's the ITR deadline this year?",
     "forbid_actions": {"launch_flow_candidate"}},
    {"name": "info_gst_price", "prompt": "how much does GST registration cost?",
     "forbid_actions": {"launch_flow_candidate"}},
    {"name": "info_freelance_filing", "prompt": "do I need to file if I'm freelancing on the side?",
     "forbid_actions": {"launch_flow_candidate"}},
    {"name": "info_tds_documents", "prompt": "what documents do I need for TDS filing?",
     "forbid_actions": {"launch_flow_candidate"}},
    {"name": "info_business_reg_types", "prompt": "what's the difference between a private limited and an LLP?",
     "forbid_actions": {"launch_flow_candidate"}},
    {"name": "off_topic_weather", "prompt": "what's the weather like in Mumbai today?",
     "expect_action": "continue_chat"},
    {"name": "small_talk_thanks", "prompt": "thanks, that's all for now!",
     "expect_action": "continue_chat"},

    # --- Ambiguous — a real judgment call between clarify_first and
    # continue_chat, but launching is always wrong here. -------------------
    {"name": "ambiguous_gst_help", "prompt": "help me with GST",
     "forbid_actions": {"launch_flow_candidate"}},
    {"name": "ambiguous_notice_mention", "prompt": "I received a notice, not sure what it's about",
     "forbid_actions": {"launch_flow_candidate"}},
]


class Command(BaseCommand):
    help = (
        "Runs a labeled set of prompts through services.classify_and_respond and "
        "checks the resulting action/flow_type against what's expected — one "
        "clearly-launching message per FLOW_TYPE_CHOICES, plus informational/"
        "off-topic/ambiguous messages that must never trigger a launch. Makes real "
        "Gemini API calls — run by hand after editing the classification prompt, "
        "not part of `manage.py test`."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--case", default=None,
            help="Run only the named case (see CASES in this file) instead of all of them.",
        )

    def handle(self, *args, **options):
        # See eval_ai_persona.py's identical guard — replies/logs can contain
        # ₹ or em-dashes that crash Windows' default console codepage.
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

        if not settings.GEMINI_API_KEY:
            raise CommandError("GEMINI_API_KEY is not set — this eval needs real Gemini calls to mean anything.")

        cases = CASES
        if options["case"]:
            cases = [c for c in CASES if c["name"] == options["case"]]
            if not cases:
                raise CommandError(f"No case named {options['case']!r}. Known cases: {', '.join(c['name'] for c in CASES)}")

        failures = 0
        for case in cases:
            contact = Contact(language="en")
            result = classify_and_respond(contact, case["prompt"], history=None)
            action, flow_type, confidence = result["action"], result["flow_type"], result["confidence"]

            case_failures = []
            expect_action = case.get("expect_action")
            if expect_action and action != expect_action:
                case_failures.append(f"expected action={expect_action!r}, got {action!r}")
            expect_flow_type = case.get("expect_flow_type")
            if expect_flow_type and flow_type != expect_flow_type:
                case_failures.append(f"expected flow_type={expect_flow_type!r}, got {flow_type!r}")
            for forbidden in case.get("forbid_actions", set()):
                if action == forbidden:
                    case_failures.append(f"action={forbidden!r} must never happen for this prompt")
            if action == "launch_flow_candidate" and flow_type not in FLOW_TYPE_CHOICES:
                case_failures.append(f"launch_flow_candidate with invalid/missing flow_type={flow_type!r}")

            self.stdout.write(f"\n{'=' * 70}")
            self.stdout.write(self.style.MIGRATE_HEADING(case["name"]))
            self.stdout.write(f"Prompt:     {case['prompt']}")
            self.stdout.write(f"Result:     action={action} flow_type={flow_type} confidence={confidence}")
            self.stdout.write(f"Reply:      {result['reply']}")

            if case_failures:
                failures += len(case_failures)
                for desc in case_failures:
                    self.stdout.write(self.style.ERROR(f"FAIL — {desc}"))
            else:
                self.stdout.write(self.style.SUCCESS("PASS"))

        self.stdout.write(f"\n{'=' * 70}")
        if failures:
            raise CommandError(f"{failures} check(s) failed — see FAIL lines above.")
        self.stdout.write(self.style.SUCCESS(f"All {len(cases)} case(s) passed."))
