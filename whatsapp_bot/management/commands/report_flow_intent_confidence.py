"""
Reports, per confidence bucket, how often a real customer actually confirmed
vs. rejected a natural-language flow-intent launch — see
models.FlowIntentClassification's docstring for why this reads from a DB
table rather than the [flow-intent] log line (never persisted anywhere;
myproject/settings.py's LOGGING has no file handler, so that line only ever
reached a console window).

Unlike eval_flow_intent.py (a synthetic, hand-labeled sample run against
Gemini directly, pre-deploy), this is real production traffic — the actual
evidence needed before trusting confidence="high" to skip the Confirm/Edit
prompt outright (see tasks.py's pending_flow_type_candidate branch and its
"Never auto-launch, regardless of confidence" comment). Safe to run anytime,
read-only, no API calls.
"""
from django.core.management.base import BaseCommand

from whatsapp_bot.models import FlowIntentClassification


class Command(BaseCommand):
    help = "Reports confirm/reject rates per confidence bucket for real flow-intent launches."

    def add_arguments(self, parser):
        parser.add_argument(
            "--min-resolved", type=int, default=20,
            help="Below this many resolved (confirmed+rejected) cases, flag the bucket's "
            "rate as not yet statistically meaningful rather than presenting it as fact. "
            "Default 20 — a rough floor, not a rigorous sample-size calculation.",
        )

    def handle(self, *args, **options):
        min_resolved = options["min_resolved"]
        rows = FlowIntentClassification.objects.all()
        if not rows.exists():
            self.stdout.write("No FlowIntentClassification rows yet — nothing to report.")
            return

        for confidence, label in FlowIntentClassification.CONFIDENCE_CHOICES:
            bucket = rows.filter(confidence=confidence)
            total = bucket.count()
            confirmed = bucket.filter(decision=FlowIntentClassification.DECISION_CONFIRMED).count()
            rejected = bucket.filter(decision=FlowIntentClassification.DECISION_REJECTED).count()
            pending = bucket.filter(decision=FlowIntentClassification.DECISION_PENDING).count()
            resolved = confirmed + rejected

            self.stdout.write(f"\n{label} confidence — {total} total ({pending} still pending)")
            if resolved == 0:
                self.stdout.write("  No resolved cases yet.")
                continue

            reject_rate = rejected / resolved
            self.stdout.write(f"  confirmed={confirmed}  rejected={rejected}  reject_rate={reject_rate:.0%}")
            if resolved < min_resolved:
                self.stdout.write(
                    self.style.WARNING(
                        f"  Only {resolved} resolved case(s) — below --min-resolved={min_resolved}, "
                        "too small to act on yet."
                    )
                )

        self.stdout.write(
            "\nA confidence bucket's reject_rate near 0%, on a large-enough sample, is the "
            "actual evidence needed before letting that bucket skip the Confirm/Edit prompt — "
            "see tasks.py's pending_flow_type_candidate branch."
        )
