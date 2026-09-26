import os

from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "myproject.settings")

app = Celery("myproject")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()

# Periodic tasks (requires `celery -A myproject beat` running alongside the
# worker — see start_bot.ps1). Static schedule rather than django-celery-beat:
# one task, no admin-editable schedule needed yet, so it's not worth the
# extra dependency/DB tables.
app.conf.beat_schedule = {
    "abandoned-cart-nudges": {
        "task": "whatsapp_bot.tasks.send_abandoned_cart_nudges",
        "schedule": 900.0,  # every 15 minutes
    },
    "dropoff-nudges": {
        "task": "whatsapp_bot.tasks.send_dropoff_nudges",
        "schedule": 900.0,  # every 15 minutes
    },
    # Reinstated at a wider spacing (2h, see tasks.HOT_LEAD_NUDGE_AFTER_HOURS)
    # and scoped to hot leads who never reached the review screen at all
    # (review_reached_at is None) — a hot lead who *did* abandon a cart
    # already gets dropoff-nudges' hot-only 30m/7d touches instead, so the
    # same contact never gets both.
    "hot-lead-nudges": {
        "task": "whatsapp_bot.tasks.send_hot_lead_nudges",
        "schedule": 900.0,  # every 15 minutes
    },
    "warm-lead-nudges": {
        "task": "whatsapp_bot.tasks.send_warm_lead_nudges",
        "schedule": 900.0,  # every 15 minutes
    },
}
