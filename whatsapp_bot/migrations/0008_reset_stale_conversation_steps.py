"""
Defensive data migration, paired with the 0007 step-value rename
(menu_shown -> category_shown, awaiting_price_clarification -> split into
awaiting_quantity/awaiting_topic).

Renaming a CharField's `choices` does NOT touch existing row data — a
Contact already sitting on the literal string "menu_shown" or
"awaiting_price_clarification" would silently stop matching any branch in
process_incoming_message and fall through to free-chat forever. That's
exactly the bug traced on 2026-08-22 (contact 919321835370 got stuck after
an earlier migration reset its conversation_step). This resets any
contact still on one of the old, now-invalid values back to blank, which
process_incoming_message treats as "re-show the category menu" rather
than silently free-chatting.
"""
from django.db import migrations

OLD_STALE_STEPS = ["menu_shown", "awaiting_price_clarification"]


def reset_stale_steps(apps, schema_editor):
    Contact = apps.get_model("whatsapp_bot", "Contact")
    Contact.objects.filter(conversation_step__in=OLD_STALE_STEPS).update(
        conversation_step="", pending_service_code=""
    )


def noop_reverse(apps, schema_editor):
    # Nothing to reverse — we don't know what the original stale value was.
    pass


class Migration(migrations.Migration):

    dependencies = [
        ("whatsapp_bot", "0007_servicecategory_alter_servicepricing_options_and_more"),
    ]

    operations = [
        migrations.RunPython(reset_stale_steps, noop_reverse),
    ]
