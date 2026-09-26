from django.contrib import admin

from .document_loader import process_document
from .models import (
    BusinessProfile, Contact, FlowIntentClassification, KnowledgeBaseDocument, KnowledgeBaseEntry, Message,
    Order, ServiceCategory, ServicePricing,
)


@admin.register(BusinessProfile)
class BusinessProfileAdmin(admin.ModelAdmin):
    list_display = ("name", "updated_at")

    def has_add_permission(self, request):
        # Singleton — one profile is all the bot ever reads.
        return not BusinessProfile.objects.exists()


@admin.register(KnowledgeBaseDocument)
class KnowledgeBaseDocumentAdmin(admin.ModelAdmin):
    list_display = ("title", "processed", "entry_count", "uploaded_at")
    list_filter = ("processed",)
    readonly_fields = ("processed", "processing_error")
    actions = ["process_documents"]

    @admin.display(description="KB entries generated")
    def entry_count(self, obj):
        return obj.entries.count()

    @admin.action(description="Process into knowledge base entries")
    def process_documents(self, request, queryset):
        for document in queryset:
            try:
                count = process_document(document)
                document.processed = True
                document.processing_error = ""
                document.save(update_fields=["processed", "processing_error"])
                self.message_user(request, f"{document.title}: created {count} KB entries.")
            except Exception as exc:
                document.processed = False
                document.processing_error = str(exc)
                document.save(update_fields=["processed", "processing_error"])
                self.message_user(request, f"{document.title}: failed — {exc}", level="error")


@admin.register(KnowledgeBaseEntry)
class KnowledgeBaseEntryAdmin(admin.ModelAdmin):
    list_display = ("title", "source_document", "is_active", "order")
    list_filter = ("is_active", "source_document")
    search_fields = ("title", "content")
    ordering = ("order",)


@admin.register(ServiceCategory)
class ServiceCategoryAdmin(admin.ModelAdmin):
    list_display = ("display_name", "category_code", "url", "order")
    ordering = ("order",)


@admin.register(ServicePricing)
class ServicePricingAdmin(admin.ModelAdmin):
    list_display = ("display_name", "category", "parent_service", "question_type", "price_display", "is_active", "order")
    list_filter = ("category", "parent_service", "question_type", "is_active")
    ordering = ("category__order", "order")


@admin.register(Contact)
class ContactAdmin(admin.ModelAdmin):
    list_display = (
        "phone_number", "name", "lead_tier", "lead_score", "needs_human_review",
        "needs_human_review_reason", "bot_paused_until", "flow_furthest_screen", "predicted_next_action",
        "first_time_filer", "callback_slot", "language", "conversation_step",
        "satisfaction_rating", "created_at",
    )
    # needs_human_review is the primary triage filter until a live alert
    # channel (Slack/email/WhatsApp) is wired up — see services.notify_human_alert.
    # bot_paused_until alongside it: needs_human_review is "someone should
    # look at this", bot_paused_until is "the bot has actually gone quiet
    # for this contact" — related but not the same signal, see resume_bot.
    list_filter = (
        "needs_human_review", "lead_tier", "flow_furthest_screen", "conversation_step",
        "satisfaction_rating", "callback_slot", "first_time_filer",
    )
    search_fields = ("phone_number", "name")
    ordering = ("-needs_human_review", "-lead_score")
    actions = ["resume_bot"]

    @admin.action(description="Resume bot (mark handled)")
    def resume_bot(self, request, queryset):
        """Clears the needs_human_review triage flag — the "CA closes the
        conversation" handback. Also clears the now-inert bot_paused_until
        field for any contact still carrying a value set before the
        human-handoff pause feature was removed (tasks.process_incoming_message
        no longer gates auto-replies on it, so this half of the update is a
        no-op for anything set after that change)."""
        updated = queryset.update(
            bot_paused_until=None, needs_human_review=False,
            needs_human_review_reason="", needs_human_review_at=None,
        )
        self.message_user(request, f"Resumed the bot for {updated} contact(s).")

    readonly_fields = (
        "lead_score", "lead_tier", "flow_furthest_screen", "predicted_next_action",
        "satisfaction_rating", "satisfaction_rating_at",
        "needs_human_review", "needs_human_review_reason", "needs_human_review_at",
        "review_reached_at", "cart_nudge_2h_sent_at", "cart_nudge_24h_sent_at", "referral_ask_sent_at",
    )


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ("contact", "direction", "body", "created_at")
    list_filter = ("direction",)
    search_fields = ("body",)


@admin.register(FlowIntentClassification)
class FlowIntentClassificationAdmin(admin.ModelAdmin):
    list_display = ("contact", "flow_type", "confidence", "decision", "created_at", "resolved_at")
    list_filter = ("confidence", "decision", "flow_type")
    search_fields = ("contact__phone_number", "contact__name", "message_text")
    ordering = ("-created_at",)
    # Written only by tasks.py's classify_and_respond branch and
    # _resolve_pending_flow_intent_classification — an analytics trail, same
    # never-hand-edited reasoning as Order's fields above.
    readonly_fields = (
        "contact", "flow_type", "confidence", "message_text", "decision", "created_at", "resolved_at",
    )


@admin.register(Order)
class OrderAdmin(admin.ModelAdmin):
    list_display = ("id", "contact", "amount", "is_advance", "status", "payment_link_id", "created_at")
    list_filter = ("status", "is_advance", "gateway")
    search_fields = ("contact__phone_number", "contact__name", "payment_link_id", "gateway_payment_id")
    ordering = ("-created_at",)
    # Every field here is written only by services.maybe_send_payment_cta /
    # tasks.process_payment_webhook — never hand-edited, same reasoning as
    # Contact's lead-score/needs_human_review fields above.
    readonly_fields = (
        "contact", "quote_snapshot", "amount", "is_advance", "gateway",
        "payment_link_id", "payment_link_url", "gateway_payment_id",
        "status", "failure_reason", "created_at", "updated_at",
    )
