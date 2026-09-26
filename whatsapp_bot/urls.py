from django.urls import path

from . import views

urlpatterns = [
    path("webhook/", views.whatsapp_webhook, name="whatsapp_webhook"),
    path("flow/", views.whatsapp_flow_endpoint, name="whatsapp_flow_endpoint"),
    path("razorpay/webhook/", views.razorpay_webhook, name="razorpay_webhook"),
]
