from django.urls import path
from . import views

urlpatterns = [
    path('call/outbound/', views.make_outbound_call, name='outbound_call'),
    path('twiml/<str:call_id>/', views.twiml_response, name='twiml_response'),
    path('call/inbound/', views.inbound_call, name='inbound_call'),
]
