from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    # Twilio Media Streams will open WSS to this path
    re_path(r'ws/twilio/(?P<call_sid>[^/]+)/?$', consumers.TwilioMediaConsumer.as_asgi()),
    # re_path(r"ws/twilio/inbound/$", consumers.TwilioMediaConsumer.as_asgi()),
]
