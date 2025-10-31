import os
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import redirect
from django.conf import settings
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream

TWILIO_SID = settings.TWILIO_ACCOUNT_SID
TWILIO_TOKEN = settings.TWILIO_AUTH_TOKEN
TWILIO_FROM = settings.TWILIO_NUMBER
PUBLIC_URL = settings.PUBLIC_URL
client = Client(TWILIO_SID, TWILIO_TOKEN)

@csrf_exempt
def make_outbound_call(request):
    """
    POST params: to=<phone_number>
    Initiates outbound call and points Twilio to our TwiML which contains <Connect><Stream>
    """
    to = request.POST.get('to') or request.GET.get('to')
    if not to:
        return HttpResponse("Specify ?to=+NUMBER", status=400)
    
    call = client.calls.create(
        to=to,
        from_=TWILIO_FROM,
        url=request.build_absolute_uri(f'/agent/twiml/{to}/')  # we use 'to' in URL for uniqueness; your TwiML view should accept path
    )
    return HttpResponse(f"Call initiated: {call.sid}\nTwiml URL: {request.build_absolute_uri(f'/agent/twiml/{to}/')}")

@csrf_exempt
def inbound_call(request):
    """Handles inbound call and connects caller to WebSocket stream."""
    call_sid = request.POST.get('CallSid', '')
    stream_url = request.build_absolute_uri(f'/ws/twilio/{call_sid}/').replace('http', 'ws')
    vr = VoiceResponse()
    connect = Connect()
    connect.append(Stream(url=stream_url))
    vr.append(connect)
    return HttpResponse(str(vr), content_type='application/xml')

@csrf_exempt
def twiml_response(request, call_id):
    """
    Returns TwiML that tells Twilio to Connect->Stream to our WebSocket consumer.
    Twilio will then open a WSS to wss://yourserver/ws/twilio/{CallSid}/
    Twilio will send JSON messages (connected, start, media...) and accept JSON messages from us to play audio back.
    """
    call_sid = request.POST.get('CallSid', '') or request.GET.get('CallSid', '')
    # Twilio requires a wss endpoint publically reachable
    stream_url = f'{PUBLIC_URL}/ws/twilio/{call_sid}/'.replace('http', 'ws')
    vr = VoiceResponse()
    connect = Connect()
    connect.append(Stream(url=stream_url))
    vr.append(connect)
    return HttpResponse(str(vr), content_type='application/xml')
