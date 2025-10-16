import json
import base64
import asyncio
import os
import queue
from channels.generic.websocket import AsyncWebsocketConsumer
from vosk import Model, KaldiRecognizer
import soundfile as sf
from gtts import gTTS
# from pydub import AudioSegment
import io
from openai import AzureOpenAI
from dotenv import load_dotenv
load_dotenv()
endpoint = os.getenv("ENDPOINT_URL", "https://jivihireopenai.openai.azure.com/")

    # # # Initialize Azure OpenAI Service client with key-based authentication
client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=os.getenv('OPENAI_API_KEY'),
        api_version="2024-05-01-preview",
    )
# Initialize VOSK model once (download a model and give path)
VOSK_MODEL_PATH = os.environ.get('VOSK_MODEL_PATH', '/path/to/vosk-model-small-en-us-0.15')
if not os.path.exists(VOSK_MODEL_PATH):
    print("Warning: VOSK model path not found. Real ASR won't work until you set VOSK_MODEL_PATH correctly.")
else:
    vosk_model = Model(VOSK_MODEL_PATH)

# helper: convert raw 16-bit base64 frames from Twilio to PCM numpy or feed to VOSK
# Twilio sends audio as base64-encoded PCM16 samples at 8kHz (single-channel) — this matches many telephony flows.
# We'll feed them to VOSK recognizer (which expects 16k or 8k depending on model).
# We'll accumulate frames into recognizer until final result.

def pcm16_bytes_to_wav_bytes(pcm_bytes, sample_rate=8000):
    """Wrap raw PCM16 bytes into a WAV in memory (for pydub or other libs)."""
    buf = io.BytesIO()
    sf.write(buf, sf.blocks_to_array([pcm_bytes]), samplerate=sample_rate, format='WAV', subtype='PCM_16')
    return buf.getvalue()

class TwilioMediaConsumer(AsyncWebsocketConsumer):
    """
    Handles Twilio Media Streams WebSocket messages.
    Twilio sends JSON messages; important types: 'connected', 'start','media','stop'.
    We will send messages back to Twilio to play audio:
      {"event":"media","media":{"payload":"<base64_audio_frame>"}}
    The payload must be raw PCM16 bytes at 8000 Hz encoded base64.
    """

    async def connect(self):
        await self.accept()
        self.call_sid = None
        # Create a VOSK recognizer stream per connection if model exists
        self.recognizer = None
        self.buffer_queue = asyncio.Queue()
        self.processing_task = None
        print("WebSocket: accepted connection")

    async def disconnect(self, close_code):
        if self.processing_task:
            self.processing_task.cancel()
        print("WebSocket disconnected", close_code)

    async def receive(self, text_data=None, bytes_data=None):
        # Twilio sends JSON text messages
        if bytes_data:
            print("Got binary; ignoring for now")
            return

        data = json.loads(text_data)
        event = data.get('event')
        if event == 'connected':
            # Twilio connected; can inspect for callSid
            print("Twilio connected message", data)
            # No action needed
            return

        if event == 'start':
            # Start event contains sampleRate, callSid etc.
            start = data.get('start', {})
            self.call_sid = start.get('callSid') or start.get('call_sid') or 'unknown'
            sample_rate = start.get('sample_rate', 8000)
            print("Stream start - CallSid:", self.call_sid, "sample_rate:", sample_rate)
            # initialize VOSK recognizer
            if 'vosk_model' in globals() and 'vosk_model' in globals() and os.path.exists(VOSK_MODEL_PATH):
                # VOSK expects sample rate matching the model; use 8000 if provided
                self.recognizer = KaldiRecognizer(vosk_model, sample_rate)
                print("VOSK recognizer initialized with sample_rate", sample_rate)
            # start a background processing task that will handle responses (optional)
            self.processing_task = asyncio.create_task(self.process_queue())
            return

        if event == 'media':
            print("In media...........................")
            media = data.get('media', {})
            print(media)
            payload_b64 = media.get('payload')
            if not payload_b64:
                return
            # Twilio media payload is base64 of raw PCM16LE audio samples
            pcm = base64.b64decode(payload_b64)
            print(pcm)
            # For demo: feed bytes to recognizer if exists
            if self.recognizer:
                ok = self.recognizer.AcceptWaveform(pcm)
                if ok:
                    result = json.loads(self.recognizer.Result())
                    text = result.get('text', '')
                    print(text,"tttttttttttttttttttttttttt")
                    if text.strip():
                        print("ASR final:", text)
                        # push recognized text to processing queue
                        await self.buffer_queue.put({'type': 'asr_final', 'text': text})
                else:
                    # partial result available via PartialResult() — not using for now
                    pass

            return

        if event == 'stop':
            print("Stream stopped by Twilio")
            # finish
            if self.processing_task:
                self.processing_task.cancel()
            return

        #for testing only
        # if event == "asr_text":
        #     user_text = data.get("text", "")
        #     print("👂 User said:", user_text)

        #     # Call Azure OpenAI to generate AI reply
        #     try:
        #         completion = client.chat.completions.create(
        #             model="gpt-4o-mini",
        #             messages=[
        #                 {"role": "system", "content": "You are a friendly AI voice agent helping customers."},
        #                 {"role": "user", "content": user_text}
        #             ]
        #         )
        #         ai_reply = completion.choices[0].message.content
        #     except Exception as e:
        #         ai_reply = f"Sorry, AI service error: {e}"

        #     # Send AI text response back
        #     await self.send(json.dumps({
        #         "event": "ai_text",
        #         "ai_text": ai_reply
        #     }))


    async def process_queue(self):
        """
        Background worker: consumes recognized texts, calls bot logic, synthesizes TTS, streams audio back.
        """
        try:
            print("in process queue,,,,,,,,,,,,,,,,,,,,,,,")
            while True:
                item = await self.buffer_queue.get()
                if item['type'] == 'asr_final':
                    user_text = item['text']
                    # run simple bot logic
                    reply_text = self.generate_reply(user_text)
                    print(f"Bot reply: {reply_text}")
                    # synthesize reply_text to audio (mp3/wav) and stream back
                    await self.stream_tts_to_twilio(reply_text)
        except asyncio.CancelledError:
            print("Erroororoor")
            pass

    async def generate_reply(self, user_text):
        """
        Generate a conversational reply using Azure OpenAI.
        """
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a friendly outbound voice assistant. Keep replies under 20 words."},
                {"role": "user", "content": user_text}
            ]
        )
        return completion.choices[0].message.content.strip()

    async def stream_tts_to_twilio(self, text):
        """
        Synthesize `text` to audio, convert to 8kHz PCM16 mono, chunk into frames and send them back as Twilio 'media' messages.
        Twilio expects the audio payload to be raw PCM16LE at 8kHz for telephony - we will convert.
        """
        # 1) Synthesize using gTTS -> MP3 in-memory
        tts = gTTS(text=text, lang='en')
        mp3_buf = io.BytesIO()
        tts.write_to_fp(mp3_buf)
        mp3_buf.seek(0)

        data, sr = sf.read(mp3_buf, dtype='int16')
        if sr != 8000:
            # downsample manually to 8kHz
            import librosa
            data = librosa.resample(data.astype(float), orig_sr=sr, target_sr=8000)
            data = (data * 32767).astype(np.int16)

        pcm_bytes = data.tobytes()

        # 3) Twilio expects small frames — we'll send 3200-byte frames (~200ms @8kHz*2bytes = 16000bytes per second; so chunk e.g. 3200 = 0.2s)
        chunk_size = 3200
        for i in range(0, len(pcm_bytes), chunk_size):
            chunk = pcm_bytes[i:i+chunk_size]
            if not chunk:
                continue
            b64 = base64.b64encode(chunk).decode('ascii')
            msg = {
                "event": "media",
                "media": {
                    "payload": b64
                }
            }
            await self.send(text_data=json.dumps(msg))
            # small sleep to simulate streaming rate to Twilio (keep close to real-time)
            await asyncio.sleep(0.18)
        # Optionally send a "mark" event when done playing (Twilio supports mark messages for bidirectional streams)
        mark_msg = {"event": "mark", "mark": {"name": "tts_end"}}
        await self.send(text_data=json.dumps(mark_msg))
