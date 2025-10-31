import json,os,io,time,asyncio,base64,math
from channels.generic.websocket import AsyncWebsocketConsumer
import numpy as np
import soundfile as sf
# from gtts import gTTS
from openai import OpenAI
from dotenv import load_dotenv
from .rag import search_similar_docs
# import librosa
from scipy.signal import resample_poly
import audioop
# try:
#     from faster_whisper import WhisperModel
# except Exception:
#     WhisperModel = None

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# whisper_model = None
# if WhisperModel:
#     try:
#         # choose "tiny" or "small" by tradeoff (tiny = faster)
#         whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
#         print("✅ faster-whisper model loaded (tiny)")
#     except Exception as e:
#         print("⚠️ faster-whisper load failed:", e)

# --- TUNABLE PARAMETERS ---
SAMPLE_RATE = 8000               # incoming Twilio audio (usually 8000 Hz)
BYTES_PER_SEC = SAMPLE_RATE * 2  # PCM16LE, mono
PROCESS_CHUNK_SEC = 1.0          # process every ~0.6s chunk for partials
CHUNK_OVERLAP_SEC = 0.2          # overlap window to avoid chopping words
PROCESS_CHUNK_BYTES = int(PROCESS_CHUNK_SEC * BYTES_PER_SEC)
CHUNK_OVERLAP_BYTES = int(CHUNK_OVERLAP_SEC * BYTES_PER_SEC)
SILENCE_THRESHOLD = 50          # RMS threshold for voice detection; tune for your audio
SPEECH_TIMEOUT = 1.2             # seconds of silence -> finalize utterance
MIN_UTTERANCE_SEC = int(SAMPLE_RATE * 0.15 * 2)

def resample_audio_float32(float_audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample a 1-D float32 numpy array from orig_sr -> target_sr using resample_poly.
    Returns float32 array.
    """
    if orig_sr == target_sr:
        return float_audio
    # compute integer up/down factors as small integers
    # up/down = target_sr / orig_sr simplified by gcd
    up = target_sr
    down = orig_sr
    g = math.gcd(up, down)
    up //= g
    down //= g
    # resample_poly expects float input
    res = resample_poly(float_audio, up, down)
    return res.astype(np.float32)

def pcm16_to_int16_arr(pcm_bytes):
    return np.frombuffer(pcm_bytes, dtype=np.int16)

def int16_to_float32(arr):
    return arr.astype(np.float32) / 32768.0

class TwilioMediaConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        self.call_sid = None
        self.stream_sid = None
        self.sample_rate = SAMPLE_RATE
        # self.recognizer = None           # for VOSK incremental if used
        self._buffer = bytearray()       # rolling buffer of recent raw PCM bytes
        self._last_voice_time = 0.0
        self._last_partial_text = ""
        self._last_partial_ts = 0.0
        self._transcribe_lock = asyncio.Lock()
        self.buffer_queue = asyncio.Queue()   # for final ASR -> AI processing
        self._process_task = asyncio.create_task(self.process_queue())
        print("WebSocket accepted")

    async def disconnect(self, close_code):
        if self._process_task:
            self._process_task.cancel()
        self._buffer.clear()
        print("WebSocket disconnected", close_code)

    async def receive(self, text_data=None, bytes_data=None):
        if bytes_data:
            return
        try:
            data = json.loads(text_data)
        except Exception as e:
            print("Invalid JSON:", e)
            return

        event = data.get("event")
        if event == "connected":
            print("Twilio connected.")
            return

        if event == "start":
            start = data.get("start", {})
            self.call_sid = start.get("callSid") or start.get("call_sid") or self.call_sid
            self.stream_sid = start.get("streamSid") or start.get("stream_sid") or self.stream_sid
            self.sample_rate = start.get("sample_rate", SAMPLE_RATE)
            print(f"Stream start - CallSid={self.call_sid} sample_rate={self.sample_rate}")
            try:
                await self.stream_tts_to_twilio("Hello, How can i help you?.")
            except Exception as e:
                print("Welcome message error:",e)

        if event == "media":
            media = data.get("media", {})
            payload_b64 = media.get("payload")
            if not payload_b64:
                return
            try:
                # pcm = base64.b64decode(payload_b64)
                pcm = audioop.ulaw2lin(base64.b64decode(payload_b64), 2)
            except Exception as e:
                print("Base64 conversion Error.")
            if not pcm:
                print("Empty PCM data received")
                return

            # --- faster-whisper incremental path (rolling buffer + partials) ---
            # compute RMS of incoming chunk
            try:
                arr = pcm16_to_int16_arr(pcm)
                rms = int(np.sqrt(np.mean(arr.astype(np.float32) ** 2))) if arr.size > 0 else 0
            except Exception:
                rms = 0
            print(rms,"rrrrrrrrrrrrrrrrrmmmmmmmmmmmmmmmmmmmmmmmmmsssssssssssssssssssssssss")
            now = time.time()
            if rms >= SILENCE_THRESHOLD:
                self._last_voice_time = now

            # append to buffer
            self._buffer += pcm

            # if buffer reached processing threshold -> create sliding window and transcribe partial
            if len(self._buffer) >= PROCESS_CHUNK_BYTES:
                # window includes overlap for smoother partials
                start_idx = max(0, len(self._buffer) - PROCESS_CHUNK_BYTES - CHUNK_OVERLAP_BYTES)
                window = bytes(self._buffer[start_idx:])
                # only launch transcription if not already running
                if not self._transcribe_lock.locked():
                    print("not loked.........................")
                    asyncio.create_task(self._transcribe_partial(window))
                # cap buffer growth (keep last ~4s)
                max_keep = int(BYTES_PER_SEC * 4)
                if len(self._buffer) > max_keep:
                    self._buffer = bytearray(self._buffer[-max_keep:])
            print(now-self._last_voice_time,"TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT")
            # finalize utterance if silence for SPEECH_TIMEOUT
            min_buffer_size = int(SAMPLE_RATE * 0.5 * 2) 
            if (now - self._last_voice_time) > SPEECH_TIMEOUT and len(self._buffer) > min_buffer_size:
                buf_copy = bytes(self._buffer)
                self._buffer = bytearray()
                self._last_voice_time = now
                asyncio.create_task(self._transcribe_final_and_queue(buf_copy))
            return

        if event == "stop":
            print("Stream stop event")
            # finalize any remaining audio
            if len(self._buffer) > 0:
                buf_copy = bytes(self._buffer)
                self._buffer = bytearray()
                asyncio.create_task(self._transcribe_final_and_queue(buf_copy))
            return

    async def _transcribe_partial(self, pcm_bytes: bytes):
        """Transcribe a sliding window and send an `asr_partial`."""
        async with self._transcribe_lock:
            try:
                if len(pcm_bytes) < int(SAMPLE_RATE * 0.3 * 2):
                    print("⚠️ PCM chunk too small for transcription")
                    return
                int16 = pcm16_to_int16_arr(pcm_bytes)
                float_audio = int16_to_float32(int16)
                if self.sample_rate != 16000:
                    # float_audio = await asyncio.to_thread(librosa.resample, float_audio, self.sample_rate, 16000)
                    float_audio = await asyncio.to_thread(resample_audio_float32, float_audio, self.sample_rate, 16000)
                # def run_whisper(audio_arr):
                #     segments, _ = whisper_model.transcribe(audio_arr, language="en", beam_size=3)
                #     return " ".join([seg.text for seg in segments]).strip()

                # text = await asyncio.to_thread(run_whisper, float_audio)
                text = await self.transcribe_openai(pcm_bytes)
                if text and len(text) > 2:
                    print("partial textnnnnnnnnnnnnnnnnnnnnnn",text)
                    # throttle partials a bit to avoid flooding
                    now = time.time()
                    if text != self._last_partial_text or (now - self._last_partial_ts) > 0.8:
                        self._last_partial_text = text
                        self._last_partial_ts = now
                        await self.buffer_queue.put({"type":"asr_partial", "text": text})
                        # await self.send(json.dumps({"event":"asr_partial", "text": text}))
            except Exception as e:
                print("Partial transcribe error:", e)

    async def _transcribe_final_and_queue(self, pcm_bytes: bytes):
        """Transcribe final utterance and put into buffer_queue as asr_final."""
        async with self._transcribe_lock:
            try:
                if len(pcm_bytes) < int(SAMPLE_RATE * 0.3 * 2):
                    print("⚠️ PCM chunk too small for final transcription")
                    return
                int16 = pcm16_to_int16_arr(pcm_bytes)
                float_audio = int16_to_float32(int16)
                if self.sample_rate != 16000:
                    # float_audio = await asyncio.to_thread(librosa.resample, float_audio, self.sample_rate, 16000)
                    float_audio = await asyncio.to_thread(resample_audio_float32, float_audio, self.sample_rate, 16000)

                # def run_whisper(audio_arr):
                #     segments, _ = whisper_model.transcribe(audio_arr, language="en", beam_size=5)
                #     return " ".join([seg.text for seg in segments]).strip()

                # text = await asyncio.to_thread(run_whisper, float_audio)
                text = await self.transcribe_openai(pcm_bytes)
                if text and len(text) > 2:
                    print("FInal text 77777777777777777777777777777777777777777")
                    await self.buffer_queue.put({"type":"asr_final", "text": text})
                    await self.send(json.dumps({"event":"asr_final", "text": text}))
                else:
                    print("No final transcription produced")
                    await self.buffer_queue.put({"type": "asr_final", "text": "Sorry, I didn't hear anything."})
                    await self.stream_tts_to_twilio("I didn't hear anything. Could you repeat?")
            except Exception as e:
                print("Final transcribe error:", e)

    async def process_queue(self):
        """Consume final ASR results, call AI and stream TTS back to Twilio."""
        while True:
            try:
                print("In the process queue..........................")
                item = await self.buffer_queue.get()
                if item["type"] == "asr_final":
                    user_text = item["text"]
                    print("ASR final =>", user_text)
                    # generate reply (can be heavy; keep in thread if needed inside)
                    reply = await self.generate_reply(user_text)
                    await self.stream_tts_to_twilio(reply)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print("process_queue error:", e)
                await asyncio.sleep(0.1)

    async def generate_reply(self, user_text: str) -> str:
        """Call OpenAI (or local LLM) — keep it reasonably short."""
        try:
            print("In generate 4444444444444444444444444444444444")
            relevant_docs = [d if isinstance(d, str) else d[0] for d in search_similar_docs(user_text, top_k=2)]
            context = "\n".join(relevant_docs)
            print("context",context)
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role":"system","content":"You are a concise friendly voice assistant."},
                    {"role":"system","content":f"Company data:\n{context}"},
                    {"role":"user","content":user_text}
                ],
                temperature=0.3
            )
            reply = completion.choices[0].message.content.strip()
            print(reply,"rrrrrrrrrrrrrrrreeeeeeeeeeeeeeppppppppp")
            return reply
        except Exception as e:
            print("AI error:", e)
            return "Sorry — I didn't catch that."

    # async def stream_tts_to_twilio(self, text: str):
    #     """Synthesize text to audio, downsample to 8kHz PCM16, and stream back chunked to Twilio."""
    #     try:
    #         print("Final stram to tttsmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm",text)
    #         # synthesize MP3 via gTTS in thread
    #         def gen_mp3(t):
    #             buf = io.BytesIO()
    #             gTTS(text=t, lang="en").write_to_fp(buf)
    #             buf.seek(0)
    #             return buf.read()
    #         mp3_bytes = await asyncio.to_thread(gen_mp3, text)
    #         mp3_buf = io.BytesIO(mp3_bytes)

    #         # read mp3 to float array
    #         data, sr = sf.read(mp3_buf, dtype="float32")
    #         if data.ndim > 1:
    #             data = np.mean(data, axis=1)  # Convert to mono
    #         else:
    #             data = data.flatten()
    #         if sr != SAMPLE_RATE:
    #             # data = await asyncio.to_thread(librosa.resample, data, sr, SAMPLE_RATE)
    #             data = await asyncio.to_thread(resample_audio_float32, data, sr, SAMPLE_RATE)
                
    #         pcm_int16 = (data * 32767).astype(np.int16)
    #         pcm_bytes = pcm_int16.tobytes()

    #         # send in small real-time chunks (~100ms)
    #         chunk_size = int(BYTES_PER_SEC * 0.1)  # 0.1s
    #         for i in range(0, len(pcm_bytes), chunk_size):
    #             chunk = pcm_bytes[i:i+chunk_size]
    #             if not chunk:
    #                 continue
    #             msg = {"event":"media","streamSid": self.stream_sid,"track": "outbound","media":{"payload": base64.b64encode(chunk).decode('ascii')}}
    #             # await self.send(text_data=json.dumps(msg))
    #             await self.safe_send(msg)
    #             await asyncio.sleep(0.02)
    #         # send mark
    #         # await self.send(text_data=json.dumps({"event":"mark","streamSid": self.stream_sid,"mark":{"name":"tts_end"}}))
    #         await self.send(text_data=json.dumps({"event":"mark","streamSid": self.stream_sid,"mark":{"name":"tts_end"}}))
    #         print("All done 444444444444444444444444444444444444444444444444444")
    #     except Exception as e:
    #         print("TTS error:", e)

    async def stream_tts_to_twilio(self, text: str):
        """Synthesize text to audio with OpenAI TTS and stream to Twilio."""
        try:
            print("🔊 TTS (OpenAI):", text)

            # Generate speech from GPT model
            speech_response = await asyncio.to_thread(
                lambda: client.audio.speech.create(
                    model="gpt-4o-mini-tts",
                    voice="alloy",  # voices: alloy, verse, shimmer, etc.
                    input=text
                )
            )

            pcm_bytes = speech_response.read()  # returns 24kHz mono PCM16 WAV

            # Convert to float32 & downsample to 8kHz (Twilio requirement)
            data, sr = sf.read(io.BytesIO(pcm_bytes), dtype="float32")
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            data = await asyncio.to_thread(resample_audio_float32, data, sr, SAMPLE_RATE)

            pcm_int16 = (data * 32767).astype(np.int16)
            pcm_bytes = pcm_int16.tobytes()

            # Stream back in 100ms chunks
            chunk_size = int(BYTES_PER_SEC * 0.1)
            for i in range(0, len(pcm_bytes), chunk_size):
                chunk = pcm_bytes[i:i+chunk_size]
                msg = {
                    "event": "media",
                    "streamSid": self.stream_sid,
                    # "track": "outbound",
                    "media": {"payload": base64.b64encode(chunk).decode("ascii")},
                }
                await self.safe_send(msg)
                await asyncio.sleep(0.02)

            await self.send(json.dumps({
                "event": "mark",
                "streamSid": self.stream_sid,
                "mark": {"name": "tts_end"}
            }))

        except Exception as e:
            print("TTS error:", e)

    async def safe_send(self, message: dict):
        """Send safely — skip if websocket already closed."""
        try:
            await self.send(text_data=json.dumps(message))
        except Exception as e:
            print("⚠️ safe_send error:", e)

    async def transcribe_openai(self, pcm_bytes: bytes) -> str:
        """Use OpenAI's STT (gpt-4o-mini-transcribe) for transcription."""
        try:
            with io.BytesIO(pcm_bytes) as f:
                # Send to OpenAI STT
                result = await asyncio.to_thread(
                    lambda: client.audio.transcriptions.create(
                        model="gpt-4o-mini-transcribe",
                        file=("speech.wav", f, "audio/wav")
                    )
                )
            return result.text.strip()
        except Exception as e:
            print("STT error:", e)
            return ""
