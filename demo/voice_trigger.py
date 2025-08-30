import azure.cognitiveservices.speech as speechsdk

# --- Use your existing key/region as-is for now ---
SPEECH_KEY = ""
SPEECH_REGION = "southeastasia"

# Build configs
speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
speech_config.speech_recognition_language = "en-US"
# Teams-like auto punctuation/casing:
speech_config.set_property(
    speechsdk.PropertyId.SpeechServiceResponse_PostProcessingOption, "TrueText"
)

# Force your laptop mic (from arecord -L)
try_first = "plughw:CARD=L1080p,DEV=0"   # best choice per your listing
audio_config = speechsdk.audio.AudioConfig(device_name=try_first)

# If that ever fails on your machine, uncomment the next line to try Pulse default:
# audio_config = speechsdk.audio.AudioConfig(device_name="default")

recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

print("Speak… (Ctrl+C to stop)")

def on_recognizing(evt):
    if evt.result.text:
        print("INTERIM:", evt.result.text)

def on_recognized(evt):
    if evt.result.text:
        print("FINAL  :", evt.result.text)

recognizer.recognizing.connect(on_recognizing)
recognizer.recognized.connect(on_recognized)

recognizer.start_continuous_recognition()
try:
    while True:
        pass
except KeyboardInterrupt:
    recognizer.stop_continuous_recognition()
    print("\nStopped.")