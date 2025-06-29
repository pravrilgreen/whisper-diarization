import sounddevice as sd
import numpy as np
import threading
import websocket
import struct
import json
import time
import sys

# ===================== CONFIG ========================
SERVER_URL = "ws://localhost:8012"
SAMPLE_RATE = 16000
LANGS = ["en", "vi", "ja"]
SRC_LANG_IDX = 0
SRC_LANG = LANGS[SRC_LANG_IDX]
TGT_LANG = "en"
USER = "TuanLM12-en"
# =====================================================

# Flags
ready_flag = threading.Event()
stop_flag = threading.Event()
src_lang_lock = threading.Lock()


def send_audio(ws):
    def callback(indata, frames, time_info, status):
        if stop_flag.is_set():
            return
        if status:
            print("[Audio Callback] Status:", status)

        with src_lang_lock:
            src_lang_value = SRC_LANG

        pcm16 = (indata[:, 0]).astype(np.int16).tobytes()

        header_dict = {
            "sampleRate": SAMPLE_RATE,
            "dtype": "int16",
            "channels": 1,
            "language": src_lang_value,  # khớp với server field
            "user": USER
        }

        header_json = json.dumps(header_dict).encode("utf-8")
        header_len = struct.pack("<I", len(header_json))
        message = header_len + header_json + pcm16

        try:
            if ws.sock and ws.sock.connected:
                ws.send(message, opcode=websocket.ABNF.OPCODE_BINARY)
        except Exception as e:
            print("[Send Audio] Error sending:", e)
            stop_flag.set()

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16', callback=callback):
            print("🎙️ Recording... Press Ctrl+C to stop.")
            while not stop_flag.is_set() and ws.sock and ws.sock.connected:
                time.sleep(0.1)
    except Exception as e:
        print("[Audio] Input error:", e)
        stop_flag.set()


def on_message(ws, message):
    try:
        if isinstance(message, bytes):
            try:
                message = message.decode("utf-8")
            except UnicodeDecodeError:
                print("[Client] ⚠️ Received non-UTF8 binary message. Skipping.")
                return

        data = json.loads(message)
        msg_type = data.get("type")

        if msg_type == "status" and data.get("ready"):
            print("✅ Server is ready. Starting audio stream...")
            ready_flag.set()

        elif msg_type == "diarizedTranscript":
            print("📜 Diarized Transcript:")
            for seg in data.get("segments", []):
                speaker = seg.get("speaker", "unknown")
                text = seg.get("text", "")
                print(f"🗣 [{speaker}] {text}")

        else:
            print(f"[Client] ❓ Unknown message type: {msg_type}")

    except json.JSONDecodeError as e:
        print("[Client] ❌ JSON decode error:", e)
        print("[Client] Raw message (truncated):", str(message)[:100])
    except Exception as e:
        print("[Client] ❌ Unexpected error:", e)


def on_error(ws, error):
    print("❌ WebSocket error:", error)
    stop_flag.set()


def on_close(ws, close_status_code, close_msg):
    print("🔌 Connection closed.")
    stop_flag.set()


def on_open(ws):
    print("🌐 Connected to server.")

    def wait_and_stream():
        print("⌛ Waiting for server to be ready...")
        if not ready_flag.wait(timeout=20):
            print("❗ Server did not become ready in time.")
            stop_flag.set()
            try:
                ws.close()
            except:
                pass
            return
        send_audio(ws)

    threading.Thread(target=wait_and_stream, daemon=True).start()


def input_listener():
    global SRC_LANG_IDX, SRC_LANG
    print("👉 Nhấn 'n' rồi Enter để chuyển SRC_LANG (en <-> vi <-> ja)")
    while not stop_flag.is_set():
        cmd = input()
        if cmd.strip().lower() == 'n':
            with src_lang_lock:
                SRC_LANG_IDX = (SRC_LANG_IDX + 1) % len(LANGS)
                SRC_LANG = LANGS[SRC_LANG_IDX]
                print(f"🔁 SRC_LANG changed to: {SRC_LANG}")
        time.sleep(0.05)


if __name__ == "__main__":
    threading.Thread(target=input_listener, daemon=True).start()

    try:
        ws = websocket.WebSocketApp(
            SERVER_URL,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        ws.run_forever()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user.")
        stop_flag.set()
        try:
            ws.close()
        except:
            pass
        sys.exit(0)
