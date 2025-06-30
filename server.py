from multi_lang_whisper_manager import MultiLangWhisperQueueManager
from pyannote.core import Segment, Annotation
from pyannote.audio import Pipeline, Model, Inference
from datetime import datetime
from collections import deque
import numpy as np
import websockets
import threading
import webrtcvad
import asyncio
import queue
import torch
import time
import wave
import json
import os
import io

# ======================= CONFIG ============================
HF_TOKEN = os.getenv("HF_TOKEN")
VAD_MODE = 1
SILERO_THRESHOLD = 0.8
MIN_SILENCE_DURATION = 0.2
# ===========================================================

DEVICE="cuda" if torch.cuda.is_available() else "cpu"
global_whisper_manager = MultiLangWhisperQueueManager(device=DEVICE)

class SpeakerMemory:
    def __init__(self, path="speaker_memory.pt"):
        self.speakers = {}
        self.counter = 0
        self.path = path
        self._load()
        print(f"[MEMORY] Total users in memory: {len(self.speakers)}")

    def add_new_speaker(self, embedding, metadata=None):
        embedding = embedding / np.linalg.norm(embedding)

        for user_id, info in self.speakers.items():
            for existing_embedding in info["embeddings"]:
                existing_embedding = existing_embedding / np.linalg.norm(existing_embedding)
                similarity = np.dot(embedding, existing_embedding)
                if similarity > 0.3: 
                    return None


        user_id = f"User_{self.counter}"
        self.counter += 1
        self.speakers[user_id] = {
            "embeddings": [embedding],
            "metadata": metadata or {}
        }
        self._save()
        print(f"[NEW SPEAKER] Added {user_id}")
        return user_id


    def identify_speaker(self, embedding, threshold=0.8):
        embedding = embedding / np.linalg.norm(embedding)
        best_match = None
        best_score = -1.0

        for user_id, info in self.speakers.items():
            all_scores = []
            for stored_embedding in info["embeddings"]:
                stored_embedding = stored_embedding / np.linalg.norm(stored_embedding)
                similarity = np.dot(embedding, stored_embedding)
                all_scores.append(similarity)

            if not all_scores:
                continue

            max_score = max(all_scores)
            if max_score > best_score and max_score >= threshold:
                best_score = max_score
                best_match = user_id

        return best_match

    def update_embedding(self, user_id, new_embedding, max_memory=20):
        new_embedding = new_embedding / np.linalg.norm(new_embedding)

        if user_id in self.speakers:
            for e in self.speakers[user_id]["embeddings"]:
                e = e / np.linalg.norm(e)
                similarity = np.dot(e, new_embedding)
                if similarity > 0.99:
                    return

            if any(np.dot(e / np.linalg.norm(e), new_embedding) > 0.8 for e in self.speakers[user_id]["embeddings"]):
                self.speakers[user_id]["embeddings"].append(new_embedding)
                if len(self.speakers[user_id]["embeddings"]) > max_memory:
                    self.speakers[user_id]["embeddings"] = self.speakers[user_id]["embeddings"][-max_memory:]
                self._save()


    def _save(self):
        data = {
            "counter": self.counter,
            "speakers": {
                user_id: {
                    "embeddings": [e for e in info["embeddings"]],
                    "metadata": info.get("metadata", {})
                }
                for user_id, info in self.speakers.items()
            }
        }
        torch.save(data, self.path)

    def _load(self):
        if os.path.exists(self.path):
            try:
                data = torch.load(self.path, weights_only = False)
                self.counter = data.get("counter", 0)
                self.speakers = data.get("speakers", {})
                print(f"[LOAD] Loaded {len(self.speakers)} speakers.")
            except Exception as e:
                print(f"[LOAD ERROR] {e}")

class ClientSession:
    def __init__(self, websocket):
        self.websocket = websocket
        self.audio_queue = asyncio.Queue()
        self.inference_queue = asyncio.Queue()
        self.stop_flag = threading.Event()

        self.language = None
        self.user = None

        self.audio_buffer = bytearray()
        self.vad_buffer = bytearray()
        self.pre_buffer = deque(maxlen=16000)
        self.noise_profile = deque(maxlen=16000 * 3)
        self.last_voice_time = 0

        self.vad = webrtcvad.Vad()
        self.vad.set_mode(VAD_MODE)

        self.silero_buffer = bytearray()
        self.is_webrtc_speech_active = False
        self.is_silero_speech_active = False

        self.session_speaker_map = {}
        self.speaker_memory = SpeakerMemory(path="speaker_memory.pt")

        self.embedding_model = Model.from_pretrained("pyannote/embedding", use_auth_token=HF_TOKEN)
        self.inference = Inference(self.embedding_model, window="whole")
        self.inference.to(torch.device(DEVICE))

        self.silero_vad_model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", verbose=False, onnx=False)
        print("🔧 Loading diarization pipeline...")
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN
        )
        self.diarization_pipeline.to(torch.device(DEVICE))

        self.whisper_model = global_whisper_manager
        self._transcribe_result = None
        self._event = threading.Event()

        self.silero_queue = queue.Queue()
        self.silero_thread = threading.Thread(target=self._silero_worker, daemon=True)
        self.silero_thread.start()

    def _save_transcript_audio(self, pcm_bytes):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join("transcript_audio", f"unknown_{timestamp}.wav")
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # int16 = 2 bytes
            wf.setframerate(16000)
            wf.writeframes(pcm_bytes)
        print(f"[DEBUG] Saved transcript audio: {filename}")

    def _is_webrtc_speech(self, pcm_bytes):
        for i in range(0, len(pcm_bytes), 960):
            frame = pcm_bytes[i:i + 960]
            if len(frame) == 960 and self.vad.is_speech(frame, 16000):
                return True
        return False

    def _silero_worker(self):
        while True:
            pcm_bytes = self.silero_queue.get()
            if pcm_bytes is None:
                break

            audio_np = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            if len(audio_np) < 512:
                continue

            for i in range(0, len(audio_np) - 512 + 1, 512):
                frame = audio_np[i:i + 512]
                tensor = torch.from_numpy(frame).unsqueeze(0)
                try:
                    prob = self.silero_vad_model(tensor, 16000).item()
                    if prob > SILERO_THRESHOLD:
                        self.is_silero_speech_active = True
                        break
                except:
                    pass
            else:
                self.is_silero_speech_active = False

    def _check_voice_activity(self, pcm_bytes):
        self.is_webrtc_speech_active = self._is_webrtc_speech(pcm_bytes)

        if self.is_webrtc_speech_active:
            self.silero_buffer.extend(pcm_bytes)
            if len(self.silero_buffer) >= 1024:
                chunk = self.silero_buffer[:1024]
                self.silero_buffer = self.silero_buffer[1024:]
                self.silero_queue.put(chunk)
        else:
            self.silero_buffer.clear()
            if not self.is_silero_speech_active:
                self.noise_profile.extend(np.frombuffer(pcm_bytes, dtype=np.int16))

    def _is_voice_active(self):
        return self.is_webrtc_speech_active and self.is_silero_speech_active

    def _save_speaker_audio(self, audio_array, speaker_id, start, end):
        folder = "speaker_segments"
        os.makedirs(folder, exist_ok=True)

        timestamp = f"{start:.2f}-{end:.2f}".replace(".", "_")
        filename = os.path.join(folder, f"{speaker_id}_{timestamp}.wav")

        audio_int16 = (np.clip(audio_array, -1.0, 1.0) * 32767).astype(np.int16)

        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_int16.tobytes())

        print(f"[AUDIO] Saved segment for {speaker_id}: {filename}")

    async def send_messages(self):
        try:
            while not self.stop_flag.is_set():
                msg = await self.audio_queue.get()
                await self.websocket.send(msg)
        except:
            pass

    def run_diarization(self, audio_bytes):
        wav_io = io.BytesIO()
        with wave.open(wav_io, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_bytes)
        wav_io.seek(0)

        try:
            diarization = self.diarization_pipeline(wav_io)
            speakers_detected = set(label for _, _, label in diarization.itertracks(yield_label=True))
            print(f"[DEBUG] Detected speakers: {speakers_detected} (count: {len(speakers_detected)})")

            print(f"[INFO] Diarization: {len(set(label for _, _, label in diarization.itertracks(yield_label=True)))} speakers.")
            return diarization
        except Exception as e:
            print(f"[ERROR] Diarization failed: {e}")
            return None

    def process_diarization(self, diarization, audio_np, duration):
        speaker_mapping = {}
        if not diarization:
            return speaker_mapping

        # Convert diarization to list of segments
        segments = list(diarization.itertracks(yield_label=True))
        unique_speakers = list({label for _, _, label in segments})

        if len(unique_speakers) == 1:
            # Single speaker: assign entire audio to them
            full_segment = Segment(0.0, duration)
            speaker_label = unique_speakers[0]
            segments = [(full_segment, None, speaker_label)]

        elif len(unique_speakers) == 2:
            # Fill missing gaps between speaker segments
            segments.sort(key=lambda x: x[0].start)
            filled_segments = []
            last_end = 0.0

            for seg, _, label in segments:
                if seg.start > last_end:
                    filler_seg = Segment(last_end, seg.start)
                    filler_label = label  # assign to next speaker for simplicity
                    filled_segments.append((filler_seg, None, filler_label))
                filled_segments.append((seg, None, label))
                last_end = seg.end

            if last_end < duration:
                filled_segments.append((Segment(last_end, duration), None, segments[-1][2]))

            segments = filled_segments

        else:
            # Skip if more than 2 speakers
            print("[SKIP] More than 2 speakers — skipping processing")
            return {}

        print(f"[DIARIZATION] Processed {len(segments)} speaker segments (1-2 speakers enforced).")

        for turn, _, label in segments:
            segment = Segment(turn.start, turn.end)
            if segment.end > duration + 0.1:
                continue

            try:
                start_sample = int(segment.start * 16000)
                end_sample = int(segment.end * 16000)
                segment_audio = audio_np[start_sample:end_sample]
                seg_duration = segment.end - segment.start

                # if seg_duration < 1.0 or len(segment_audio) < 2400:
                #     continue

                waveform_tensor = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)
                input_dict = {"waveform": waveform_tensor, "sample_rate": 16000}
                embedding_np = self.inference.crop(input_dict, segment)
                embedding = embedding_np / np.linalg.norm(embedding_np)
                if embedding is None or not isinstance(embedding, np.ndarray):
                    continue
            
                matched_user_id = None
                best_user = None
                best_score = -1.0

                for user_id, info in self.speaker_memory.speakers.items():
                    scores = [np.dot(embedding, e / np.linalg.norm(e)) for e in info["embeddings"]]
                    if not scores:
                        continue
                    max_score = max(scores)
                    if max_score > best_score:
                        best_score = max_score
                        best_user = user_id

                print(f"[SEGMENT] {segment.start:.2f}–{segment.end:.2f}s")

                for user_id, info in self.speaker_memory.speakers.items():
                    scores = [
                        np.dot(embedding, e / np.linalg.norm(e))
                        for e in info["embeddings"]
                    ]
                    if not scores:
                        continue
                    max_score = max(scores)
                    if max_score > best_score:
                        best_score = max_score
                        best_user = user_id

                if seg_duration >= 3.0:
                    if best_score >= 0.8:
                        matched_user_id = best_user
                        print(f"[RECOGNITION] {matched_user_id} matched (score: {best_score:.4f})")
                    else:
                        matched_user_id = self.speaker_memory.add_new_speaker(embedding)
                        if matched_user_id:
                            print(f"[ADD] New user added: {matched_user_id}")
                        else:
                            print(f"[RECOGNITION] Rejected, too similar to {best_user} (score: {best_score:.4f})")
                            matched_user_id = best_user if best_user else "unknown_user"
                else:
                    if best_score >= 0.8:
                        matched_user_id = best_user
                        print(f"[RECOGNITION] {matched_user_id} matched (score: {best_score:.4f})")
                    else:
                        if best_user:
                            print(f"[RECOGNITION] Closest match: {best_user} (score: {best_score:.4f})")
                            matched_user_id = best_user
                        else:
                            print(f"[RECOGNITION] Unknown speaker — no users in memory")
                            matched_user_id = "unknown_user"

                if matched_user_id != "unknown_user":
                    num_embeddings = len(self.speaker_memory.speakers[matched_user_id]["embeddings"])
                    active = list(diarization.get_labels(segment))
                    is_overlap = len(active) > 1
                    update_allowed = (seg_duration >= 4.0 or num_embeddings < 3) and not is_overlap

                    if update_allowed:
                        self.speaker_memory.update_embedding(matched_user_id, embedding)
                        total = len(self.speaker_memory.speakers[matched_user_id]["embeddings"])
                        print(f"[UPDATE] {matched_user_id} updated → total embeddings: {total}")
                    else:
                        print(f"[SKIP] No update for {matched_user_id} (embeddings: {num_embeddings})")

                speaker_mapping[(segment.start, segment.end)] = matched_user_id
                #self._save_speaker_audio(segment_audio, speaker_id=matched_user_id, start=segment.start, end=segment.end)
                #print(f"[SAVE] Segment saved → {matched_user_id} [{segment.start:.2f}–{segment.end:.2f}]")

            except Exception:
                continue

        return speaker_mapping

    def transcribe_audio(self, audio_np):
        segments = self.whisper_model.transcribe(
            audio_np,
            language=self.language,
        )
        segments = list(segments)
        return segments
    
    def assign_speakers_to_segments(self, segments, speaker_mapping):
        results = []
        segments = list(segments)

        annotation = Annotation()
        for (start, end), speaker in speaker_mapping.items():
            annotation[Segment(start, end)] = speaker

        for segment in segments:
            seg_text = segment.text.strip()
            seg_start, seg_end = segment.start, segment.end
            seg = Segment(seg_start, seg_end)

            cropped = annotation.crop(seg, mode="loose")
            speakers = cropped.labels()

            if not speakers:
                speaker = "unknown_user"
            elif len(speakers) == 1:
                speaker = speakers[0]
            else:
                durations = {
                    spk: cropped.label_duration(spk)
                    for spk in speakers
                }
                speaker = max(durations, key=durations.get)

            results.append({
                "start": seg_start,
                "end": seg_end,
                "text": seg_text,
                "speaker": speaker
            })

        return results

    async def process_audio_worker(self):
        while not self.stop_flag.is_set():
            try:
                audio_bytes = await self.inference_queue.get()

                audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                duration = len(audio_np) / 16000.0

                print(f"[INFO] Processing {duration:.2f} seconds of audio...")

                diarization = self.run_diarization(audio_bytes)
                speaker_mapping = self.process_diarization(diarization, audio_np, duration)

                segments = self.transcribe_audio(audio_np)

                results = self.assign_speakers_to_segments(segments, speaker_mapping)

                if results:
                    await self.audio_queue.put(json.dumps({
                        "type": "diarizedTranscript",
                        "segments": results,
                    }))

            except Exception as e:
                print(f"[CRITICAL] Error in process_audio_worker: {e}")


    async def receive_audio(self):
        try:
            async for message in self.websocket:
                if isinstance(message, bytes) and len(message) >= 4:
                    hdr_len = int.from_bytes(message[:4], 'little')
                    header = json.loads(message[4:4 + hdr_len].decode("utf-8"))
                    pcm16k = message[4 + hdr_len:]

                    language = header.get("language")
                    if language and language != self.language:
                        print(f"[LANG] Language changed → {self.language} → {language}")
                        self.language = language

                    self.vad_buffer.extend(pcm16k)

                    while len(self.vad_buffer) >= 960:
                        frame = bytes(self.vad_buffer[:960])
                        self.vad_buffer = self.vad_buffer[960:]
                        self._check_voice_activity(frame)

                    self.pre_buffer.extend(pcm16k)
                    now = time.time()

                    if self._is_voice_active():
                        self.last_voice_time = now
                        if not self.audio_buffer:
                            self.audio_buffer.extend(self.pre_buffer)
                            self.pre_buffer.clear()
                        self.audio_buffer.extend(pcm16k)
                    else:
                        if now - self.last_voice_time < MIN_SILENCE_DURATION:
                            self.audio_buffer.extend(pcm16k)
                        elif len(self.audio_buffer) > 0:
                            await self.inference_queue.put(bytes(self.audio_buffer))
                            self.audio_buffer.clear()
                            self.pre_buffer.clear()
        except:
            pass

    async def run(self):
        await self.websocket.send(json.dumps({"type": "status", "ready": True}))
        sender = asyncio.create_task(self.send_messages())
        receiver = asyncio.create_task(self.receive_audio())
        processor = asyncio.create_task(self.process_audio_worker())

        done, pending = await asyncio.wait([sender, receiver, processor], return_when=asyncio.FIRST_COMPLETED)
        self.stop_flag.set()
        for task in pending:
            task.cancel()
        self.silero_queue.put(None)

async def handler(websocket):
    session = ClientSession(websocket)
    await session.run()

async def main():
    async with websockets.serve(handler, "0.0.0.0", 8012):
        print("✅ Server listening on ws://0.0.0.0:8012")
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("❌ Server stopped manually.")
 