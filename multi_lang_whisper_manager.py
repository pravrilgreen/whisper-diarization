from faster_whisper import WhisperModel
import threading
import queue
import torch

class TranscribeTask:
    def __init__(self, audio, lang, kwargs, result_callback):
        self.audio = audio
        self.lang = lang
        self.kwargs = kwargs
        self.result_callback = result_callback

class MultiLangWhisperQueueManager:
    def __init__(self, device="cuda"):
        self.device = device
        self.task_queues = {}
        self.models = {
            "en": WhisperModel(
                model_size_or_path="medium.en",
                device=device,
                compute_type="float16" if torch.cuda.is_available() else "int8"
            ),
            "ja": WhisperModel(
                model_size_or_path="kotoba-tech/kotoba-whisper-v2.0-faster",
                device=device,
                compute_type="float16" if torch.cuda.is_available() else "int8"
            ),
            "vi": WhisperModel(
                model_size_or_path="kiendt/PhoWhisper-large-ct2",
                device=device,
                compute_type="float16" if torch.cuda.is_available() else "int8"
            )
        }
        self.workers = {}
        for lang in self.models:
            self.task_queues[lang] = queue.Queue()
            t = threading.Thread(target=self.worker, args=(lang,), daemon=True)
            t.start()
            self.workers[lang] = t

    def worker(self, lang):
        model = self.models[lang]
        task_queue = self.task_queues[lang]
        while True:
            task = task_queue.get()
            try:
                result = model.transcribe(
                    task.audio,
                    language=lang,
                    beam_size=5,
                    vad_filter=True,
                    # word_timestamps=True,
                    chunk_length=15,
                    condition_on_previous_text=False,
                    **task.kwargs
                )
                if isinstance(result, tuple):
                    segments, info = result
                else:
                    segments = list(result)
                task.result_callback(segments)
            except Exception as e:
                print(f"[ERROR][worker {lang}] {e}")
                task.result_callback({"error": str(e)})
            task_queue.task_done()

    def transcribe(self, audio, language, **kwargs):
        if language not in self.models:
            raise ValueError("Unsupported language code!")
        result_data = {}
        event = threading.Event()
        def callback(result):
            result_data['result'] = result
            event.set()
        task = TranscribeTask(audio, language, kwargs, callback)
        self.task_queues[language].put(task)
        event.wait(timeout=600)
        if 'result' not in result_data:
            raise TimeoutError("Transcription timed out!")
        return result_data['result']