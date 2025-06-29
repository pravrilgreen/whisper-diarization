import os
import sounddevice as sd
from scipy.io.wavfile import write
import time

def record_audio(filename, duration=5, fs=16000):
    print(f"\n🎤 Recording: {filename} ({duration}s)")
    print("👉 Please speak after the beep...")
    time.sleep(1)
    print("🔔 Beep!")
    
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    
    write(filename, fs, audio)
    print(f"✅ Saved: {filename}")

def main():
    duration = float(input("Enter duration per sample (in seconds): ").strip())
    num_samples = int(input("Enter number of samples to record: ").strip())

    output_dir = os.path.join("audio", speaker_name)
    os.makedirs(output_dir, exist_ok=True)

    for i in range(1, num_samples + 1):
        filename = f"sample_{i}.wav"
        record_audio(filename, duration=duration)

    print(f"\n✅ Done recording {num_samples} samples for speaker: {speaker_name}")

if __name__ == "__main__":
    main()
