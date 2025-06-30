import os
import torch
import numpy as np
from pyannote.audio import Model, Inference
from scipy.io import wavfile

# === Config ===
AUDIO_ROOT = "speaker_segments"
OUTPUT_PT = "speaker_memory.pt"
HF_TOKEN = os.getenv("HF_TOKEN")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MIN_DURATION = 1.0  # seconds
EMBEDDING_SIMILARITY_THRESHOLD = 0.6

# === Cosine similarity ===
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# === Speaker memory ===
class SpeakerMemory:
    def __init__(self):
        self.speakers = {}
        self.counter = 0

    def add(self, user_id, embeddings):
        self.speakers[user_id] = {
            "embeddings": embeddings,
            "metadata": {}
        }

    def save(self, path):
        torch.save({
            "counter": self.counter,
            "speakers": self.speakers
        }, path)
        print(f"[SAVE] speaker_memory saved to {path}")

# === Load model ===
embedding_model = Model.from_pretrained("pyannote/embedding", use_auth_token=HF_TOKEN)
inference = Inference(embedding_model, window="whole")
inference.to(torch.device(DEVICE))

def load_wav(filepath):
    sr, data = wavfile.read(filepath)
    if sr != 16000:
        raise ValueError(f"{filepath} must be 16kHz mono.")
    if len(data.shape) > 1:
        data = data[:, 0]  # mono
    return data.astype(np.float32) / 32768.0  # to float32 in range [-1, 1]

def extract_embedding(filepath):
    data = load_wav(filepath)
    duration = len(data) / 16000
    if duration < MIN_DURATION:
        print(f"[SKIP] {filepath} too short ({duration:.2f}s)")
        return None
    tensor = torch.tensor(data).unsqueeze(0)
    input_dict = {"waveform": tensor, "sample_rate": 16000}
    return inference(input_dict)

def compute_similarity_matrix(embeddings):
    n = len(embeddings)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim_matrix[i, j] = cosine_similarity(embeddings[i], embeddings[j])
    return sim_matrix

def select_representative_embeddings(embeddings, threshold=EMBEDDING_SIMILARITY_THRESHOLD, filenames=None):
    if len(embeddings) <= 1:
        return embeddings

    if filenames is None:
        filenames = [f"sample_{i+1}" for i in range(len(embeddings))]

    sim_matrix = compute_similarity_matrix(embeddings)

    n = len(embeddings)
    visited = [False] * n
    groups = []

    def dfs(i, group):
        visited[i] = True
        group.append(i)
        for j in range(n):
            if not visited[j] and sim_matrix[i][j] >= threshold:
                dfs(j, group)

    for i in range(n):
        if not visited[i]:
            group = []
            dfs(i, group)
            if group:
                groups.append(group)

    if not groups:
        print("  → ⚠️ Không tìm được nhóm embedding đồng nhất. Giữ lại tất cả.")
        return embeddings

    best_group = max(groups, key=len)
    print(f"  → ✅ Chọn nhóm {len(best_group)} embedding đồng nhất (similarity ≥ {threshold})")

    print("     ↳ Các cặp similarity score trong nhóm được chọn:")
    for i in range(len(best_group)):
        for j in range(i + 1, len(best_group)):
            a, b = best_group[i], best_group[j]
            score = sim_matrix[a][b]
            print(f"       - {filenames[a]} vs {filenames[b]}: similarity = {score:.4f}")

    print("  → ℹ️ Các nhóm embedding khác (không được chọn):")
    for idx, group in enumerate(groups):
        if group == best_group:
            continue
        print(f"     - Nhóm {idx+1} có {len(group)} embedding:")
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                score = sim_matrix[a][b]
                print(f"         {filenames[a]} vs {filenames[b]}: similarity = {score:.4f}")
        print("       ↳ Các file trong nhóm bị loại:", [filenames[i] for i in group])

    return [embeddings[i] for i in best_group]

def main():
    memory = SpeakerMemory()

    for user_folder in os.listdir(AUDIO_ROOT):
        user_path = os.path.join(AUDIO_ROOT, user_folder)
        if not os.path.isdir(user_path):
            continue

        print(f"[USER] Đang xử lý: {user_folder}")
        user_embeddings = []
        embedding_filenames = []

        for file in os.listdir(user_path):
            if not file.endswith(".wav"):
                continue
            path = os.path.join(user_path, file)
            embedding = extract_embedding(path)
            if embedding is not None:
                user_embeddings.append(embedding)
                embedding_filenames.append(file)

        if not user_embeddings:
            print(f"  → ⚠️ Không có audio hợp lệ cho {user_folder}")
            continue

        selected = select_representative_embeddings(user_embeddings, filenames=embedding_filenames)
        user_id = f"User_{user_folder}"
        memory.add(user_id, selected)
        print(f"[ADD] {user_id} với {len(selected)} embeddings")

    memory.save(OUTPUT_PT)

if __name__ == "__main__":
    main()
