import numpy as np
import torch
import os

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