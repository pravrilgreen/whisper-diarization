import torch
import argparse
import os

def load_speaker_memory(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    data = torch.load(path)
    return data

def save_speaker_memory(data, path):
    torch.save(data, path)
    print(f"[💾] Saved to: {path}")

def list_speakers(data):
    speakers = sorted(data["speakers"].keys())
    print("\n📋 List of speakers:")
    for idx, user_id in enumerate(speakers):
        embeddings = data["speakers"][user_id]
        count = len(embeddings) if hasattr(embeddings, '__len__') else 'N/A'
        print(f" [{idx}] {user_id} - {count} embedding(s)")
    return speakers

def rename_speaker(data, old_id, new_id):
    if old_id not in data["speakers"]:
        print(f"[❌] Not found: {old_id}")
        return False
    if new_id in data["speakers"]:
        print(f"[⚠] Label '{new_id}' already exists.")
        return False
    data["speakers"][new_id] = data["speakers"].pop(old_id)
    print(f"[✔] Renamed '{old_id}' to '{new_id}'")
    return True

def main():
    parser = argparse.ArgumentParser(description="Rename a speaker in the speaker_memory.pt file")
    parser.add_argument("--path", type=str, default="speaker_memory.pt", help="Path to the .pt file")
    args = parser.parse_args()

    data = load_speaker_memory(args.path)

    while True:
        speakers = list_speakers(data)
        choice = input("\nEnter the speaker's index you want to rename (or 'q' to quit): ").strip()
        if choice.lower() == 'q':
            break
        if not choice.isdigit() or int(choice) >= len(speakers):
            print("[⚠] Invalid choice. Please try again.")
            continue

        index = int(choice)
        old_id = speakers[index]
        print(f"\n🔧 You are about to rename: [{index}] {old_id}")
        confirm = input("Do you want to proceed? (y to continue, any other key to go back): ").strip().lower()
        if confirm != 'y':
            print("[↩] Going back to choose another speaker.")
            continue

        new_id = input(f"Enter the new label for '{old_id}': ").strip()
        if not new_id:
            print("[⚠] Label cannot be empty. Canceling rename.")
            continue

        success = rename_speaker(data, old_id, new_id)
        if success:
            save_speaker_memory(data, args.path)

if __name__ == "__main__":
    main()
