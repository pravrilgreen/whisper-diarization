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
    print(f"[💾] Đã lưu lại vào: {path}")

def list_speakers(data):
    speakers = sorted(data["speakers"].keys())
    print("\n📋 Danh sách người nói:")
    for idx, user_id in enumerate(speakers):
        embeddings = data["speakers"][user_id]
        count = len(embeddings) if hasattr(embeddings, '__len__') else 'N/A'
        print(f" [{idx}] {user_id} - {count} embedding(s)")
    return speakers

def rename_speaker(data, old_id, new_id):
    if old_id not in data["speakers"]:
        print(f"[❌] Không tìm thấy: {old_id}")
        return False
    if new_id in data["speakers"]:
        print(f"[⚠] Nhãn '{new_id}' đã tồn tại.")
        return False
    data["speakers"][new_id] = data["speakers"].pop(old_id)
    print(f"[✔] Đã đổi tên '{old_id}' thành '{new_id}'")
    return True

def main():
    parser = argparse.ArgumentParser(description="Đổi tên người nói trong file speaker_memory.pt")
    parser.add_argument("--path", type=str, default="speaker_memory.pt", help="Đường dẫn tới file .pt")
    args = parser.parse_args()

    data = load_speaker_memory(args.path)

    while True:
        speakers = list_speakers(data)
        choice = input("\nNhập số thứ tự người muốn đổi tên (hoặc 'q' để thoát): ").strip()
        if choice.lower() == 'q':
            break
        if not choice.isdigit() or int(choice) >= len(speakers):
            print("[⚠] Lựa chọn không hợp lệ. Thử lại.")
            continue

        index = int(choice)
        old_id = speakers[index]
        print(f"\n🔧 Bạn đang chọn đổi tên: [{index}] {old_id}")
        confirm = input("Bạn có muốn tiếp tục? (y để tiếp tục, bất kỳ phím nào khác để quay lại): ").strip().lower()
        if confirm != 'y':
            print("[↩] Quay lại chọn người khác.")
            continue

        new_id = input(f"Nhập nhãn mới cho '{old_id}': ").strip()
        if not new_id:
            print("[⚠] Nhãn không được để trống. Hủy đổi tên.")
            continue

        success = rename_speaker(data, old_id, new_id)
        if success:
            save_speaker_memory(data, args.path)

if __name__ == "__main__":
    main()
