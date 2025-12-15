import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def visualize_mfcc_both(file_path):
    if not os.path.exists(file_path):
        print("❌ File không tồn tại:", file_path)
        return
    elif not file_path.lower().endswith(".wav"):
        print("❌ File không phải định dạng WAV:", file_path)
        return

    print("✅ Đang xử lý file:", file_path)
    y, sr = librosa.load(file_path, sr=None)

    # Trích xuất MFCC đầy đủ (dạng ma trận)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # ===== Biểu đồ 1: MFCC CHƯA sàng lọc (biểu diễn theo thời gian) =====
    plt.figure(figsize=(10, 4))
    for i in range(mfcc.shape[0]):  # Duyệt qua từng chỉ số MFCC
        plt.plot(mfcc[i, :], label=f'MFCC {i+1}')  # Hiển thị theo từng chỉ số MFCC
    plt.title("MFCC chưa sàng lọc (dạng tuyến tính theo thời gian)")
    plt.xlabel("Thời gian")
    plt.ylabel("Giá trị MFCC")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ===== Biểu đồ 2: MFCC SAU khi sàng lọc (lấy trung bình) =====
    mfcc_mean = np.mean(mfcc, axis=1)  # Trung bình theo thời gian (trục 1)

    plt.figure(figsize=(8, 4))
    plt.plot(mfcc_mean, marker='o')
    plt.title("MFCC sau khi sàng lọc (trung bình theo thời gian)")
    plt.xlabel("Chỉ số MFCC (1 → 13)")
    plt.ylabel("Giá trị trung bình")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    file_path = r"D:\DeepFake-Audio-Detection-MFCC-main\DeepFake-Audio-Detection-MFCC-main\\real_audio\\vinh-_1_.wav"
    visualize_mfcc_both(file_path)
