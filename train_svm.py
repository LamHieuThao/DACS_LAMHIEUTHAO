import os
import glob
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_curve, auc, classification_report
import joblib
import json

def extract_mfcc(audio_path, n_mfcc=13, n_fft=2048, hop_length=512):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        if len(y) < n_fft:
            print(f"[SHORT] {audio_path} - too short ({len(y)} samples)")
            return None
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        print(f"[ERROR] {audio_path}: {e}")
        return None

def load_dataset(path, label):
    features, labels = [], []
    files = glob.glob(os.path.join(path, "*.wav"))
    print(f"Loading from {path} - Found {len(files)} files")
    for f in files:
        mfcc = extract_mfcc(f)
        if mfcc is not None:
            features.append(mfcc)
            labels.append(label)
    return features, labels

def balance_data(X, y):
    X = np.array(X)
    y = np.array(y)
    class0 = X[y == 0]
    class1 = X[y == 1]

    if len(class0) > len(class1):
        class1 = resample(class1, replace=True, n_samples=len(class0), random_state=42)
    else:
        class0 = resample(class0, replace=True, n_samples=len(class1), random_state=42)

    X_balanced = np.vstack((class0, class1))
    y_balanced = np.array([0] * len(class0) + [1] * len(class1))

    return X_balanced, y_balanced

def vẽ_roc(y_true, y_score, tiêu_đề, tên_tệp):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(tiêu_đề)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(tên_tệp)
    plt.close()
    print(f"💾 Đã lưu {tên_tệp}")

def vẽ_roc_so_sánh(y_trains, y_probas, labels, title, filename):
    plt.figure()
    for y_true, y_score, label in zip(y_trains, y_probas, labels):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    print(f"💾 Đã lưu {filename}")

def lưu_báo_cáo_thành_ảnh(report_text, filename, title):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')
    ax.text(0, 1, title + "\n\n" + report_text, fontsize=10, va='top', family='monospace')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"🖼️ Đã lưu {filename}")

def train():
    real_dir = r"D:\DeepFake-Audio-Detection-MFCC-main\DeepFake-Audio-Detection-MFCC-main\real_audio"
    fake_dir = r"D:\DeepFake-Audio-Detection-MFCC-main\DeepFake-Audio-Detection-MFCC-main\deepfake_audio"

    X_real, y_real = load_dataset(real_dir, 0)
    X_fake, y_fake = load_dataset(fake_dir, 1)

    print(f"Loaded Real: {len(X_real)}, Fake: {len(X_fake)}")

    X, y = X_real + X_fake, y_real + y_fake

    # ➕ Test chưa cân bằng
    X_orig_train, X_orig_test, y_orig_train, y_orig_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    scaler_orig = StandardScaler()
    X_orig_train_scaled = scaler_orig.fit_transform(X_orig_train)
    X_orig_test_scaled = scaler_orig.transform(X_orig_test)

    joblib.dump(scaler_orig, "bộ_chuẩn_hóa_chưa_cân_bằng.pkl")
    joblib.dump((X_orig_test_scaled, y_orig_test), "tập_test_chưa_cân_bằng.pkl")

    # ➕ Cân bằng dữ liệu
    X_bal, y_bal = balance_data(X, y)
    print(f"After balancing: Total = {len(X_bal)} samples")

    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.2, stratify=y_bal, random_state=42)

    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVC(kernel='linear', probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Dự đoán
    y_train_pred = model.predict(X_train_scaled)
    y_train_proba = model.predict_proba(X_train_scaled)[:, 1]

    y_test_pred = model.predict(X_test_scaled)
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]

    y_orig_test_proba = model.predict_proba(X_orig_test_scaled)[:, 1]

    # ROC từng tập
    vẽ_roc(y_train, y_train_proba, "ROC - Tập Huấn Luyện", "roc_train.png")
    vẽ_roc(y_test, y_test_proba, "ROC - Tập Test Cân Bằng", "roc_test_balanced.png")
    vẽ_roc(y_orig_test, y_orig_test_proba, "ROC - Tập Test Chưa Cân Bằng", "roc_test_unbalanced.png")

    # So sánh ROC
    vẽ_roc_so_sánh(
        [y_train, y_test, y_orig_test],
        [y_train_proba, y_test_proba, y_orig_test_proba],
        ["Train", "Test Cân Bằng", "Test Chưa Cân Bằng"],
        "So Sánh ROC Các Tập",
        "roc_so_sanh.png"
    )

    vẽ_roc_so_sánh(
        [y_test, y_orig_test],
        [y_test_proba, y_orig_test_proba],
        ["Test Cân Bằng", "Test Chưa Cân Bằng"],
        "So Sánh ROC - Test Cân Bằng vs Chưa Cân Bằng",
        "roc_test_comparison_bal_vs_unbal.png"
    )

    # 📋 Báo cáo phân loại
    train_report = classification_report(y_train, y_train_pred, target_names=["Real", "Fake"])
    test_report = classification_report(y_test, y_test_pred, target_names=["Real", "Fake"])

    print("\n📋 Báo Cáo - Tập Huấn Luyện:")
    print(train_report)
    print("📋 Báo Cáo - Tập Test Cân Bằng:")
    print(test_report)

    # Ghi file text
    with open("báo_cáo_đánh_giá.txt", "w", encoding="utf-8") as f:
        f.write("📋 Báo Cáo - Tập Huấn Luyện:\n")
        f.write(train_report)
        f.write("\n📋 Báo Cáo - Tập Test Cân Bằng:\n")
        f.write(test_report)

    # Vẽ báo cáo thành ảnh
    lưu_báo_cáo_thành_ảnh(train_report, "báo_cáo_train.png", "Báo Cáo Huấn Luyện")
    lưu_báo_cáo_thành_ảnh(test_report, "báo_cáo_test.png", "Báo Cáo Test Cân Bằng")

    # Ghi lại kết quả
    history = {
        "train_acc": accuracy_score(y_train, y_train_pred),
        "val_acc": accuracy_score(y_test, y_test_pred),
        "train_loss": 1 - accuracy_score(y_train, y_train_pred),
        "val_loss": 1 - accuracy_score(y_test, y_test_pred)
    }

    with open("lịch_sử_huấn_luyện.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False)

    # Lưu mô hình và dữ liệu
    joblib.dump(model, "mô_hình_svm.pkl")
    joblib.dump(scaler, "bộ_chuẩn_hóa.pkl")
    np.savez("tập_test_cân_bằng.npz", X_test=X_test_scaled, y_test=y_test)

    print("✅ Huấn luyện hoàn tất. Đã lưu mô hình, scaler, dữ liệu, biểu đồ và báo cáo.")

if __name__ == "__main__":
    train()
