import os
import json
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, classification_report,
    roc_curve, auc
)

# 🎨 Tự động chọn màu chữ phù hợp với màu nền
def get_text_color(cell_value, max_value, threshold=0.4):
    # Nếu giá trị lớn thì chữ trắng, nhỏ thì chữ đen
    return 'white' if cell_value > max_value * threshold else 'black'

# 📊 Hàm vẽ ma trận nhầm lẫn
def draw_confusion_matrix(cm, title, filename, color_map='Blues', save=False):
    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"])
    disp.plot(cmap=color_map, ax=ax, colorbar=False)
    plt.title(title)
    plt.grid(False)

    # Hiển thị số liệu rõ ràng
    max_val = cm.max()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = get_text_color(cm[i, j], max_val)
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=12, color=color)

    if save:
        plt.savefig(filename)
        print(f"💾 Đã lưu {filename}")
    plt.show()

# 📈 Vẽ biểu đồ độ chính xác và độ lỗi từ file JSON
def plot_history(save=False):
    try:
        with open("lịch_sử_huấn_luyện.json", "r", encoding="utf-8") as f:
            history = json.load(f)
    except FileNotFoundError:
        print("❌ Không tìm thấy file 'lịch_sử_huấn_luyện.json'")
        return

    acc = [history["train_acc"], history["val_acc"]]
    loss = [history["train_loss"], history["val_loss"]]
    labels = ["Huấn luyện", "Kiểm tra"]

    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    bars = plt.bar(labels, acc, color=['mediumseagreen', 'darkorange'])
    plt.title("Accuracy")
    plt.ylim(0, 1)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.2f}", ha='center', fontsize=12)
    plt.ylabel("Tỉ lệ chính xác")

    # Loss
    plt.subplot(1, 2, 2)
    bars = plt.bar(labels, loss, color=['skyblue', 'tomato'])
    plt.title("Loss")
    plt.ylim(0, 1)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.2f}", ha='center', fontsize=12)
    plt.ylabel("Độ lỗi")

    plt.suptitle("Hiệu suất mô hình", fontsize=16)
    plt.tight_layout()
    if save:
        plt.savefig("biểu_đồ_độ_chính_xác_độ_lỗi.png")
        print("💾 Đã lưu biểu đồ Accuracy/Loss")
    plt.show()

# 🧪 Vẽ biểu đồ ROC
def plot_roc_curve(y_true, y_proba, title, filename):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='crimson', lw=2, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"💾 Đã lưu biểu đồ ROC: '{filename}'")
    plt.show()

# ✅ Đánh giá mô hình trên tập test đã cân bằng
def evaluate_test_set(save=False):
    try:
        data = np.load("tập_test_cân_bằng.npz")
        X_test, y_test = data["X_test"], data["y_test"]
        model = joblib.load("mô_hình_svm.pkl")
    except Exception as e:
        print(f"❌ Lỗi khi load dữ liệu hoặc mô hình: {e}")
        return

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    draw_confusion_matrix(cm, "Ma trận nhầm lẫn (Cân bằng)", "ma_trận_nhầm_lẫn_cân_bằng.png", "Blues", save)

    print("📄 Báo cáo phân loại (Cân bằng):")
    print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))
    auc_score = roc_auc_score(y_test, y_proba)
    print(f"🔍 AUC Score (Cân bằng): {auc_score:.4f}")

    if save:
        plot_roc_curve(y_test, y_proba, "ROC - Dữ liệu đã cân bằng", "biểu_đồ_ROC_cân_bằng.png")

# ⚖️ Đánh giá mô hình trên tập test chưa cân bằng
def evaluate_unbalanced_test_set(save=False):
    try:
        model = joblib.load("mô_hình_svm.pkl")
        X_test, y_test = joblib.load("tập_test_chưa_cân_bằng.pkl")
    except Exception as e:
        print(f"❌ Lỗi khi load dữ liệu chưa cân bằng: {e}")
        return

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    draw_confusion_matrix(cm, "Ma trận nhầm lẫn (Chưa cân bằng)", "ma_trận_nhầm_lẫn_chưa_cân_bằng.png", "Oranges", save)

    print("📄 Báo cáo phân loại (Chưa cân bằng):")
    print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))
    auc_score = roc_auc_score(y_test, y_proba)
    print(f"🔍 AUC Score (Chưa cân bằng): {auc_score:.4f}")

    if save:
        plot_roc_curve(y_test, y_proba, "ROC - Dữ liệu chưa cân bằng", "biểu_đồ_ROC_chưa_cân_bằng.png")

# 🚀 Chạy toàn bộ
if __name__ == "__main__":
    plot_history(save=True)
    evaluate_test_set(save=True)
    evaluate_unbalanced_test_set(save=True)
