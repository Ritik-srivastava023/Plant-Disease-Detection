"""
Fit and serialize SVM models (with probability calibration) on cached penultimate features.
Saves:
- artifacts/mobilenet_svm.joblib
- artifacts/cnn_svm.joblib
"""
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).parent
OUTPUT_DIR = ROOT / "artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42

print("=" * 70)
print("SERIALIZING SOTA SVM CLASSIFIERS FOR STREAMLIT APP")
print("=" * 70)

# 1. MobileNetV2 + SVM (RBF)
print("\n[1/2] Training MobileNetV2 + SVM (RBF, calibrated probabilities)...")
mob_data = np.load(OUTPUT_DIR / "features" / "MobileNetV2_features.npz")
X_tr_mob, y_tr_mob = mob_data["train_feats"], mob_data["train_labels"]
X_te_mob, y_te_mob = mob_data["test_feats"], mob_data["test_labels"]

scaler_mob = StandardScaler()
X_tr_mob_std = scaler_mob.fit_transform(X_tr_mob)
X_te_mob_std = scaler_mob.transform(X_te_mob)

t0 = time.time()
svm_mob = SVC(kernel="rbf", probability=True, random_state=SEED)
svm_mob.fit(X_tr_mob_std, y_tr_mob)
tr_time_mob = time.time() - t0

y_pred_mob = svm_mob.predict(X_te_mob_std)
acc_mob = accuracy_score(y_te_mob, y_pred_mob) * 100.0
print(f"  MobileNetV2 + SVM Fit Time: {tr_time_mob:.2f}s | Test Accuracy: {acc_mob:.2f}%")

mob_payload = {
    "scaler": scaler_mob,
    "svm": svm_mob,
    "backbone": "MobileNetV2",
    "acc": acc_mob
}
joblib.dump(mob_payload, OUTPUT_DIR / "mobilenet_svm.joblib")
print(f"  Saved -> {OUTPUT_DIR / 'mobilenet_svm.joblib'}")

# 2. Custom CNN + SVM (RBF)
print("\n[2/2] Training Custom CNN + SVM (RBF, calibrated probabilities)...")
cnn_data = np.load(OUTPUT_DIR / "features" / "CNN_features.npz")
X_tr_cnn, y_tr_cnn = cnn_data["train_feats"], cnn_data["train_labels"]
X_te_cnn, y_te_cnn = cnn_data["test_feats"], cnn_data["test_labels"]

scaler_cnn = StandardScaler()
X_tr_cnn_std = scaler_cnn.fit_transform(X_tr_cnn)
X_te_cnn_std = scaler_cnn.transform(X_te_cnn)

t0 = time.time()
svm_cnn = SVC(kernel="rbf", probability=True, random_state=SEED)
svm_cnn.fit(X_tr_cnn_std, y_tr_cnn)
tr_time_cnn = time.time() - t0

y_pred_cnn = svm_cnn.predict(X_te_cnn_std)
acc_cnn = accuracy_score(y_te_cnn, y_pred_cnn) * 100.0
print(f"  Custom CNN + SVM Fit Time: {tr_time_cnn:.2f}s | Test Accuracy: {acc_cnn:.2f}%")

cnn_payload = {
    "scaler": scaler_cnn,
    "svm": svm_cnn,
    "backbone": "Custom CNN",
    "acc": acc_cnn
}
joblib.dump(cnn_payload, OUTPUT_DIR / "cnn_svm.joblib")
print(f"  Saved -> {OUTPUT_DIR / 'cnn_svm.joblib'}")

print("\nDone! Both SOTA SVM payloads serialized successfully.")

