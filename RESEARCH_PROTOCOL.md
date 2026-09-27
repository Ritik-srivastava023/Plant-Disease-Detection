# Plant Disease Classification: Research Evaluation Protocol

## 1. Fixed Data Protocol

- **Dataset:** 20,638 labelled PlantVillage images across 15 agricultural classes (Bell Pepper, Potato, Tomato).
- **Split:** Stratified 70% train (14,446 images), 15% validation (3,096 images), 15% untouched test (3,096 images).
- **Random seed:** `42` (all splits, initializations, and shuffles).
- **Data Leakage Prevention:** Training data may be augmented (rotation, zoom, horizontal flip). Validation and test splits must NEVER be augmented and must evaluate with `shuffle=False`.
- **Decision Integrity:** Validation data select checkpoints and trigger early stopping. Final research claims are exclusively drawn from the untouched 3,096-image test split.

---

## 2. Deep Learning Architectural Benchmark

We evaluate three distinct deep learning paradigms to study the trade-off between edge efficiency and classification capacity:

| Paradigm | Architecture | Primary Research Objective |
| :--- | :--- | :--- |
| **Scratch Baseline** | **Custom CNN (GAP + BatchNorm)** | Establishes domain performance without pre-trained ImageNet priors; ultra-lightweight (<200k params). |
| **Edge Lightweight** | **MobileNetV2 (Two-Stage Fine-Tuned)** | Inverted residual bottleneck design optimized for rural farmer mobile/edge deployment. |
| **High-Capacity SOTA** | **EfficientNet-B0 (Two-Stage Fine-Tuned)** | Compound-scaled architecture with Squeeze-and-Excitation attention for maximum diagnostic capability. |

---

## 3. Standard Evaluation & Latency Metrics

For every model, report performance against the untouched test split:

| Model | Paradigm | Params (M) | Size (MB) | CPU Latency (ms/img) | Test Acc (%) | Macro F1 (%) | Weighted F1 (%) | Macro OvR AUC |
| :--- | :--- | ---:| ---:| ---:| ---:| ---:| ---:| ---:|
| **Custom CNN (GAP + BN)** | Scratch Baseline | 0.113 | 0.48 | 18.46 | 68.12 | 63.03 | 67.27 | 0.9576 |
| **MobileNetV2 (Two-Stage)** | Edge Lightweight | 2.429 | 11.51 | 87.78 | 94.57 | 93.57 | 94.48 | 0.9987 |
| **EfficientNet-B0 (Two-Stage)** | High-Capacity SOTA | 4.221 | 17.50 | 151.89 | 95.70 | 94.94 | 95.74 | 0.9991 |

> [!NOTE]
> **Ablation & Parameter Trade-Off:**
> The Custom CNN baseline was deliberately compressed to an ultra-lightweight footprint (<200k params) using GlobalAveragePooling2D and Batch Normalization. While achieving an inference latency of 18.46 ms/img (54.2 img/s) and a compact 0.48 MB disk size, its scratch representation capacity is constrained to 68.12% test accuracy. In contrast, pre-trained transfer learning architectures (MobileNetV2 and EfficientNet-B0) leverage rich hierarchical ImageNet feature priors to break past 94% accuracy with negligible diagnostic error.

Artifacts generated:
- Model confusion matrix (`ConfusionMatrixDisplay`)
- Per-class classification report (precision, recall, F1 per disease class)
- One-vs-Rest ROC curves with micro and per-class AUC
- Row-level test predictions CSV with all 15 class probabilities (`cnn_test_predictions.csv`, `mobilenet_test_predictions.csv`, `efficientnet_test_predictions.csv`)
- Latency and efficiency trade-off figure (`artifacts/model_comparison.png`)

---

## 4. Transfer Learning Protocol (Two-Stage Fine-Tuning)

For transfer learning models (`mobilenet.ipynb` and `efficientnet.ipynb`):
1. **Stage 1 (Feature Extraction / Warmup):**
   - Backbone frozen (`base_model.trainable = False`).
   - Train custom classification head with `Adam(learning_rate=1e-3)` for up to 15 epochs with early stopping.
2. **Stage 2 (Fine-Tuning Domain Adaptation):**
   - Unfreeze upper layers of the backbone (layers from 100+ for MobileNetV2, 180+ for EfficientNet-B0).
   - Recompile with a low learning rate: `Adam(learning_rate=1e-5)`.
   - Train for remaining epochs with `ReduceLROnPlateau(patience=2, min_lr=1e-7)` and `EarlyStopping`.

---

## 5. Robustness & Explainable AI (XAI)

- **Grad-CAM Visual Explanations:**
  - Produce heatmaps for both correct diagnoses and edge-case classification mistakes.
  - Verify that activations focus on leaf tissue and pathogen-induced lesions (e.g. concentric rings of Early Blight vs water-soaked lesions of Late Blight) rather than laboratory background surfaces.
- **Seeds for Statistical Significance:**
  - Evaluate the final configuration across seeds 42, 123, and 2026.
  - Report metrics as $\mu \pm \sigma$.

---

## 6. Real-World Claim Boundaries

- **Laboratory vs. Field Conditions:**
  PlantVillage images were collected under controlled laboratory illumination with uniform backgrounds.
- Models deployed in real agricultural fields require external validation on in-the-wild datasets (e.g., PlantDoc) to quantify domain shift robustness.
