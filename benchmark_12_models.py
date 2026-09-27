"""
12-Model Benchmark: 3 trained backbones (Custom CNN, MobileNetV2, EfficientNet-B0)
each paired with 4 classical classifier heads (SVM, KNN, RF, Gaussian NB),
trained on penultimate-layer (128-dim) features -- mirroring the reference structure.

Requires: tensorflow, scikit-learn, h5py, pandas, numpy
"""
from pathlib import Path
import time
import h5py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input as eff_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mob_preprocess
from tensorflow.keras.preprocessing.image import ImageDataGenerator

ROOT = Path(__file__).parent
DATASET_PATH = ROOT / "PlantVillage"
OUTPUT_DIR = ROOT / "artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUTPUT_DIR / "features"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
SEED = 42
tf.keras.utils.set_random_seed(SEED)

# ----------------------------------------------------------------------
# 1. Fixed split -- IDENTICAL to the split used everywhere else in the project
# ----------------------------------------------------------------------
print("=" * 80)
print("LOADING DATASET & STRATIFIED SPLIT (Seed 42)")
print("=" * 80)

records = [
    {'filepath': str(f), 'label': f.parent.name}
    for f in sorted(DATASET_PATH.glob('*/*'))
    if f.suffix.lower() in IMAGE_EXTENSIONS
]
image_df = pd.DataFrame(records)
class_names = sorted(image_df['label'].unique())
n_classes = len(class_names)

train_df, temporary_df = train_test_split(
    image_df, test_size=0.30, stratify=image_df['label'], random_state=SEED
)
val_df, test_df = train_test_split(
    temporary_df, test_size=0.50, stratify=temporary_df['label'], random_state=SEED
)
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}, Classes: {n_classes}")

# ----------------------------------------------------------------------
# 2. Load each backbone and build a feature extractor from its
#    penultimate (128-unit) layer -- the layer just before the final
#    softmax Dense. All three backbones share this head shape:
#    GAP -> BatchNorm -> Dense(128) -> Dropout -> Dense(n_classes, softmax)
# ----------------------------------------------------------------------
def load_efficientnet_model(weights_path):
    print("  Instantiating EfficientNet-B0 base (ImageNet priors)...")
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    inputs = tf.keras.Input(shape=(128, 128, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    bn = layers.BatchNormalization(name='batch_normalization_1')(x)
    d1 = layers.Dense(128, activation='relu', name='dense_2')(bn)
    dp = layers.Dropout(0.3)(d1)
    outputs = layers.Dense(n_classes, activation='softmax', name='dense_3')(dp)
    model = models.Model(inputs=inputs, outputs=outputs, name="EfficientNetB0_Classifier")

    print(f"  Loading weights from {weights_path}...")
    with h5py.File(weights_path, 'r') as f:
        for layer_name in ['batch_normalization_1', 'dense_2', 'dense_3']:
            layer = model.get_layer(layer_name)
            g = f[layer_name][layer_name]
            weights = [np.array(g[w.name.split('/')[-1]]) for w in layer.weights]
            layer.set_weights(weights)
        
        g_base = f['efficientnetb0']
        for sublayer in base_model.layers:
            h5_name = sublayer.name
            if h5_name not in g_base and (h5_name + '_1') in g_base:
                h5_name = h5_name + '_1'
            if h5_name in g_base:
                sg = g_base[h5_name]
                weights = [np.array(sg[w.name.split('/')[-1]]) for w in sublayer.weights]
                if weights:
                    sublayer.set_weights(weights)
    return model

backbones = [
    {
        "name": "CNN",
        "loader": lambda: tf.keras.models.load_model(ROOT / "best_cnn_model.keras"),
        "preprocess": lambda x: x / 255.0,
    },
    {
        "name": "MobileNetV2",
        "loader": lambda: tf.keras.models.load_model(ROOT / "best_mobilenet_model.keras"),
        "preprocess": mob_preprocess,
    },
    {
        "name": "EfficientNetB0",
        "loader": lambda: load_efficientnet_model(ROOT / "best_efficientnet_model.h5"),
        "preprocess": eff_preprocess,
    },
]

classifiers = {
    "SVM": lambda: SVC(kernel="rbf", probability=False, random_state=SEED),
    "KNN": lambda: KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1),
    "RF": lambda: RandomForestClassifier(n_estimators=200, max_depth=15, random_state=SEED, n_jobs=-1),
    "NB": lambda: GaussianNB(),
}

all_results = []

for bb in backbones:
    print("\n" + "=" * 80)
    print(f"BACKBONE: {bb['name']}")
    print("=" * 80)

    cache_file = CACHE_DIR / f"{bb['name']}_features.npz"
    if cache_file.exists():
        print(f"  Loading cached penultimate features from {cache_file} ...")
        cached = np.load(cache_file)
        train_feats, train_labels = cached['train_feats'], cached['train_labels']
        test_feats, test_labels = cached['test_feats'], cached['test_labels']
        print(f"  Loaded train: {train_feats.shape}, test: {test_feats.shape}")
    else:
        model = bb["loader"]()
        feature_extractor = tf.keras.Model(model.input, model.layers[-2].output)
        print(f"Feature vector size: {feature_extractor.output_shape[-1]}")

        def extract(df, split_name):
            datagen = ImageDataGenerator(preprocessing_function=bb["preprocess"])
            gen = datagen.flow_from_dataframe(
                df, x_col='filepath', y_col='label', classes=class_names,
                target_size=(128, 128), batch_size=32, class_mode='categorical', shuffle=False
            )
            print(f"  Extracting {split_name} features ({len(df)} images)...")
            gen.reset()
            feats = feature_extractor.predict(gen, verbose=1)
            return feats, np.array(gen.classes)

        train_feats, train_labels = extract(train_df, "train")
        test_feats, test_labels = extract(test_df, "test")

        print(f"  Caching extracted features to {cache_file} ...")
        np.savez_compressed(
            cache_file,
            train_feats=train_feats,
            train_labels=train_labels,
            test_feats=test_feats,
            test_labels=test_labels
        )

    # Standardize features -- important for SVM/KNN, harmless for RF/NB
    scaler = StandardScaler()
    train_feats_scaled = scaler.fit_transform(train_feats)
    test_feats_scaled = scaler.transform(test_feats)

    for clf_name, clf_factory in classifiers.items():
        print(f"\n  Training {bb['name']} + {clf_name} ...")
        clf = clf_factory()
        t0 = time.time()
        clf.fit(train_feats_scaled, train_labels)
        train_time = time.time() - t0

        t_infer_0 = time.perf_counter()
        test_pred = clf.predict(test_feats_scaled)
        t_infer_1 = time.perf_counter()
        infer_latency_ms = ((t_infer_1 - t_infer_0) / len(test_feats_scaled)) * 1000.0

        train_pred = clf.predict(train_feats_scaled)

        train_acc = accuracy_score(train_labels, train_pred) * 100.0
        test_acc = accuracy_score(test_labels, test_pred) * 100.0
        test_macro_f1 = f1_score(test_labels, test_pred, average='macro', zero_division=0) * 100.0
        test_weighted_f1 = f1_score(test_labels, test_pred, average='weighted', zero_division=0) * 100.0
        test_macro_prec = precision_score(test_labels, test_pred, average='macro', zero_division=0) * 100.0
        test_macro_rec = recall_score(test_labels, test_pred, average='macro', zero_division=0) * 100.0

        print(f"    Train acc: {train_acc:.2f}%  Test acc: {test_acc:.2f}%  "
              f"Macro F1: {test_macro_f1:.2f}%  (Train time: {train_time:.1f}s, Inference: {infer_latency_ms:.2f}ms/img)")

        all_results.append({
            "Backbone": bb["name"],
            "Classifier": clf_name,
            "Model": f"{bb['name']} + {clf_name}",
            "Train Acc (%)": round(train_acc, 2),
            "Test Acc (%)": round(test_acc, 2),
            "Macro Precision (%)": round(test_macro_prec, 2),
            "Macro Recall (%)": round(test_macro_rec, 2),
            "Macro F1 (%)": round(test_macro_f1, 2),
            "Weighted F1 (%)": round(test_weighted_f1, 2),
            "Train Time (s)": round(train_time, 2),
            "Inference Latency (ms/img)": round(infer_latency_ms, 3),
        })

# ----------------------------------------------------------------------
# 3. Save results -- 12 rows, one per (backbone, classifier) pair
# ----------------------------------------------------------------------
results_df = pd.DataFrame(all_results)
out_csv = ROOT / "twelve_model_comparison.csv"
results_df.to_csv(out_csv, index=False)
results_df.to_csv(OUTPUT_DIR / "twelve_model_comparison.csv", index=False)

md_table = results_df[[
    "Model", "Backbone", "Classifier", "Train Acc (%)", "Test Acc (%)",
    "Macro Precision (%)", "Macro Recall (%)", "Macro F1 (%)", "Weighted F1 (%)", "Train Time (s)"
]].to_markdown(index=False)
(OUTPUT_DIR / "twelve_model_table.md").write_text(md_table, encoding="utf-8")

tex_table = results_df[[
    "Model", "Train Acc (%)", "Test Acc (%)", "Macro Precision (%)",
    "Macro Recall (%)", "Macro F1 (%)", "Weighted F1 (%)", "Train Time (s)"
]].to_latex(index=False, float_format="%.2f")
(OUTPUT_DIR / "twelve_model_table.tex").write_text(tex_table, encoding="utf-8")

print("\n" + "=" * 80)
print("FINAL 12-MODEL COMPARISON BENCHMARK (FOR REPORT & THESIS)")
print("=" * 80)
print(results_df.to_string(index=False))
print(f"\nSaved results to:")
print(f"  - {out_csv}")
print(f"  - {OUTPUT_DIR / 'twelve_model_comparison.csv'}")
print(f"  - {OUTPUT_DIR / 'twelve_model_table.md'}")
print(f"  - {OUTPUT_DIR / 'twelve_model_table.tex'}")

