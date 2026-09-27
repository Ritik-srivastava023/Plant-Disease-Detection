"""
Extract Full 15-Class Probability Vectors and Authentically Compute Metrics.
Zero Hardcoding. Evaluates on the untouched 3,096-image test split (Seed 42).
"""
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input as eff_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mob_preprocess
from tensorflow.keras.preprocessing.image import ImageDataGenerator

ROOT = Path(__file__).parent
DATASET_PATH = ROOT / "PlantVillage"
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
SEED = 42

print("=" * 80)
print("1. LOADING DATASET & STRATIFIED SPLIT (Seed 42)")
print("=" * 80)

records = [
    {'filepath': str(file), 'label': file.parent.name}
    for file in sorted(DATASET_PATH.glob('*/*'))
    if file.suffix.lower() in IMAGE_EXTENSIONS
]
image_df = pd.DataFrame(records)
class_names = sorted(image_df['label'].unique())
n_classes = len(class_names)
print(f"Total images found: {len(image_df)} across {n_classes} classes.")

train_df, temporary_df = train_test_split(
    image_df, test_size=0.30, stratify=image_df['label'], random_state=SEED
)
val_df, test_df = train_test_split(
    temporary_df, test_size=0.50, stratify=temporary_df['label'], random_state=SEED
)
print(f"Split sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# ----------------------------------------------------------------------
# 2. Helper to Build & Load EfficientNet-B0 with Exact Checkpoint Weights
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
        # Head layers
        for layer_name in ['batch_normalization_1', 'dense_2', 'dense_3']:
            layer = model.get_layer(layer_name)
            g = f[layer_name][layer_name]
            weights = [np.array(g[w.name.split('/')[-1]]) for w in layer.weights]
            layer.set_weights(weights)
        
        # Base model layers
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

# ----------------------------------------------------------------------
# 3. Model Definitions to Evaluate
# ----------------------------------------------------------------------
models_to_evaluate = [
    {
        "name": "Custom CNN (GAP + BN)",
        "csv_file": "cnn_test_predictions.csv",
        "preprocess": lambda x: x / 255.0,
        "loader": lambda: tf.keras.models.load_model(ROOT / "best_cnn_model.keras"),
    },
    {
        "name": "MobileNetV2 (Two-Stage)",
        "csv_file": "mobilenet_test_predictions.csv",
        "preprocess": mob_preprocess,
        "loader": lambda: tf.keras.models.load_model(ROOT / "best_mobilenet_model.keras"),
    },
    {
        "name": "EfficientNet-B0 (Two-Stage)",
        "csv_file": "efficientnet_test_predictions.csv",
        "preprocess": eff_preprocess,
        "loader": lambda: load_efficientnet_model(ROOT / "best_efficientnet_model.h5"),
    },
]

# True labels binarized for One-vs-Rest AUC
y_true_indices = np.array([class_names.index(label) for label in test_df['label']])
y_true_onehot = label_binarize(y_true_indices, classes=np.arange(n_classes))

all_metrics = []

for m_info in models_to_evaluate:
    name = m_info["name"]
    csv_file = m_info["csv_file"]
    preprocess_fn = m_info["preprocess"]
    loader_fn = m_info["loader"]

    print("\n" + "=" * 80)
    print(f"EVALUATING MODEL: {name}")
    print("=" * 80)

    model = loader_fn()
    total_params = model.count_params()
    print(f"Loaded {name}. Total Parameters: {total_params:,}")

    # Build evaluation generator
    datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)
    test_gen = datagen.flow_from_dataframe(
        test_df,
        x_col='filepath',
        y_col='label',
        classes=class_names,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )

    print(f"Running inference over {len(test_df)} test samples...")
    test_gen.reset()
    probabilities = model.predict(test_gen, verbose=1)

    # Sanity checks
    assert probabilities.shape == (3096, 15), f"Unexpected shape {probabilities.shape}"
    row_sums = probabilities.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-4), "Probabilities do not sum to 1.0!"

    y_pred_indices = np.argmax(probabilities, axis=1)
    y_true_from_gen = test_gen.classes
    assert np.array_equal(y_true_indices, y_true_from_gen), "Label index alignment mismatch!"

    # Compute authentic metrics
    acc = accuracy_score(y_true_indices, y_pred_indices) * 100.0
    macro_prec = precision_score(y_true_indices, y_pred_indices, average='macro', zero_division=0) * 100.0
    macro_rec = recall_score(y_true_indices, y_pred_indices, average='macro', zero_division=0) * 100.0
    macro_f1 = f1_score(y_true_indices, y_pred_indices, average='macro', zero_division=0) * 100.0
    weighted_f1 = f1_score(y_true_indices, y_pred_indices, average='weighted', zero_division=0) * 100.0

    # AUTHENTIC COMPUTED AUC
    macro_auc = roc_auc_score(y_true_onehot, probabilities, average='macro', multi_class='ovr')
    weighted_auc = roc_auc_score(y_true_onehot, probabilities, average='weighted', multi_class='ovr')

    print("\n--- AUTHENTIC COMPUTED METRICS ---")
    print(f"Test Accuracy:          {acc:.2f}% ({np.sum(y_true_indices == y_pred_indices)} / {len(y_true_indices)})")
    print(f"Macro Precision:        {macro_prec:.2f}%")
    print(f"Macro Recall:           {macro_rec:.2f}%")
    print(f"Macro F1-Score:         {macro_f1:.2f}%")
    print(f"Weighted F1-Score:      {weighted_f1:.2f}%")
    print(f"Macro OvR ROC-AUC:      {macro_auc:.4f}")
    print(f"Weighted OvR ROC-AUC:   {weighted_auc:.4f}")

    # Build comprehensive dataframe with all 15 probability columns
    pred_data = {
        'filepath': test_gen.filepaths,
        'true_label': [class_names[i] for i in y_true_indices],
        'predicted_label': [class_names[i] for i in y_pred_indices],
        'confidence': probabilities.max(axis=1),
    }
    for c_idx, c_name in enumerate(class_names):
        pred_data[f"prob_{c_name}"] = probabilities[:, c_idx]

    df_out = pd.DataFrame(pred_data)
    out_path = ROOT / csv_file
    df_out.to_csv(out_path, index=False)
    print(f"SUCCESS: Saved {len(df_out)} predictions with all 15 probability columns to {out_path}")

    all_metrics.append({
        "Model": name,
        "Total Params": total_params,
        "Test Accuracy (%)": round(acc, 2),
        "Macro Precision (%)": round(macro_prec, 2),
        "Macro Recall (%)": round(macro_rec, 2),
        "Macro F1 (%)": round(macro_f1, 2),
        "Weighted F1 (%)": round(weighted_f1, 2),
        "Macro OvR AUC": round(macro_auc, 4),
        "Weighted OvR AUC": round(weighted_auc, 4),
    })

print("\n" + "=" * 80)
print("FINAL CONSOLIDATED RESULTS FROM LIVE EXECUTION")
print("=" * 80)
summary_df = pd.DataFrame(all_metrics)
print(summary_df.to_string(index=False))
print("=" * 80)

