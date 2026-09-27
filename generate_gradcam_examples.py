"""Create correct and incorrect MobileNet Grad-CAM test examples."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

SEED = 42
ROOT = Path(__file__).parent
DATASET = ROOT / "PlantVillage"
OUTPUT = ROOT / "artifacts" / "gradcam"
OUTPUT.mkdir(parents=True, exist_ok=True)


def heatmap(model, image_batch, last_conv_layer_name="out_relu"):
    layer_names = [l.name for l in model.layers]
    if last_conv_layer_name in layer_names:
        conv_layer = model.get_layer(last_conv_layer_name)
        conv_idx = model.layers.index(conv_layer)
        feature_extractor = tf.keras.Model(model.input, conv_layer.output)
        remaining_layers = model.layers[conv_idx + 1:]
    else:
        sub_model = None
        for l in model.layers:
            if hasattr(l, "layers") and last_conv_layer_name in [sub_l.name for sub_l in l.layers]:
                sub_model = l
                break
        if sub_model is None:
            raise ValueError(f"Layer '{last_conv_layer_name}' not found in model or sub-models!")
        conv_layer = sub_model.get_layer(last_conv_layer_name)
        feature_extractor = tf.keras.Model(sub_model.input, conv_layer.output)
        sub_idx = model.layers.index(sub_model)
        remaining_layers = model.layers[sub_idx + 1:]

    with tf.GradientTape() as tape:
        conv_output = feature_extractor(image_batch)
        tape.watch(conv_output)
        
        # Forward pass through remaining layers up to pre-softmax logits
        x = conv_output
        for layer in remaining_layers[:-1]:
            x = layer(x)
            
        last_layer = remaining_layers[-1]
        if hasattr(last_layer, "kernel") and hasattr(last_layer, "bias"):
            logits = tf.matmul(x, last_layer.kernel) + last_layer.bias
        else:
            logits = last_layer(x)
            
        predicted_class = tf.argmax(logits[0])
        score = logits[:, predicted_class]

    gradients = tape.gradient(score, conv_output)
    weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
    result = tf.squeeze(conv_output[0] @ weights[..., tf.newaxis])
    return tf.maximum(result, 0) / (tf.reduce_max(result) + tf.keras.backend.epsilon())

records = [
    {"filepath": str(file), "label": file.parent.name}
    for file in sorted(DATASET.glob("*/*"))
    if file.suffix.lower() in {".jpg", ".jpeg", ".png"}
]
image_df = pd.DataFrame(records)
classes = sorted(image_df.label.unique())
_, temporary = train_test_split(
    image_df, test_size=0.30, stratify=image_df.label, random_state=SEED
)
_, test_df = train_test_split(
    temporary, test_size=0.50, stratify=temporary.label, random_state=SEED
)

model = tf.keras.models.load_model(ROOT / "mobilenet_model.h5")
preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
paths = test_df.filepath.tolist()
true_ids = np.array([classes.index(label) for label in test_df.label])

predictions = []
for start in range(0, len(paths), 32):
    batch = [
        preprocess(tf.keras.utils.img_to_array(tf.keras.utils.load_img(path, target_size=(128, 128))))
        for path in paths[start : start + 32]
    ]
    predictions.append(model.predict(np.asarray(batch), verbose=0))
probabilities = np.vstack(predictions)
predicted_ids = probabilities.argmax(axis=1)

correct = np.flatnonzero(predicted_ids == true_ids)[:2]
incorrect = np.flatnonzero(predicted_ids != true_ids)[:2]
selected = list(correct) + list(incorrect)

summary = []
for number, index in enumerate(selected, start=1):
    original = tf.keras.utils.load_img(paths[index], target_size=(128, 128))
    image = tf.keras.utils.img_to_array(original)
    input_batch = np.expand_dims(preprocess(image.copy()), axis=0)
    cam = heatmap(model, input_batch)
    cam = tf.image.resize(cam[..., tf.newaxis], (128, 128)).numpy().squeeze()
    status = "correct" if predicted_ids[index] == true_ids[index] else "incorrect"
    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].imshow(original)
    axes[0].set_title(f"True: {classes[true_ids[index]]}\nPredicted: {classes[predicted_ids[index]]}")
    axes[0].axis("off")
    axes[1].imshow(original)
    axes[1].imshow(cam, cmap="jet", alpha=0.45)
    axes[1].set_title(f"Grad-CAM ({status})")
    axes[1].axis("off")
    figure.tight_layout()
    filename = OUTPUT / f"{number}_{status}.png"
    figure.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(figure)
    summary.append({
        "file": filename.name,
        "status": status,
        "true_label": classes[true_ids[index]],
        "predicted_label": classes[predicted_ids[index]],
        "confidence": probabilities[index, predicted_ids[index]],
    })

pd.DataFrame(summary).to_csv(OUTPUT / "summary.csv", index=False)
print(f"Saved {len(summary)} Grad-CAM examples to {OUTPUT}")
