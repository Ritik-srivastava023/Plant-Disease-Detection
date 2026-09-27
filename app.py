"""
PhytoAI: Deep Learning & SOTA Deep Hybrid Plant Disease Diagnostic Platform.
Built with Streamlit, TensorFlow, Scikit-Learn, and Grad-CAM Explainability.
"""
from pathlib import Path
import time
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mob_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess

from disease_info import DISEASE_KNOWLEDGE_BASE

ROOT = Path(__file__).parent
ARTIFACTS_DIR = ROOT / "artifacts"
PLANTVILLAGE_DIR = ROOT / "PlantVillage"

CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

# -----------------------------------------------------------------------------
# Streamlit Page Setup & Custom Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="PhytoAI | Plant Disease Diagnosis",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1b5e20;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4a5568;
        margin-bottom: 1.5rem;
    }
    .status-badge-healthy {
        background-color: #d1fae5;
        color: #065f46;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
    }
    .status-badge-diseased {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem;
        margin-top: 0.5rem;
    }
    .advisory-box {
        background: #f0fdf4;
        border-left: 4px solid #16a34a;
        padding: 1rem;
        border-radius: 6px;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Cached Model & Asset Loaders
# -----------------------------------------------------------------------------
@st.cache_resource
def load_mobilenet_models():
    """Load MobileNetV2 base model, feature extractor, and pre-trained SVM."""
    base_model = tf.keras.models.load_model(ROOT / "best_mobilenet_model.keras")
    feature_extractor = tf.keras.Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
    
    svm_path = ARTIFACTS_DIR / "mobilenet_svm.joblib"
    svm_payload = joblib.load(svm_path) if svm_path.exists() else None
    return base_model, feature_extractor, svm_payload


@st.cache_resource
def load_cnn_models():
    """Load Custom CNN baseline model, feature extractor, and pre-trained SVM."""
    base_model = tf.keras.models.load_model(ROOT / "best_cnn_model.keras")
    feature_extractor = tf.keras.Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
    
    svm_path = ARTIFACTS_DIR / "cnn_svm.joblib"
    svm_payload = joblib.load(svm_path) if svm_path.exists() else None
    return base_model, feature_extractor, svm_payload


@st.cache_resource
def load_efficientnet_pipeline():
    """Load EfficientNet-B0 model checkpoint."""
    import h5py
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications.efficientnet import EfficientNetB0
    
    base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
    inputs = tf.keras.Input(shape=(128, 128, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    bn = layers.BatchNormalization(name="batch_normalization_1")(x)
    d1 = layers.Dense(128, activation="relu", name="dense_2")(bn)
    dp = layers.Dropout(0.3)(d1)
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax", name="dense_3")(dp)
    model = models.Model(inputs=inputs, outputs=outputs, name="EfficientNetB0_Classifier")

    weights_path = ROOT / "best_efficientnet_model.h5"
    if weights_path.exists():
        with h5py.File(weights_path, "r") as f:
            for layer_name in ["batch_normalization_1", "dense_2", "dense_3"]:
                layer = model.get_layer(layer_name)
                g = f[layer_name][layer_name]
                weights = [np.array(g[w.name.split("/")[-1]]) for w in layer.weights]
                layer.set_weights(weights)
            g_base = f["efficientnetb0"]
            for sublayer in base_model.layers:
                h5_name = sublayer.name
                if h5_name not in g_base and (h5_name + "_1") in g_base:
                    h5_name = h5_name + "_1"
                if h5_name in g_base:
                    sg = g_base[h5_name]
                    weights = [np.array(sg[w.name.split("/")[-1]]) for w in sublayer.weights]
                    if weights:
                        sublayer.set_weights(weights)
    return model


# -----------------------------------------------------------------------------
# Grad-CAM Explainability Algorithm (Pre-Softmax Logits)
# -----------------------------------------------------------------------------
def generate_gradcam_heatmap(image_batch, target_model, last_conv_layer_name="out_relu"):
    """Compute normalized Grad-CAM activation map on the deepest convolutional layer."""
    layer_names = [l.name for l in target_model.layers]
    if last_conv_layer_name in layer_names:
        conv_layer = target_model.get_layer(last_conv_layer_name)
        conv_idx = target_model.layers.index(conv_layer)
        feature_extractor = tf.keras.Model(target_model.input, conv_layer.output)
        remaining_layers = target_model.layers[conv_idx + 1:]
    else:
        sub_model = None
        for l in target_model.layers:
            if hasattr(l, "layers") and last_conv_layer_name in [sub_l.name for sub_l in l.layers]:
                sub_model = l
                break
        if sub_model is None:
            # Fallback to last 4D conv layer
            for l in reversed(target_model.layers):
                if len(l.output_shape) == 4:
                    conv_layer = l
                    break
            feature_extractor = tf.keras.Model(target_model.input, conv_layer.output)
            conv_idx = target_model.layers.index(conv_layer)
            remaining_layers = target_model.layers[conv_idx + 1:]
        else:
            conv_layer = sub_model.get_layer(last_conv_layer_name)
            feature_extractor = tf.keras.Model(sub_model.input, conv_layer.output)
            sub_idx = target_model.layers.index(sub_model)
            remaining_layers = target_model.layers[sub_idx + 1:]

    with tf.GradientTape() as tape:
        conv_output = feature_extractor(image_batch)
        tape.watch(conv_output)
        
        x = conv_output
        for layer in remaining_layers[:-1]:
            x = layer(x)
            
        last_layer = remaining_layers[-1]
        if hasattr(last_layer, "kernel") and hasattr(last_layer, "bias"):
            logits = tf.matmul(x, last_layer.kernel) + last_layer.bias
        else:
            logits = last_layer(x)
            
        pred_idx = tf.argmax(logits[0])
        score = logits[:, pred_idx]

    grads = tape.gradient(score, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.squeeze(conv_output[0] @ pooled_grads[..., tf.newaxis])
    cam = tf.maximum(cam, 0) / (tf.reduce_max(cam) + tf.keras.backend.epsilon())
    return cam.numpy()


def superimpose_heatmap(original_pil_img, heatmap, alpha=0.45):
    """Blends original image with jet colormap heatmap overlay."""
    img_array = np.array(original_pil_img.convert("RGB").resize((128, 128)))
    heatmap_uint8 = np.uint8(255 * heatmap)
    
    colormap = matplotlib.colormaps["jet"]
    colors = colormap(np.arange(256))[:, :3]
    colored_heatmap = colors[heatmap_uint8]
    colored_heatmap = np.uint8(255 * colored_heatmap)
    
    overlay = Image.fromarray(colored_heatmap).resize(original_pil_img.size, resample=Image.BICUBIC)
    overlay_array = np.array(overlay)
    
    orig_array = np.array(original_pil_img.convert("RGB"))
    blended = np.uint8(orig_array * (1.0 - alpha) + overlay_array * alpha)
    return Image.fromarray(blended)


# -----------------------------------------------------------------------------
# Curated Sample Leaves for Fast Demonstration
# -----------------------------------------------------------------------------
CURATED_SAMPLES = {
    "Select a pre-loaded sample leaf...": None,
    "Tomato - Early Blight (Concentric Rings)": "Tomato_Early_blight",
    "Tomato - Late Blight (Foliar Necrosis)": "Tomato_Late_blight",
    "Tomato - Yellow Leaf Curl Virus (TYLCV)": "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato - Healthy Leaf": "Tomato_healthy",
    "Potato - Early Blight (Target Spots)": "Potato___Early_blight",
    "Potato - Late Blight": "Potato___Late_blight",
    "Potato - Healthy Leaf": "Potato___healthy",
    "Pepper Bell - Bacterial Spot": "Pepper__bell___Bacterial_spot",
    "Pepper Bell - Healthy Leaf": "Pepper__bell___healthy"
}


def get_sample_image_path(class_dir_name):
    """Finds the first valid sample image from the dataset folder."""
    target_dir = PLANTVILLAGE_DIR / class_dir_name
    if target_dir.exists():
        for ext in [".JPG", ".jpg", ".jpeg", ".png", ".PNG"]:
            matches = list(target_dir.glob(f"*{ext}"))
            if matches:
                return matches[0]
    return None


# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/leaf.png", width=64)
    st.title("PhytoAI Engine")
    st.markdown("**Intelligent Agricultural Vision**")
    st.write("---")

    model_option = st.selectbox(
        "🧠 Select Diagnostic Model",
        [
            "MobileNetV2 + RBF SVM (Deep Hybrid SOTA - 97.00% Acc)",
            "MobileNetV2 End-to-End (94.57% Acc + Grad-CAM)",
            "Custom CNN + RBF SVM (Scratch SOTA - 97.32% Acc)",
            "EfficientNet-B0 (Compound Scaling - 95.70% Acc)",
            "Custom CNN Baseline (GAP + Softmax - 68.12% Acc)"
        ],
        index=0,
        help="Select between our pre-trained End-to-End Deep Learning baselines and SOTA Deep Hybrid Penultimate Margin Classifiers."
    )

    st.write("---")
    st.subheader("⚡ Model Performance Matrix")
    if "MobileNetV2 + RBF SVM" in model_option:
        st.info("⭐ **Top Pick**: 97.00% Test Accuracy | 96.27% Macro F1 | 0.27 ms latency. Combines inverted residual visual features with non-linear RBF margin separation.")
    elif "MobileNetV2 End-to-End" in model_option:
        st.info("94.57% Test Accuracy | 0.9987 AUC | 87.78 ms CPU latency. Pre-trained on ImageNet, fine-tuned on PlantVillage with Grad-CAM explainability.")
    elif "Custom CNN + RBF SVM" in model_option:
        st.info("97.32% Test Accuracy | 96.75% Macro F1 | 0.28 ms latency. Scratch CNN features mapped through infinite-dimensional RKHS kernel.")
    elif "EfficientNet-B0" in model_option:
        st.info("95.70% Test Accuracy | 0.9991 AUC. Compound scaled depth/width/resolution architecture.")
    else:
        st.warning("68.12% Test Accuracy | 113k parameters. Demonstrates the linear separability ceiling of scratch CNNs with softmax.")

    st.write("---")
    st.caption("College Final Year Project | IEEE Conference Paper")
    st.caption("Dataset: PlantVillage (20,638 images | 15 Classes)")


# -----------------------------------------------------------------------------
# Main Application Content
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">🌿 PhytoAI: Plant Disease Diagnostic Portal</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Upload a leaf image to diagnose pathology, inspect neural activation heatmaps (Grad-CAM), and receive instant agronomic treatment advisories.'
    '</div>',
    unsafe_allow_html=True
)

col_upload, col_sample = st.columns([2, 2])

with col_upload:
    uploaded_file = st.file_uploader("📂 Upload Leaf Image from Computer / Phone", type=["jpg", "jpeg", "png"])

with col_sample:
    sample_choice = st.selectbox("🧪 Or Quick-Test with Curated Dataset Leaf", list(CURATED_SAMPLES.keys()))

input_image = None
image_source_label = ""

if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
    image_source_label = f"Uploaded File: {uploaded_file.name}"
elif sample_choice and CURATED_SAMPLES[sample_choice] is not None:
    sample_path = get_sample_image_path(CURATED_SAMPLES[sample_choice])
    if sample_path and sample_path.exists():
        input_image = Image.open(sample_path)
        image_source_label = f"Dataset Sample: {sample_choice}"
    else:
        st.warning("Sample image not found on disk.")

if input_image is not None:
    st.write("---")
    col_img, col_diag = st.columns([1.1, 1.4])

    # 1. Preprocess Image
    img_rgb = input_image.convert("RGB").resize((128, 128))
    img_array = np.array(img_rgb, dtype=np.float32)

    predicted_label = None
    confidence = 0.0
    probabilities = None
    cam_heatmap = None
    inference_time = 0.0

    # 2. Run Inference Based on Selected Model
    t_start = time.time()

    if "MobileNetV2 + RBF SVM" in model_option:
        base_model, feat_extractor, svm_payload = load_mobilenet_models()
        mob_input = mob_preprocess(img_array.copy())
        mob_batch = np.expand_dims(mob_input, axis=0)
        
        # Penultimate feature vector (128-dim)
        feats = feat_extractor(mob_batch).numpy()
        feats_std = svm_payload["scaler"].transform(feats)
        probs = svm_payload["svm"].predict_proba(feats_std)[0]
        
        pred_idx = np.argmax(probs)
        predicted_label = CLASS_NAMES[pred_idx]
        confidence = probs[pred_idx]
        probabilities = probs
        inference_time = (time.time() - t_start) * 1000.0

        # Compute Grad-CAM heatmap via underlying MobileNetV2
        try:
            cam_heatmap = generate_gradcam_heatmap(mob_batch, base_model, last_conv_layer_name="out_relu")
        except Exception:
            cam_heatmap = None

    elif "MobileNetV2 End-to-End" in model_option:
        base_model, _, _ = load_mobilenet_models()
        mob_input = mob_preprocess(img_array.copy())
        mob_batch = np.expand_dims(mob_input, axis=0)
        
        probs = base_model.predict(mob_batch, verbose=0)[0]
        pred_idx = np.argmax(probs)
        predicted_label = CLASS_NAMES[pred_idx]
        confidence = probs[pred_idx]
        probabilities = probs
        inference_time = (time.time() - t_start) * 1000.0
        
        try:
            cam_heatmap = generate_gradcam_heatmap(mob_batch, base_model, last_conv_layer_name="out_relu")
        except Exception:
            cam_heatmap = None

    elif "Custom CNN + RBF SVM" in model_option:
        base_model, feat_extractor, svm_payload = load_cnn_models()
        cnn_input = img_array.copy() / 255.0
        cnn_batch = np.expand_dims(cnn_input, axis=0)
        
        feats = feat_extractor(cnn_batch).numpy()
        feats_std = svm_payload["scaler"].transform(feats)
        probs = svm_payload["svm"].predict_proba(feats_std)[0]
        
        pred_idx = np.argmax(probs)
        predicted_label = CLASS_NAMES[pred_idx]
        confidence = probs[pred_idx]
        probabilities = probs
        inference_time = (time.time() - t_start) * 1000.0
        
        try:
            cam_heatmap = generate_gradcam_heatmap(cnn_batch, base_model, last_conv_layer_name="conv2d_2")
        except Exception:
            cam_heatmap = None

    elif "EfficientNet-B0" in model_option:
        eff_model = load_efficientnet_pipeline()
        eff_input = eff_preprocess(img_array.copy())
        eff_batch = np.expand_dims(eff_input, axis=0)
        
        probs = eff_model.predict(eff_batch, verbose=0)[0]
        pred_idx = np.argmax(probs)
        predicted_label = CLASS_NAMES[pred_idx]
        confidence = probs[pred_idx]
        probabilities = probs
        inference_time = (time.time() - t_start) * 1000.0
        cam_heatmap = None

    else:  # Custom CNN Baseline
        base_model, _, _ = load_cnn_models()
        cnn_input = img_array.copy() / 255.0
        cnn_batch = np.expand_dims(cnn_input, axis=0)
        
        probs = base_model.predict(cnn_batch, verbose=0)[0]
        pred_idx = np.argmax(probs)
        predicted_label = CLASS_NAMES[pred_idx]
        confidence = probs[pred_idx]
        probabilities = probs
        inference_time = (time.time() - t_start) * 1000.0
        
        try:
            cam_heatmap = generate_gradcam_heatmap(cnn_batch, base_model, last_conv_layer_name="conv2d_2")
        except Exception:
            cam_heatmap = None

    # Retrieve Agronomic Profile
    info = DISEASE_KNOWLEDGE_BASE.get(predicted_label, {})
    is_healthy = info.get("status", "") == "Healthy"

    # 3. Display Visual Panels (Left Column)
    with col_img:
        st.subheader("📸 Visual Analysis")
        st.caption(image_source_label)
        
        if cam_heatmap is not None:
            tab1, tab2, tab3 = st.tabs(["Superimposed Overlay", "Original Leaf", "Grad-CAM Heatmap"])
            with tab1:
                blended_img = superimpose_heatmap(input_image, cam_heatmap)
                st.image(blended_img, caption="Grad-CAM Lesion Heatmap Overlay (Red = High Attention)", use_column_width=True)
            with tab2:
                st.image(input_image, caption="Input Leaf Image", use_column_width=True)
            with tab3:
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.imshow(cam_heatmap, cmap="jet")
                ax.axis("off")
                st.pyplot(fig)
                plt.close(fig)
        else:
            st.image(input_image, caption="Input Leaf Image", use_column_width=True)
            st.info("Grad-CAM visualization is active for MobileNetV2 and Custom CNN backbones.")

    # 4. Display Diagnostic Results (Right Column)
    with col_diag:
        st.subheader("🔬 Diagnostic Result")
        
        status_badge_html = (
            f'<div class="status-badge-healthy">✅ {info.get("crop", "")}: Healthy Leaf</div>'
            if is_healthy else
            f'<div class="status-badge-diseased">⚠️ {info.get("crop", "")}: {info.get("condition", "")}</div>'
        )
        st.markdown(status_badge_html, unsafe_allow_html=True)
        st.markdown(f"### **{info.get('crop', '')} - {info.get('condition', '')}**")

        # Key Metrics Row
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric("Top-1 Confidence", f"{confidence * 100.0:.2f}%")
        with m_col2:
            st.metric("Inference Latency", f"{inference_time:.1f} ms")
        with m_col3:
            st.metric("Pathogen Type", info.get("pathogen", "N/A").split("(")[0].strip())

        # Top-5 Confidence Bar Chart
        if probabilities is not None:
            st.markdown("#### **Top Candidate Probabilities**")
            top5_indices = np.argsort(probabilities)[::-1][:5]
            top5_labels = [CLASS_NAMES[i].replace("__", " - ").replace("_", " ") for i in top5_indices]
            top5_scores = [probabilities[i] * 100.0 for i in top5_indices]

            chart_df = pd.DataFrame({
                "Disease Candidate": top5_labels[::-1],
                "Confidence (%)": top5_scores[::-1]
            })

            fig, ax = plt.subplots(figsize=(7, 2.8))
            bars = ax.barh(chart_df["Disease Candidate"], chart_df["Confidence (%)"], color="#2b5c8f", edgecolor="black", height=0.6)
            ax.set_xlim(0, 105)
            ax.set_xlabel("Confidence Probability (%)", fontsize=9, fontweight="bold")
            ax.grid(axis="x", linestyle="--", alpha=0.5)
            
            for bar in bars:
                w = bar.get_width()
                ax.annotate(f"{w:.1f}%", xy=(w, bar.get_y() + bar.get_height()/2),
                            xytext=(4, 0), textcoords="offset points", ha="left", va="center", fontsize=8, fontweight="bold")
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # 5. Agronomic Advisory & Treatment Section
    st.write("---")
    st.subheader("🌾 Agronomic Advisory & Action Plan")
    
    with st.expander(f"📋 View Full Pathological Report for {info.get('crop', '')} {info.get('condition', '')}", expanded=True):
        adv_col1, adv_col2 = st.columns(2)
        
        with adv_col1:
            st.markdown("#### **Symptoms & Visual Indicators**")
            st.write(info.get("symptoms", "No symptoms reported."))
            st.markdown("#### **Etiology / Pathogen Classification**")
            st.write(f"🔬 **{info.get('pathogen', 'N/A')}**")
            
        with adv_col2:
            st.markdown("#### **Recommended Chemical & Biological Treatments**")
            for treat in info.get("treatment", []):
                st.markdown(f"- 💊 {treat}")
                
            st.markdown("#### **Preventive Cultural Practices**")
            for prev in info.get("prevention", []):
                st.markdown(f"- 🛡️ {prev}")

# -----------------------------------------------------------------------------
# Collapsible Research Benchmark Section
# -----------------------------------------------------------------------------
st.write("---")
with st.expander("📊 IEEE Research Paper Benchmark & Model Ablation Overview"):
    st.markdown("""
    ### Experimental Comparison Across 15 Architecture Configurations
    Evaluated on the untouched **PlantVillage** test split ($N = 3,096$, Seed 42).
    """)
    twelve_path = ARTIFACTS_DIR / "twelve_model_comparison.csv"
    if twelve_path.exists():
        benchmark_df = pd.read_csv(twelve_path)
        st.dataframe(
            benchmark_df[["Model", "Backbone", "Classifier", "Test Acc (%)", "Macro F1 (%)", "Inference Latency (ms/img)"]],
            use_container_width=True
        )
    st.markdown("""
    > **Key Architectural Insight:**
    > While classical end-to-end models with linear Softmax heads encounter representational bottlenecks on edge hardware,
    > routing deep penultimate features (128-D) through non-linear margin classifiers (RBF SVM) elevates diagnostic accuracy
    > to **97.00% - 97.32%** while slashing inference latency down to $<0.3$ ms.
    """)

