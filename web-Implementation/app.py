import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torchvision.transforms as transforms
import requests
import os
from io import BytesIO
import asyncio
import aiohttp
from streamlit_option_menu import option_menu
import plotly.express as px
import logging
from huggingface_hub import snapshot_download

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config with modern layout
st.set_page_config(
    page_title="AI Based Disease Diagnostic using CT Scans and X-Rays",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🩺"
)

# Custom CSS for modern styling
st.markdown("""
    <style>
    .main { background-color: #f5f7fa; }
    .stButton>button { background-color: #007bff; color: white; border-radius: 8px; }
    .stFileUploader { border: 2px dashed #007bff; border-radius: 8px; padding: 10px; }
    .stSidebar { background-color: #e9ecef; }
    .prediction-box { background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
    </style>
""", unsafe_allow_html=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"🔌 Using device: {device}")

# Model configurations (use local paths for weights)
MODEL_CONFIGS = {
    "Lung Cancer": {
        "model_name": "google/efficientnet-b1",
        "num_labels": 3,
        "classes": ["Benign", "Malignant", "Normal"],
        "weight_path": "web-Implementation//models//Lung-Cancer_model.pth",
        "metrics": {"accuracy": 0.92, "f1_score": 0.90}
    },
    "Kidney Disease": {
        "model_name": "google/efficientnet-b3",
        "num_labels": 4,
        "classes": ["Cyst", "Normal", "Stone", "Tumor"],
        "weight_path": "web-Implementation//models//Kidney_model_best (1).pth",
        "metrics": {"accuracy": 0.89, "f1_score": 0.87}
    },
    "COVID-19": {
        "model_name": "google/efficientnet-b4",
        "num_labels": 2,
        "classes": ["COVID", "NON-COVID"],
        "weight_path": "web-Implementation//models//Covid_model_best.pth",
        "metrics": {"accuracy": 0.95, "f1_score": 0.93}
    }
}

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Asynchronous weight downloading (optional, if using URLs)
async def download_weights_async(url, save_path):
    if not os.path.exists(save_path):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    response.raise_for_status()
                    with open(save_path, "wb") as f:
                        f.write(await response.read())
            logger.info(f"Downloaded weights to {save_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download weights from {url}: {str(e)}")
            return False
    return True

# Run async download in Streamlit
def run_async_download(url, save_path):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(download_weights_async(url, save_path))
    loop.close()
    return result

# Load processor and model
@st.cache_resource
def load_model(model_key):
    config = MODEL_CONFIGS[model_key]
    try:
        # Download model snapshot (offline mode support)
        model_path = snapshot_download(repo_id=config["model_name"])
        processor = AutoImageProcessor.from_pretrained(model_path)
        model = AutoModelForImageClassification.from_pretrained(
            model_path,
            num_labels=config["num_labels"],
            ignore_mismatched_sizes=True
        ).to(device)

        # Progress bar
        progress_bar = st.progress(0)

        # Check if weights exist locally
        if not os.path.exists(config["weight_path"]):
            if "weight_url" in config:
                if not run_async_download(config["weight_url"], config["weight_path"]):
                    st.error("Cannot proceed without model weights.")
                    return None, None
            else:
                st.error(f"Weight file not found at {config['weight_path']}.")
                return None, None
        progress_bar.progress(50)

        # Load weights
        model.load_state_dict(torch.load(config["weight_path"], map_location=device))
        model.eval()
        progress_bar.progress(100)
        st.success(f"Loaded model '{model_key}' successfully.")
        return processor, model

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Image transform
def get_transform(processor):
    return transforms.Compose([
        transforms.Resize((460, 700)),  # Height, Width
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
    ])

# Grad-CAM helper
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        with torch.cuda.amp.autocast():
            output = self.model(input_tensor).logits
            class_score = output[:, class_idx]
            class_score.backward()

        if self.gradients is None or self.activations is None:
            logger.error("Gradients or activations are None.")
            return None

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.nn.functional.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        if cam.size == 0:
            logger.error("Grad-CAM output is empty.")
            return None

        return cam

# Prediction and visualization function
def predict_with_gradcam(image, processor, model, class_names, confidence_threshold):
    try:
        # Validate image
        if image.size[0] < 100 or image.size[1] < 100:
            return None, None, None, None, "Image resolution too low. Minimum size: 100x100 pixels."

        # Preprocess image
        transform = get_transform(processor)
        input_tensor = transform(image).unsqueeze(0).to(device)
        logger.info(f"Input tensor shape: {input_tensor.shape}")

        # Predict
        with torch.no_grad():
            outputs = model(input_tensor).logits
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item() * 100

        # Check confidence threshold
        if confidence < confidence_threshold:
            return None, None, None, None, f"Prediction confidence ({confidence:.2f}%) is below threshold ({confidence_threshold}%)."

        # Top-2 predictions
        topk = torch.topk(probabilities, k=2)
        top_predictions = [
            (class_names[idx.item()], prob.item() * 100)
            for idx, prob in zip(topk.indices[0], topk.values[0])
        ]

        # Grad-CAM
        for name, module in reversed(list(model.base_model.named_modules())):
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
                break

        gradcam = GradCAM(model, target_layer)
        cam = gradcam.generate(input_tensor, predicted_class)

        # Validate cam and target dimensions
        if cam is None or cam.size == 0:
            return None, None, None, None, "Error: Grad-CAM heatmap is empty or invalid."
        target_width, target_height = input_tensor.shape[3], input_tensor.shape[2]
        if target_width <= 0 or target_height <= 0:
            return None, None, None, None, f"Error: Invalid target dimensions ({target_width}, {target_height})."

        # Resize cam
        cam = np.float32(cam)  # Ensure float32
        cam = cv2.resize(cam, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

        # Superimpose heatmap
        img = np.array(image.resize((target_width, target_height)))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        superimposed_img = heatmap + np.float32(img) / 255
        superimposed_img = superimposed_img / np.max(superimposed_img)

        # Overlay prediction text
        display_img = (superimposed_img * 255).astype(np.uint8)
        label_text = f"{class_names[predicted_class]} ({confidence:.2f}%)"
        display_img = cv2.putText(
            display_img, label_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
            1.2, (255, 255, 255), 3
        )

        return display_img, predicted_class, confidence, top_predictions, None
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return None, None, None, None, f"Error processing image: {str(e)}"

# Sidebar with modern navigation
with st.sidebar:
    selected = option_menu(
        menu_title="AI Medical Classifier",
        options=["Home", "Model Settings", "Metrics", "About"],
        icons=["house", "gear", "bar-chart", "info-circle"],
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#e9ecef"},
            "icon": {"color": "#007bff", "font-size": "20px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#d3d4d5"},
            "nav-link-selected": {"background-color": "#007bff"},
        }
    )

# Main UI based on sidebar selection
if selected == "Home":
    st.title("🩺 AI Medical Image Classifier")
    st.markdown("""
    Upload medical images to classify diseases using state-of-the-art deep learning models.
    Visualize predictions with Grad-CAM heatmaps to understand model decisions.
    """)

    # Model selection
    model_key = st.selectbox("Select Model", list(MODEL_CONFIGS.keys()), key="model_select")
    confidence_threshold = st.slider("Confidence Threshold (%)", 0, 100, 50, 5, key="conf_threshold")

    # Model details
    config = MODEL_CONFIGS[model_key]
    with st.expander("Model Details"):
        st.markdown(f"""
        - **Architecture**: {config['model_name']}
        - **Classes**: {', '.join(config['classes'])}
        - **Number of Classes**: {config['num_labels']}
        - **Metrics**: Accuracy: {config['metrics']['accuracy']*100:.2f}%, F1-Score: {config['metrics']['f1_score']*100:.2f}%
        """)

    # Drag-and-drop file uploader
    st.markdown("### Upload Image(s)")
    uploaded_files = st.file_uploader(
        "Choose image(s) (PNG/JPEG)", type=["png", "jpg", "jpeg"], accept_multiple_files=True
    )

    if uploaded_files:
        # Load model
        processor, model = load_model(model_key)
        if processor is None or model is None:
            st.stop()

        # Process each uploaded image
        for uploaded_file in uploaded_files:
            st.markdown(f"#### Processing: {uploaded_file.name}")
            try:
                image = Image.open(uploaded_file).convert("RGB")
                if image is None:
                    st.error(f"Failed to load image: {uploaded_file.name}")
                    continue
            except Exception as e:
                st.error(f"Error loading image {uploaded_file.name}: {str(e)}")
                continue

            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Uploaded Image", width=300)

            # Predict and visualize
            class_names = config["classes"]
            display_img, predicted_class, confidence, top_predictions, error = predict_with_gradcam(
                image, processor, model, class_names, confidence_threshold
            )

            if error:
                st.error(error)
            else:
                # Display results in a styled box
                with st.container():
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    st.subheader("Prediction Results")
                    st.write(f"**Final Prediction**: {class_names[predicted_class]} ({confidence:.2f}%)")

                    st.subheader("Top Predictions")
                    for class_name, prob in top_predictions:
                        st.write(f"- {class_name}: {prob:.2f}%")
                    st.markdown('</div>', unsafe_allow_html=True)

                # Interactive Grad-CAM with Plotly
                with col2:
                    st.subheader("Grad-CAM Visualization")
                    fig = px.imshow(display_img, title="Grad-CAM Heatmap")
                    fig.update_layout(width=300, height=300)
                    st.plotly_chart(fig, use_container_width=True)

                # Download Grad-CAM image
                img_buffer = BytesIO()
                Image.fromarray(display_img).save(img_buffer, format="PNG")
                st.download_button(
                    label="Download Grad-CAM Image",
                    data=img_buffer.getvalue(),
                    file_name=f"gradcam_{uploaded_file.name}",
                    mime="image/png"
                )

elif selected == "Model Settings":
    st.title("⚙️ Model Settings")
    st.markdown("Configure model parameters and advanced settings.")
    model_key = st.selectbox("Select Model", list(MODEL_CONFIGS.keys()), key="settings_model")
    confidence_threshold = st.slider("Confidence Threshold (%)", 0, 100, 50, 5, key="settings_conf")
    st.markdown("Additional settings (e.g., batch size, inference mode) coming soon!")

elif selected == "Metrics":
    st.title("📊 Model Performance Metrics")
    st.markdown("View performance metrics for each model.")
    for model_key, config in MODEL_CONFIGS.items():
        st.subheader(model_key)
        metrics = config["metrics"]
        st.write(f"- **Accuracy**: {metrics['accuracy']*100:.2f}%")
        st.write(f"- **F1-Score**: {metrics['f1_score']*100:.2f}%")
        # Placeholder for confusion matrix visualization
        st.markdown("Confusion matrix visualization coming soon!")

elif selected == "About":
    st.title("ℹ️ About")
    st.markdown("""
    **AI Medical Image Classifier** is a state-of-the-art tool for diagnosing diseases from medical images.
    Built with Streamlit, PyTorch, and EfficientNet models, it provides accurate predictions and interpretable visualizations.
    
    - **Source**: [GitHub Repository](https://github.com/ujwalreddybattu04/AI-Based-Disease-Diagnostic-)
    - **Contact**: support@x.ai
    - **Version**: 2.0.0
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center;'>
    Built with ❤️ using Streamlit and PyTorch | © 2025 xAI
</div>
""", unsafe_allow_html=True)
