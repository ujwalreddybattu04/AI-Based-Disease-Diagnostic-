## AI - Based-Disease-Diagnostic
The **AI Based Disease Diagnostic using ct-scans and x-rays** is a state-of-the-art web application built with Streamlit and PyTorch to classify medical images for disease diagnosis. It leverages deep learning models (EfficientNet variants) to detect conditions such as lung cancer, kidney diseases, and COVID-19 from medical images (e.g., CT scans, X-rays). The application provides interpretable results using Grad-CAM visualizations to highlight regions of interest in the images.

This project aims to assist medical professionals and researchers by providing accurate, interpretable, and user-friendly tools for medical image analysis.

## Features

- **Multiple Disease Models**:
- Lung Cancer (Benign, Malignant, Normal)
- Kidney Disease (Cyst, Normal, Stone, Tumor)
- COVID-19 (COVID, NON-COVID)
- **Deep Learning Models**: Uses EfficientNet-B1, B3, and B4 architectures from Hugging Face’s Transformers library.
- **Grad-CAM Visualizations**: Highlights influential image regions for predictions using interactive Plotly heatmaps.
- **Batch Processing**: Supports uploading and processing multiple images simultaneously.
- **Modern UI**: Clean, responsive interface with a collapsible sidebar and custom styling.
- **Model Metrics**: Displays accuracy and F1-score for each model.
- **Error Handling**: Robust validation and logging for images, weights, and processing.
- **Downloadable Results**: Save Grad-CAM visualizations as PNG files.

## Models
The application includes three pre-trained models, each tailored to a specific medical condition. All models are based on EfficientNet architectures, fine-tuned for medical image classification.

### 1. Lung Cancer Model
- **Architecture**: EfficientNet-B1
- **Classes**: Benign, Malignant, Normal
- **Number of Classes**: 3
- **Metrics**:
  - Accuracy: 92%
  - F1-Score: 90%
- **Use Case**: Classifies lung CT scans to detect cancerous (malignant), non-cancerous (benign), or healthy (normal) tissue.
- **Weight File**: `Lung-Cancer_model.pth`

### 2. Kidney Disease Model
- **Architecture**: EfficientNet-B3
- **Classes**: Cyst, Normal, Stone, Tumor
- **Number of Classes**: 4
- **Metrics**:
  - Accuracy: 89%
  - F1-Score: 87%
- **Use Case**: Analyzes kidney CT scans to identify cysts, stones, tumors, or normal kidney tissue.
- **Weight File**: `Kidney_model_best (1).pth`

### 3. COVID-19 Model
- **Architecture**: EfficientNet-B4
- **Classes**: COVID, NON-COVID
- **Number of Classes**: 2
- **Metrics**:
  - Accuracy: 95%
  - F1-Score: 93%
- **Use Case**: Detects COVID-19 infection in chest X-rays or CT scans.
- **Weight File**: `Covid_model_best.pth`

Each model uses the Hugging Face `transformers` library for loading pre-trained EfficientNet weights, fine-tuned with custom weights stored in `.pth` files. Grad-CAM is applied to provide visual explanations of the model’s decisions.

---

## Installation

### Prerequisites
- **Python**: Version 3.8–3.11
- **Operating System**: Windows, macOS, or Linux
- **Hardware**: CPU (GPU optional with CUDA support for faster inference)
- **Model Weights**: Pre-trained `.pth` files for each model (see [Model Weights](#model-weights))

### Medical Image Classifier

A Streamlit web app that uses deep learning models to classify medical images such as:

- **COVID-19 X-rays**
- **Kidney Disease**
- **Lung Cancer**

🔗 **Live Demo:** [medical-image-classifier.streamlit.app](https://medical-image-classifier.streamlit.app)

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ujwalreddybattu04/AI-Based-Disease-Diagnostic-.git
   cd AI-Based-Disease-Diagnostic-
