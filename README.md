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
