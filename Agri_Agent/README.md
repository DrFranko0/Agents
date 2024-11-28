# AI-Powered Agricultural Conversational Assistant

This application is designed to assist farmers and agricultural enthusiasts by answering questions related to crops and seasons and identifying crops from uploaded images.

## Features

1. **Ask a Question**: 
   - Get answers to questions about crops, their growing seasons, ideal conditions, common diseases, and pests.
   - Powered by Hugging Face's `deepset/roberta-base-squad2` model for question answering.

2. **Identify Crop**: 
   - Upload an image of a crop to identify it using a ResNet50 model pretrained on the ImageNet dataset.

3. **User-Friendly Interface**:
   - A simple and intuitive Streamlit-based interface.
   - Sidebar navigation for quick access to features.

---

## Requirements

### Python Packages
The application requires the following Python libraries:

- `streamlit`
- `transformers`
- `torch`
- `Pillow`
- `requests`

### Models and Resources
- NLP Model: `deepset/roberta-base-squad2` from Hugging Face Transformers.
- Image Classification Model: Pretrained ResNet50 from PyTorch's Torch Hub.
- ImageNet Class Index: JSON file fetched dynamically from [TensorFlow storage](https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json).

---


## Clone the repository:
   ```bash
   git clone https://github.com/DrFranko0/Agents/Agri_Agent.git
