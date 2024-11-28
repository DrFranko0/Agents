import streamlit as st
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
from PIL import Image
import torch
import requests
from io import BytesIO
import json

qa_model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
qa_tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
qa_pipeline = pipeline("question-answering", model=qa_model, tokenizer=qa_tokenizer)

image_classification_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
image_classification_model.eval()

def classify_image(image):
    preprocess = torch.transforms.Compose([
        torch.transforms.Resize(256),
        torch.transforms.CenterCrop(224),
        torch.transforms.ToTensor(),
        torch.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = preprocess(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = image_classification_model(image_tensor)
    
    _, predicted_idx = torch.max(outputs, 1)
    labels = requests.get('https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json').json()
    predicted_label = labels[str(predicted_idx.item())][1]
    
    return predicted_label

with open("agriculture_data.json", "r") as f:
    agricultural_knowledge_base = json.load(f)

st.set_page_config(page_title="AI-Powered Agricultural Assistant", layout="wide")

st.title("AI-Powered Agricultural Conversational Assistant")
st.subheader("Helping farmers with crop information and image identification")

st.sidebar.title("Navigation")
option = st.sidebar.radio("Choose an option:", ["Ask a Question", "Identify Crop"])

def get_crop_details(crop_name):
    if crop_name.lower() in agricultural_knowledge_base:
        crop_info = agricultural_knowledge_base[crop_name.lower()]
        return crop_info
    else:
        return None

if option == "Ask a Question":
    st.header("Ask a Question about Crops and Seasons")

    user_question = st.text_input("Enter your question here:")

    if user_question:
        context = " ".join([f"{crop.capitalize()}: {details}" for crop, details in agricultural_knowledge_base.items()])
        
        result = qa_pipeline(question=user_question, context=context)
        
        st.write(f"Answer: {result['answer']}")

        if "crop" in user_question.lower():
            crop_name = user_question.split()[-1]
            crop_info = get_crop_details(crop_name)
            if crop_info:
                st.write(f"Detailed Information for {crop_name.capitalize()}:")
                st.write(f"Regions: {', '.join(crop_info['regions'])}")
                st.write(f"Growing Season: {crop_info['growing_season']}")
                st.write(f"Ideal Conditions: {crop_info['ideal_conditions']}")
                st.write(f"Common Diseases: {', '.join(crop_info['common_diseases'])}")
                st.write(f"Pests: {', '.join(crop_info['pests'])}")
            else:
                st.write(f"Sorry, no detailed information available for {crop_name}.")

elif option == "Identify Crop":
    st.header("Upload an Image to Identify Crop")

    uploaded_image = st.file_uploader("Choose an image of a crop", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)

        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Classifying the crop..."):
            predicted_label = classify_image(image)
            st.write(f"The crop in the image is identified as: {predicted_label}")

st.markdown("""
    ### About this project:
    This project leverages HuggingFace models for answering agricultural-related questions and 
    PyTorch-based image classification to identify crops from images.
    """)
