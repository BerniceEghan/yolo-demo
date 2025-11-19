import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("best.pt")   # Ensure best.pt is in same GitHub repo folder

model = load_model()

# Streamlit UI
st.title("YOLOv11 Object Detection App")
st.write("Upload an image to detect **Cheerios**, **Soup**, and **Candle** objects.")

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and display image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Run detection
    st.write("Running YOLOv11m inference...")
    results = model.predict(img)

    # Render detection
    result_img = results[0].plot()

    # Display output image
    st.image(result_img, caption="Detection Result", use_column_width=True)

    # Show raw predictions
    st.subheader("Raw Prediction Output")
    st.json(results[0].tojson())
