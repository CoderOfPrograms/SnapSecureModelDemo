import streamlit as st
from VRNodModel import predict_single_json
import json

st.title("Snap Secure Model Demo")

st.write("""
Welcome to the **Snap Secure Model Demo**, a live testing environment where you can upload motion data in JSON format and convert it into meaningful results.
This app supports multiple motion analysis categories across both VR and AR environments, along with heart rate and tremor detection models.
""")

# File input for each category
st.subheader("VR Motion Analysis")
# File input for VR Nod Analysis and other categories
vr_nod = st.file_uploader("Upload VR Nod JSON", type=["json"])

# Define the path to the pre-trained model (relative to the current working directory)
model_path = "NodTripleThreat.pkl"

if vr_nod:
    st.write("VR Nod JSON file uploaded! Processing...")

    # Load the uploaded JSON file
    json_data = json.load(vr_nod)

    # Run the prediction function
    result = predict_single_json(json_data, model_path)

    # Display results
    if result:
        st.write("Prediction Results:")
        st.json(result)
    else:
        st.write("No valid data found for prediction.")
vr_wave = st.file_uploader("VR Wave Motion Analysis (Controllers)", type=["json"])
vr_ymca = st.file_uploader("VR YMCA Analysis", type=["json"])

st.subheader("AR Motion Analysis")
ar_nod = st.file_uploader("AR Nod Analysis", type=["json"])
ar_tap = st.file_uploader("AR Tap Gesture", type=["json"])

st.subheader("Heart Rate Models")
vr_heart_rate = st.file_uploader("VR Heart Rate Models", type=["csv"])  # CSV input for VR heart rate
ar_heart_rate = st.file_uploader("AR Heart Rate Models", type=["json"])

st.subheader("Tremor Detection")
tremor_detection = st.file_uploader("Tremor Detection", type=["json"])

# Example of processing data after upload
if vr_wave:
    st.write("VR Wave Motion Analysis uploaded!")
    # You can add code here to process and analyze the uploaded file

if vr_heart_rate:
    st.write("VR Heart Rate Model CSV uploaded!")
    # You can add code here to process and analyze the CSV file
