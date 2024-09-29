import streamlit as st
import json
import hr  # Import your hr.py file

st.title("Snap Secure Model Demo")

st.write("""
Welcome to the **Snap Secure Model Demo**, a live testing environment where you can upload motion data in JSON format and convert it into meaningful results.
This app supports multiple motion analysis categories across both VR and AR environments, along with heart rate and tremor detection models.
""")

# File input for each category
st.subheader("Heart Rate Models")
vr_heart_rate = st.file_uploader("VR Heart Rate Models", type=["csv"])  # CSV input for VR heart rate
ar_heart_rate = st.file_uploader("AR Heart Rate Models", type=["json"])

st.subheader("Tremor Detection")
tremor_detection = st.file_uploader("Tremor Detection", type=["json"])

# Example of processing data after upload
if vr_heart_rate:
    # Read the uploaded CSV file
    csv_data = vr_heart_rate.read()  # Read the file as bytes
    csv_file_path = "uploaded_vr_heart_rate.csv"  # Temporary path for processing
    with open(csv_file_path, "wb") as f:
        f.write(csv_data)  # Write bytes to a temporary file

    st.write("VR Heart Rate Model CSV uploaded!")
    
    # Calculate heart rates using the imported function
    heart_rates = hr.calculate_and_return_heart_rates(csv_file_path)

    # Display heart rates
    st.subheader("Calculated Heart Rates:")
    for i, hr_value in enumerate(heart_rates):
        if hr_value is not None:
            st.write(f"Heart Rate at second {i+1}: {hr_value:.2f} BPM")
        else:
            st.write(f"No data at second {i+1}")

# You can add additional processing for AR heart rate and tremor detection here if needed
