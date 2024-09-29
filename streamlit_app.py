import streamlit as st
import json
import pandas as pd
import hr  # Import your hr.py file

st.title("Snap Secure Model Demo")

st.write("""
Welcome to the **Snap Secure Model Demo**, a live testing environment where you can upload motion data in JSON format and convert it into meaningful results.
This app supports multiple motion analysis categories across both VR and AR environments, along with heart rate and tremor detection models.
""")

# File input for each category
st.subheader("Heart Rate Models")
vr_heart_rate = st.file_uploader("Upload VR Heart Rate Models (CSV)", type=["csv"])  # CSV input for VR heart rate
ar_heart_rate = st.file_uploader("Upload AR Heart Rate Models (JSON)", type=["json"])

st.subheader("Tremor Detection")
tremor_detection = st.file_uploader("Upload Tremor Detection Data (JSON)", type=["json"])

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

    # Display heart rates in a line chart
    if heart_rates:
        heart_rates_df = pd.DataFrame(heart_rates, columns=["Heart Rate (BPM)"])
        heart_rates_df.index += 1  # Start index from 1 for better readability

        # Plotting the heart rates
        st.subheader("Heart Rates Over Time:")
        st.line_chart(heart_rates_df)

        # Display heart rates as a list
        st.subheader("Calculated Heart Rates:")
        for i, hr_value in enumerate(heart_rates, start=1):
            if hr_value is not None:
                st.write(f"Heart Rate at second {i}: {hr_value:.2f} BPM")
            else:
                st.write(f"No data at second {i}")

# You can add additional processing for AR heart rate and tremor detection here if needed
