import streamlit as st # type: ignore
import pickle
from preprocess import preprocess_text, extract_location, extract_disaster_type
from api_helpers import translate_text
from opencage.geocoder import OpenCageGeocode # type: ignore
import matplotlib.pyplot as plt
import folium # type: ignore
import plotly.express as px # type: ignore
import pandas as pd
import os

def apply_custom_css(theme):
    if theme == "Dark":
        css = """
        <style>
            body {
                background-color: #121212;
                color: #ffffff;
            }
            .stTextArea textarea {
                background-color: #333333;
                color: #ffffff;
            }
            .stButton button {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                cursor: pointer;
            }
            .stButton button:hover {
                background-color: #0056b3;
            }
        </style>
        """
    else:  # Light mode
        css = """
        <style>
            body {
                background-color: #f0f4f8;
                color: #333333;
            }
            .stTextArea textarea {
                background-color: #ffffff;
                color: #333333;
            }
            .stButton button {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                cursor: pointer;
            }
            .stButton button:hover {
                background-color: #0056b3;
            }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# OpenCage Geocoder
OPENCAGE_API_KEY = "0f667e5ad706483caae3a33292675c9b"
try:
    geocoder = OpenCageGeocode(OPENCAGE_API_KEY)
except Exception as e:
    st.error(f"Error initializing geocoder: {e}")
    geocoder = None

# Streamlit app
st.title("Disaster Tweet Classifier with Enhanced Details")

# Theme Selection
theme = st.sidebar.selectbox("Choose Theme", ["Light", "Dark"])
apply_custom_css(theme)

user_input = st.text_area("Enter a tweet:", key="text-area", height=150)

if "prediction_made" not in st.session_state:
    st.session_state["prediction_made"] = False
if "feedback_submitted" not in st.session_state:
    st.session_state["feedback_submitted"] = False

if st.button("Predict", key="button"):
    st.session_state["prediction_made"] = True
    st.session_state["feedback_submitted"] = False

if st.session_state["prediction_made"]:
    with st.spinner("Processing..."):
        # Translate non-English text to English
        translated_text = translate_text(user_input, target_language="en")
        if user_input != translated_text:
            st.write(f"Translated Text: {translated_text}")

        # Preprocess the input
        cleaned_input = preprocess_text(translated_text)

        # Vectorize the input
        input_tfidf = vectorizer.transform([cleaned_input])

        # Predict
        prediction = model.predict(input_tfidf)[0]
        probabilities = model.predict_proba(input_tfidf)[0] 
        confidence_score = max(probabilities)
        
        # confidence threshold
        confidence_threshold = 0.3
        if confidence_score < confidence_threshold:
            result = "Uncertain"
        else:
            result = "Disaster-related" if prediction == 1 else "Not disaster-related"

        # Extract location and disaster type
        location = extract_location(translated_text)
        disaster_type = extract_disaster_type(translated_text)

        # Geocode the location to get longitude and latitude
        longitude, latitude = None, None
        if geocoder and location and location != "No location detected":
            try:
                results = geocoder.geocode(location)
                if results and len(results):
                    longitude = results[0]['geometry']['lng']
                    latitude = results[0]['geometry']['lat']
            except Exception as e:
                st.error(f"Error during geocoding: {e}")

    st.write(f"Prediction: {result}")
    st.write(f"Confidence Score: {confidence_score:.2f}")

    fig = px.bar(
        x=['Not disaster-related', 'Disaster-related'],
        y=probabilities,
        title='Class Probabilities',
        labels={'x': 'Class', 'y': 'Probability'},
        color=['Not disaster-related', 'Disaster-related'],
        color_discrete_map={'Not disaster-related': '#1f77b4', 'Disaster-related': '#ff7f0e'}
    )
    st.plotly_chart(fig)

    st.write(f"Location: {location}")
    st.write(f"Type of Disaster: {disaster_type}")
    
    if longitude and latitude:
        st.write(f"Longitude: {longitude}, Latitude: {latitude}")
        
        m = folium.Map(location=[latitude, longitude], zoom_start=10, tiles='OpenStreetMap')
        folium.Marker([latitude, longitude], popup=f"{location} - {disaster_type}").add_to(m)
        st.write("Map:")
        st.components.v1.html(m._repr_html_(), width=800, height=600)
    else:
        st.write("Longitude and Latitude: Not available")

    if not st.session_state["feedback_submitted"]:
        st.write("Was the prediction correct?")
        feedback = st.radio("Select your feedback:", ["Yes", "No"], horizontal=True)
        if st.button("Submit Feedback"):
            st.session_state["feedback_submitted"] = True
            feedback_data = {
                "Tweet": user_input,
                "Translated Text": translated_text,
                "Prediction": result,
                "Actual Label": "Disaster-related" if feedback == "Yes" and result == "Disaster-related" else "Not disaster-related",
                "Feedback": feedback
            }
            feedback_df = pd.DataFrame([feedback_data])

            try:
                if not os.path.exists("user_feedback.csv"):
                    feedback_df.to_csv("user_feedback.csv", mode='w', index=False, encoding='utf-8')
                else:
                    existing_df = pd.read_csv("user_feedback.csv", encoding='utf-8')

                    updated_df = pd.concat([existing_df, feedback_df], ignore_index=True)
                    updated_df.to_csv("user_feedback.csv", mode='w', index=False, encoding='utf-8')

                st.success("Thank you for your feedback! It will help improve the model.")
            except Exception as e:
                st.error(f"Error saving feedback: {e}")