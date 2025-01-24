import streamlit as st
import os
import pickle

# Define the class mapping
CLASS_MAPPING = {0:'Ham', 1:'Spam'}
# Function to load the model
@st.cache_resource
def load_model():
    try:
        # Determine model file path
        model_path = os.path.join(os.path.dirname(__file__), "spam_mail_prediction.pkl")
        
        # Load model in binary mode
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please upload the correct 'modek' file.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

# Load the model
model = load_model()

# Streamlit app interface
st.title("Spam mail prediction")
st.write("Enter a text snippet, and the model will predict Spam/Ham.")

# Input text field
user_input = st.text_input("Input Text:")

# Predict language
if st.button("Predict"):
    if not model:
        st.error("Model is not loaded. Please check the logs for errors.")
    elif user_input.strip() == "":
        st.warning("Please enter some text to detect the language.")
    else:
        try:
            # Predict the class index
            class_index = model.predict([user_input])[0]
            
            # Get the corresponding language name
            Pred_mail= CLASS_MAPPING.get(class_index, "Unknown Language")
            
            st.success(f"Predicted: **{Pred_mail}**")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
