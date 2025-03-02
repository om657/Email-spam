import streamlit as st
import pickle
import pandas as pd
om = "spam_classifier_model.pkl"
ok = "tfidf_vectorizer.pkl"
# Load the saved model and vectorizer
model = pickle.load(open(om,'rb'))
vectorizer = pickle.load(open(ok, 'rb'))

# Create the Streamlit app
st.title("Spam Email Classifier")

# Get user input
user_input = st.text_area("Enter an email message:")

# Make prediction
if st.button("Predict"):
    if user_input:
        try:
            # Transform the user input using the loaded vectorizer
            input_tfidf = vectorizer.transform([user_input])
            prediction = model.predict(input_tfidf)[0]

            st.write(f"Prediction: {prediction}")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")  # Display error message
    else:
        st.warning("Please enter an email message.")

