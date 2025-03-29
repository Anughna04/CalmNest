#import the required libraries
import streamlit as st
import pickle
import numpy as np
import string
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random
import keras
import json

# Load the model, tokenizer, and label encoder
model = keras.models.load_model('chatbot_model.keras',compile=False)

with open('tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

with open('label_encoder.pkl', 'rb') as le_file:
    le = pickle.load(le_file)

# Define the input shape based on the trained model
input_shape = model.input_shape[1]

# Load the responses from the intents file
with open('intent.json') as content:
    data1 = json.load(content)

responses = {}
for intent in data1['intents']:
    responses[intent['tag']] = list(set(intent['responses']))  # Store unique responses

max_len=20
# Streamlit UI Enhancements
col1, col2 = st.columns([1.5,2])

with col1:
    st.image("bot.png", width=270)  # Adjust width as needed

with col2:
    st.markdown("<h3 style='margin-bottom: 1px;'></h3>", unsafe_allow_html=True)
    st.markdown("""
    <h1 style='text-align: center; color: #87CEEB; text-shadow: 2px 2px 15px rgba(135, 206, 235, 0.8);'>
        CalmNest 🤍
    </h1>
    """, unsafe_allow_html=True)
    st.markdown("<h4 style='margin-top: 3.5px;'>✨Where your mind finds comfort✨</h4>", unsafe_allow_html=True)

st.write("✧","・"*40,"✧")
st.markdown("<h5 style='margin-top: 0px;margin-left: 60px;'>💭Share your thoughts, I'm here to listen and help you heal</h5>", unsafe_allow_html=True)
st.write("✧","・"*40,"✧")

# Initialize chat history if not present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history with enhanced formatting
st.write("### Chat:")
unique_messages = set()
for sender, message in st.session_state.chat_history:
    if (sender, message) not in unique_messages:
        unique_messages.add((sender, message))
        if sender == "Bot":
            st.markdown(f"<p style='font-size:18px;'><span style='font-size:22px;'>🤖</span> {message}</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p style='font-size:18px;'><span style='font-size:22px;'>👤</span> {message}</p>", unsafe_allow_html=True)
        st.write("---")

# User input field at the bottom
user_input = st.text_input("Type your message here:", "")

if user_input:
    # Preprocess the user input
    user_input_clean = ''.join([letters.lower() for letters in user_input if letters not in string.punctuation])
    
    # Tokenize and pad the input
    prediction_input = tokenizer.texts_to_sequences([user_input_clean])
    if not prediction_input or all(len(seq) == 0 for seq in prediction_input):
        bot_response = "I'm sorry, I didn't understand that. Could you rephrase?"
    else:
        prediction_input = np.array(prediction_input).reshape(-1)
        prediction_input = pad_sequences([prediction_input],input_shape)

        # Predict the response
        output = model.predict(prediction_input)
        output = output.argmax()

        # Get the predicted tag and response
        response_tag = le.inverse_transform([output])[0]
        bot_response = responses[response_tag][0]  # Always pick the first response to avoid multiple ones

    # Update chat history
    st.session_state.chat_history.append(("User", user_input))
    st.session_state.chat_history.append(("Bot", bot_response))

    # If the tag is 'goodbye', add a farewell message
    if response_tag == "goodbye":
        st.session_state.chat_history.append(("Bot", "Goodbye! It was nice chatting with you."))

    st.rerun()

# CSS for animations and better styling
st.markdown(
    """
    <style>
        body { background-color: #f4f4f4; }
        .stTextInput>div>div>input { border-radius: 10px; padding: 10px; font-size: 18px; }
        .stMarkdown { animation: fadeIn 0.5s ease-in-out; }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
    </style>
    """,
    unsafe_allow_html=True
)
