# Import necessary libraries
import streamlit as st
import requests
from typing import List
from transformers import pipeline
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer

# Set up Streamlit application title
st.title("University AI Assistant")

# Initialize Qdrant client and Sentence Transformer model
qdrant = QdrantClient("http://localhost:6333")
print("Downloading Encoder")
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Downloading Done")

# Function to generate response based on user input
def generate_response(input_text):
    print(input_text)
    hits = qdrant.search(
    collection_name="university_data",
    query_vector=encoder.encode(input_text).tolist(),
    limit=3,
    )
    # Display top 3 hits
    st.info(hits[0].payload)
    st.text(" ".join(["Score:", str(hits[0].score)]))
    st.info(hits[1].payload)
    st.text(" ".join(["Score:", str(hits[1].score)]))
    st.info(hits[2].payload)
    st.text(" ".join(["Score:", str(hits[2].score)]))

# Create a form for user input
with st.form("uni_form"):
    text = st.text_area("Hello there! How can I assist you today?")
    submitted = st.form_submit_button("Ask")
    if submitted:
        # Call function to generate response based on user input
        generate_response(text)
