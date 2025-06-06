# myapp/ml_models.py
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
import os

# Load sentiment and embedding models
sentiment_analyzer = pipeline("sentiment-analysis")
semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load ChatGroq
chatgroq = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama3-70b-8192",
    temperature=0.5
)
