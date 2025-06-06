from transformers import pipeline
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
import os

_sentiment_analyzer = None
_semantic_model = None
_chatgroq = None

def sentiment_analyzer():
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        _sentiment_analyzer = pipeline("sentiment-analysis")
    return _sentiment_analyzer

def semantic_model():
    global _semantic_model
    if _semantic_model is None:
        _semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _semantic_model

def chatgroq():
    global _chatgroq
    if _chatgroq is None:
        _chatgroq = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama3-70b-8192",
            temperature=0.5
        )
    return _chatgroq
