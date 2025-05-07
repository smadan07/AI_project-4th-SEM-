# Import necessary libraries
import re
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

import numpy as np
import pandas as pd
from emoji import demojize
from nltk.stem import PorterStemmer
import string

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Convert emoji to text descriptions
    text = demojize(text)
    
    
    # Split text into tokens using whitespace
    tokens = text.split()
    
    # Initialize the stemmer and apply stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    
    # Join tokens back into a single string
    processed_text = ' '.join(stemmed_tokens)
    print(processed_text)
    return processed_text

def predict_sentiment(new_comments):
    # Load saved artifacts
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('sentiment_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Preprocess new comments
    new_cleaned = [preprocess_text(comment) for comment in new_comments]

    # Transform text using the loaded vectorizer
    X_new = vectorizer.transform(new_cleaned)

    # Predict sentiment
    predictions = model.predict(X_new)
    probabilities = model.predict_proba(X_new)  # Optional: to get class probabilities

    return predictions, probabilities
# Example usage:
new_comments = [
    '025',  # Sentiment shift
    "This is just the worst..." ,  # Sarcasm
    "The food was okay, but the service was terrible.",  # Mixed sentiment
    "I am absolutely NOT unhappy with this product.",  # Double negative
    "Wow... just wow. What an experience.",  # Ambiguous sentiment
    "I can't believe how amazing this turned out!",  # Strong positive emotion
    "It's decent, nothing too great but not bad either.",  # Neutral statement
    "I was expecting something good, but I got disappointed.",  # Expectation vs reality
    "This is beyond words... I have no idea what to feel.",  # Emotionally complex
    "Ugh, I guess it’s fine, whatever.",  # Indifference with slight negativity
]
predicted_labels, predicted_probs = predict_sentiment(new_comments)
print(predicted_labels)