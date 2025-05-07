import re
import string
from emoji import demojize
from nltk.stem import PorterStemmer
from textblob import TextBlob
import pandas as pd
import numpy as np
import pandas as pd

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

df=pd.read_csv(r"final_dataset_youtube_comments.csv")
new_df=df[df['Sentiment']!='neutral']
new_df = new_df.dropna(subset=['Comment'])
new_df['Sentiment']=new_df['Sentiment'].apply(lambda x:1 if x=='Positive' else 0 )

# Import necessary libraries
import re
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# --- Preprocessing Function ---
def clean_text(text):
    # Remove URLs and non-alphanumeric characters, and lowercase
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    return text.lower()

# Assuming new_df is your DataFrame with columns 'Comment' and 'Sentiment'
# Preprocess comments (optional, if you want to store the cleaned version)
new_df['Cleaned_Comment'] = new_df['Comment'].apply(preprocess_text)

# --- Data Splitting ---
X = new_df['Cleaned_Comment']
y = new_df['Sentiment']

# Split the data into training and testing sets (e.g., 80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# --- Feature Extraction ---
# Here we use bigrams as well and binary encoding
vectorizer = CountVectorizer(ngram_range=(1,2), binary=True)
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# --- Training the Classifier ---
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vect, y_train)

# --- Evaluate the Model ---
# Predict on the test set
y_pred = model.predict(X_test_vect)

# Calculate accuracy and print a classification report
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Test Accuracy:", accuracy)
print("Classification Report:")
print(report)

# Save the model and vectorizer for later use
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
with open('sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# --- Prediction Phase ---
# For new comments:
def predict_sentiment(new_comments):
    # Load saved artifacts
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('sentiment_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Preprocess new comments
    new_cleaned = [clean_text(comment) for comment in new_comments]

    # Transform text using the loaded vectorizer
    X_new = vectorizer.transform(new_cleaned)

    # Predict sentiment
    predictions = model.predict(X_new)
    probabilities = model.predict_proba(X_new)  # Optional: to get class probabilities

    return predictions, probabilities

# Calculate training and test accuracy scores
train_score = model.score(X_train_vect, y_train)
test_score = model.score(X_test_vect, y_test)

print("Training Accuracy:", train_score)
print("Test Accuracy:", test_score)

