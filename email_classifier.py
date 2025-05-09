import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

class EmailClassifier:
    def __init__(self):
        """Initialize the email classifier."""
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2)
            )),
            ('classifier', MultinomialNB())
        ])
        
    def preprocess_text(self, text):
        """Preprocess the text by removing special characters and converting to lowercase."""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def train(self, texts, labels):
        """Train the classifier on the given texts and labels."""
        # Preprocess all texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        # Train the pipeline
        self.pipeline.fit(processed_texts, labels)
    
    def predict(self, text):
        """Predict the topic of a given email text."""
        # Preprocess the text
        processed_text = self.preprocess_text(text)
        # Make prediction
        prediction = self.pipeline.predict([processed_text])[0]
        return prediction
    
    def predict_proba(self, text):
        """Get probability scores for each topic."""
        processed_text = self.preprocess_text(text)
        probabilities = self.pipeline.predict_proba([processed_text])[0]
        return probabilities
    
    def save_model(self, filepath):
        """Save the trained model to a file."""
        joblib.dump(self.pipeline, filepath)
    
    def load_model(self, filepath):
        """Load a trained model from a file."""
        self.pipeline = joblib.load(filepath) 