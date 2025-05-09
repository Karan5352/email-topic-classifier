import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import re
from sample_data import SAMPLE_EMAILS

class EmailTopicClassifier:
    def __init__(self):
        """Initialize the email classifier."""
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=15000,  # Increased features
                stop_words='english',
                ngram_range=(1, 3),
                min_df=2,
                max_df=0.95,  # Added max document frequency
                sublinear_tf=True  # Added sublinear term frequency scaling
            )),
            ('classifier', MultinomialNB(alpha=0.05))  # Reduced alpha for better precision
        ])
        
    def preprocess_text(self, text):
        """Preprocess the text by removing special characters and converting to lowercase."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove common words that don't add meaning
        common_words = {'please', 'thank', 'thanks', 'regards', 'best', 'sincerely', 'dear'}
        text = ' '.join(word for word in text.split() if word not in common_words)
        
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
    
    def save(self, path="model.pkl"):
        """Save the trained model to a file."""
        with open(path, "wb") as f:
            pickle.dump(self.pipeline, f)
    
    def load(self, path="model.pkl"):
        """Load a trained model from a file."""
        with open(path, "rb") as f:
            self.pipeline = pickle.load(f)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--predict", type=str, help="Predict topic for given email text")
    args = parser.parse_args()

    clf = EmailTopicClassifier()

    if args.train:
        texts, labels = zip(*SAMPLE_EMAILS)
        clf.train(texts, labels)
        clf.save()
        print("Model trained and saved as model.pkl")
    elif args.predict:
        if not os.path.exists("model.pkl"):
            print("Model not found. Run with --train first.")
            return
        clf.load()
        topic = clf.predict(args.predict)
        print(f"Predicted topic: {topic}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 