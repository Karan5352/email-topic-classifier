import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
import re
from sample_data import SAMPLE_EMAILS

class EmailTopicClassifier:
    def __init__(self):
        """Initialize the email classifier."""
        # Create a pipeline for text content
        text_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                stop_words=None,  # Disable stop words to see all terms
                ngram_range=(1, 2),
                min_df=1,
                max_df=1.0,
                sublinear_tf=True,
                strip_accents='unicode',
                analyzer='word'
            ))
        ])
        
        # Create a pipeline for sender information
        sender_pipeline = Pipeline([
            ('extract_domain', FunctionTransformer(self._extract_domain)),
            ('tfidf', TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                min_df=1,
                analyzer='word'
            ))
        ])
        
        # Combine both pipelines
        self.pipeline = Pipeline([
            ('features', ColumnTransformer([
                ('text', text_pipeline, 'content'),
                ('sender', sender_pipeline, 'sender')
            ])),
            ('classifier', MultinomialNB(alpha=0.1))
        ])
        
    def _extract_domain(self, senders):
        """Extract domain from email addresses."""
        domains = []
        for sender in senders:
            if '@' in sender:
                domain = sender.split('@')[1].lower()
                domains.append(domain)
            else:
                domains.append(sender.lower())
        return domains
    
    def _preprocess_text(self, text):
        """Preprocess the text by removing special characters and converting to lowercase."""
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Keep only alphanumeric characters and basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,!?-]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def train(self, emails, labels):
        """Train the classifier on the given emails and labels."""
        # Convert emails to the format expected by the pipeline
        processed_data = []
        for email, label in zip(emails, labels):
            if isinstance(email, tuple):
                content, sender = email
            else:
                content = email
                sender = "unknown"
            
            # Ensure content is a string
            if isinstance(content, tuple):
                content = content[0]  # Take the first element if it's a tuple
            
            processed_text = self._preprocess_text(content)
            processed_data.append({
                'content': processed_text,
                'sender': sender
            })
            
            # Debug: Print processed text
            print(f"Original: {content}")
            print(f"Processed: {processed_text}")
            print("---")
        
        # Convert to DataFrame for ColumnTransformer
        X = pd.DataFrame(processed_data)
        self.pipeline.fit(X, labels)
        
        # Debug: Print vocabulary
        text_transformer = self.pipeline.named_steps['features'].transformers_[0][1]
        vocabulary = text_transformer.named_steps['tfidf'].vocabulary_
        print("\nVocabulary size:", len(vocabulary))
        print("Sample terms:", list(vocabulary.keys())[:10])
    
    def predict(self, email):
        """Predict the topic of a given email."""
        # Convert email to the format expected by the pipeline
        if isinstance(email, tuple):
            content, sender = email
        else:
            content = email
            sender = "unknown"
        
        # Ensure content is a string
        if isinstance(content, tuple):
            content = content[0]  # Take the first element if it's a tuple
        
        processed_data = pd.DataFrame([{
            'content': self._preprocess_text(content),
            'sender': sender
        }])
        
        # Make prediction
        prediction = self.pipeline.predict(processed_data)[0]
        return prediction
    
    def predict_proba(self, email):
        """Get probability scores for each topic."""
        # Convert email to the format expected by the pipeline
        if isinstance(email, tuple):
            content, sender = email
        else:
            content = email
            sender = "unknown"
        
        # Ensure content is a string
        if isinstance(content, tuple):
            content = content[0]  # Take the first element if it's a tuple
        
        processed_data = pd.DataFrame([{
            'content': self._preprocess_text(content),
            'sender': sender
        }])
        
        probabilities = self.pipeline.predict_proba(processed_data)[0]
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
    parser.add_argument("--sender", type=str, help="Sender email address")
    args = parser.parse_args()

    clf = EmailTopicClassifier()

    if args.train:
        # Add some example senders to the training data
        emails_with_senders = []
        for content, label in SAMPLE_EMAILS:
            # Generate appropriate sender based on topic
            if label == "work":
                sender = "hr@company.com"
            elif label == "shipping":
                sender = "shipping@amazon.com"
            elif label == "finance":
                sender = "notifications@bank.com"
            elif label == "travel":
                sender = "bookings@expedia.com"
            elif label == "promotions":
                sender = "offers@store.com"
            elif label == "social":
                sender = "notifications@facebook.com"
            elif label == "updates":
                sender = "updates@service.com"
            elif label == "support":
                sender = "support@company.com"
            elif label == "spam":
                sender = "noreply@suspicious.com"
            elif label == "events":
                sender = "events@conference.com"
            else:
                sender = "unknown@example.com"
            
            emails_with_senders.append(((content, sender), label))
        
        clf.train(emails_with_senders, [label for _, label in emails_with_senders])
        clf.save()
        print("Model trained and saved as model.pkl")
    elif args.predict:
        if not os.path.exists("model.pkl"):
            print("Model not found. Run with --train first.")
            return
        clf.load()
        email = (args.predict, args.sender if args.sender else "unknown")
        topic = clf.predict(email)
        print(f"Predicted topic: {topic}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 