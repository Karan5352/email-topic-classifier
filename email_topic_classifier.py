import os
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import re
from sample_data import SAMPLE_EMAILS
from test_data import TEST_EMAILS

class EmailTopicClassifier:
    def __init__(self):
        """Initialize the email classifier."""
        # Create a pipeline for text content with improved parameters
        text_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=10000,  # Increased to handle more vocabulary
                stop_words='english',  # Use English stop words
                ngram_range=(1, 3),  # Include trigrams for better context
                min_df=2,  # Minimum document frequency
                max_df=0.95,  # Maximum document frequency
                sublinear_tf=True,
                strip_accents='unicode',
                analyzer='word'
            ))
        ])
        
        # Create a pipeline for sender information
        sender_pipeline = Pipeline([
            ('extract_domain', FunctionTransformer(self._extract_domain)),
            ('tfidf', TfidfVectorizer(
                max_features=2000,  # Increased for more domain patterns
                ngram_range=(1, 2),
                min_df=2,
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
        
        # Initialize metrics
        self.cv_scores = None
        self.classification_metrics = None
        self.confusion_mat = None
        self.test_metrics = None
    
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
        
        # Keep alphanumeric characters, basic punctuation, and newlines
        # Note: We need to escape the hyphen and handle newline separately
        text = re.sub(r'[^a-zA-Z0-9\s.,!?\-\n]', ' ', text)
        
        # Replace multiple newlines with space
        text = re.sub(r'\n+', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def train(self, emails, labels, test_size=0.2, random_state=42):
        """Train the classifier and evaluate on test set."""
        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            emails, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        # Process training data
        processed_train = []
        for email in X_train:
            if isinstance(email, tuple):
                content, sender = email
            else:
                content = email
                sender = "unknown"
            
            if isinstance(content, tuple):
                content = content[0]
            
            processed_text = self._preprocess_text(content)
            processed_train.append({
                'content': processed_text,
                'sender': sender
            })
        
        # Convert to DataFrame
        X_train_df = pd.DataFrame(processed_train)
        
        # Process test data
        processed_test = []
        for email in X_test:
            if isinstance(email, tuple):
                content, sender = email
            else:
                content = email
                sender = "unknown"
            
            if isinstance(content, tuple):
                content = content[0]
            
            processed_text = self._preprocess_text(content)
            processed_test.append({
                'content': processed_text,
                'sender': sender
            })
        
        X_test_df = pd.DataFrame(processed_test)
        
        # Perform cross-validation on training data
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        self.cv_scores = cross_val_score(self.pipeline, X_train_df, y_train, cv=cv, scoring='accuracy')
        
        # Train the final model
        self.pipeline.fit(X_train_df, y_train)
        
        # Evaluate on test set
        y_pred = self.pipeline.predict(X_test_df)
        self.test_metrics = {
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'accuracy': (y_pred == y_test).mean()
        }
        
        # Print metrics
        print("\nTraining Cross-validation scores:", self.cv_scores)
        print("Mean CV accuracy: {:.2f}% (+/- {:.2f}%)".format(
            self.cv_scores.mean() * 100, self.cv_scores.std() * 2 * 100))
        
        print("\nTest Set Performance:")
        print(f"Test accuracy: {self.test_metrics['accuracy']:.2%}")
        print("\nTest Set Classification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nTest Set Confusion Matrix:")
        print(self.test_metrics['confusion_matrix'])
        
        return self.cv_scores.mean(), self.cv_scores.std()
    
    def evaluate_on_test_data(self, test_emails=None):
        """Evaluate the model on the separate test dataset."""
        if test_emails is None:
            test_emails = TEST_EMAILS
        
        # Process test data
        processed_test = []
        test_labels = []
        
        for email, label in test_emails:
            if isinstance(email, tuple):
                content, sender = email
            else:
                content = email
                sender = "unknown"
            
            if isinstance(content, tuple):
                content = content[0]
            
            processed_text = self._preprocess_text(content)
            processed_test.append({
                'content': processed_text,
                'sender': sender
            })
            test_labels.append(label)
        
        X_test_df = pd.DataFrame(processed_test)
        
        # Make predictions
        y_pred = self.pipeline.predict(X_test_df)
        
        # Calculate metrics
        test_metrics = {
            'classification_report': classification_report(test_labels, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(test_labels, y_pred),
            'accuracy': (y_pred == test_labels).mean()
        }
        
        # Print metrics
        print("\nTest Dataset Performance:")
        print(f"Test accuracy: {test_metrics['accuracy']:.2%}")
        print("\nTest Dataset Classification Report:")
        print(classification_report(test_labels, y_pred))
        
        print("\nTest Dataset Confusion Matrix:")
        print(test_metrics['confusion_matrix'])
        
        return test_metrics
    
    def get_metrics(self):
        """Get the model's performance metrics."""
        return {
            'cv_scores': self.cv_scores,
            'mean_accuracy': self.cv_scores.mean() if self.cv_scores is not None else None,
            'std_accuracy': self.cv_scores.std() if self.cv_scores is not None else None,
            'classification_metrics': self.classification_metrics,
            'confusion_matrix': self.confusion_mat,
            'test_metrics': self.test_metrics
        }
    
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
            content = content[0]
        
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
            content = content[0]
        
        processed_data = pd.DataFrame([{
            'content': self._preprocess_text(content),
            'sender': sender
        }])
        
        probabilities = self.pipeline.predict_proba(processed_data)[0]
        return probabilities
    
    def save(self, path="model.pkl"):
        """Save the trained model and metrics to a file."""
        model_data = {
            'pipeline': self.pipeline,
            'cv_scores': self.cv_scores,
            'classification_metrics': self.classification_metrics,
            'confusion_mat': self.confusion_mat,
            'test_metrics': self.test_metrics
        }
        with open(path, "wb") as f:
            pickle.dump(model_data, f)
    
    def load(self, path="model.pkl"):
        """Load a trained model and metrics from a file."""
        with open(path, "rb") as f:
            model_data = pickle.load(f)
            self.pipeline = model_data['pipeline']
            self.cv_scores = model_data['cv_scores']
            self.classification_metrics = model_data['classification_metrics']
            self.confusion_mat = model_data['confusion_mat']
            self.test_metrics = model_data.get('test_metrics')

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--predict", type=str, help="Predict topic for given email text")
    parser.add_argument("--sender", type=str, help="Sender email address")
    parser.add_argument("--metrics", action="store_true", help="Show model metrics")
    parser.add_argument("--test", action="store_true", help="Evaluate on test dataset")
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
        
        mean_acc, std_acc = clf.train(emails_with_senders, [label for _, label in emails_with_senders])
        clf.save()
        print(f"\nModel trained and saved as model.pkl")
        print(f"Mean accuracy: {mean_acc:.2%} (±{std_acc:.2%})")
    elif args.test:
        if not os.path.exists("model.pkl"):
            print("Model not found. Run with --train first.")
            return
        clf.load()
        clf.evaluate_on_test_data()
    elif args.predict:
        if not os.path.exists("model.pkl"):
            print("Model not found. Run with --train first.")
            return
        clf.load()
        email = (args.predict, args.sender if args.sender else "unknown")
        topic = clf.predict(email)
        print(f"Predicted topic: {topic}")
    elif args.metrics:
        if not os.path.exists("model.pkl"):
            print("Model not found. Run with --train first.")
            return
        clf.load()
        metrics = clf.get_metrics()
        print("\nModel Performance Metrics:")
        print(f"Mean CV accuracy: {metrics['mean_accuracy']:.2%} (±{metrics['std_accuracy']:.2%})")
        if metrics['test_metrics']:
            print(f"\nTest accuracy: {metrics['test_metrics']['accuracy']:.2%}")
        print("\nPer-class metrics:")
        for label, scores in metrics['classification_metrics'].items():
            if isinstance(scores, dict):
                print(f"\n{label}:")
                print(f"  Precision: {scores['precision']:.2%}")
                print(f"  Recall: {scores['recall']:.2%}")
                print(f"  F1-score: {scores['f1-score']:.2%}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 