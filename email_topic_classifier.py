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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import re
from sample_data import SAMPLE_EMAILS
from test_data import STRAIGHTFORWARD_TESTS, AMBIGUOUS_TESTS, TEST_EMAILS

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
            ('classifier', MultinomialNB(alpha=3.0))
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
    
    def train(self, test_size=0.2, random_state=42):
        """Train the classifier using the sample data."""
        # Add example senders to the training data
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
        
        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            emails_with_senders, [label for _, label in emails_with_senders],
            test_size=test_size, random_state=random_state, stratify=[label for _, label in emails_with_senders]
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
        
        # Save the model
        self.save("email_classifier.pkl")
        
        return self.cv_scores.mean(), self.cv_scores.std()
    
    def evaluate_on_test_data(self, test_data=None):
        """Evaluate the model on test data and print detailed metrics."""
        if test_data is None:
            test_data = TEST_EMAILS
            
        # Prepare test data
        test_texts = [text for text, _ in test_data]
        test_labels = [label for _, label in test_data]
        
        # Make predictions for all test texts
        predictions = self.predict(test_texts)
        
        # Calculate and print metrics
        print("\nTest Set Evaluation:")
        print("-" * 50)
        print(f"Test Accuracy: {accuracy_score(test_labels, predictions):.2%}")
        print("\nClassification Report:")
        print(classification_report(test_labels, predictions))
        print("\nConfusion Matrix:")
        print(confusion_matrix(test_labels, predictions))
        
        return accuracy_score(test_labels, predictions)

    def evaluate_all_test_sets(self):
        """Evaluate the model on both straightforward and ambiguous test sets."""
        print("\nEvaluating on Straightforward Test Cases:")
        print("=" * 50)
        straightforward_accuracy = self.evaluate_on_test_data(STRAIGHTFORWARD_TESTS)
        
        print("\nEvaluating on Ambiguous Test Cases:")
        print("=" * 50)
        ambiguous_accuracy = self.evaluate_on_test_data(AMBIGUOUS_TESTS)
        
        print("\nOverall Test Set Evaluation:")
        print("=" * 50)
        overall_accuracy = self.evaluate_on_test_data(TEST_EMAILS)
        
        print("\nSummary:")
        print("-" * 50)
        print(f"Straightforward Cases Accuracy: {straightforward_accuracy:.2%}")
        print(f"Ambiguous Cases Accuracy: {ambiguous_accuracy:.2%}")
        print(f"Overall Test Accuracy: {overall_accuracy:.2%}")
        
        return {
            'straightforward_accuracy': straightforward_accuracy,
            'ambiguous_accuracy': ambiguous_accuracy,
            'overall_accuracy': overall_accuracy
        }
    
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
    
    def predict(self, emails):
        """Predict the topic(s) of given email(s)."""
        # Handle both single email and list of emails
        if isinstance(emails, (str, tuple)):
            emails = [emails]
            
        # Process all emails
        processed_data = []
        for email in emails:
            if isinstance(email, tuple):
                content, sender = email
            else:
                content = email
                sender = "unknown"
            
            # Ensure content is a string
            if isinstance(content, tuple):
                content = content[0]
            
            processed_data.append({
                'content': self._preprocess_text(content),
                'sender': sender
            })
        
        # Convert to DataFrame
        processed_df = pd.DataFrame(processed_data)
        
        # Make predictions
        predictions = self.pipeline.predict(processed_df)
        
        # Return single prediction if input was single email
        if len(predictions) == 1 and isinstance(emails[0], (str, tuple)):
            return predictions[0]
        return predictions
    
    def predict_proba(self, emails):
        """Get probability scores for each topic."""
        # Handle both single email and list of emails
        if isinstance(emails, (str, tuple)):
            emails = [emails]
            
        # Process all emails
        processed_data = []
        for email in emails:
            if isinstance(email, tuple):
                content, sender = email
            else:
                content = email
                sender = "unknown"
            
            # Ensure content is a string
            if isinstance(content, tuple):
                content = content[0]
            
            processed_data.append({
                'content': self._preprocess_text(content),
                'sender': sender
            })
        
        # Convert to DataFrame
        processed_df = pd.DataFrame(processed_data)
        
        # Get probabilities
        probabilities = self.pipeline.predict_proba(processed_df)
        
        # Return single probability array if input was single email
        if len(probabilities) == 1 and isinstance(emails[0], (str, tuple)):
            return probabilities[0]
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
    """Main function to handle command line arguments and run the classifier."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Email Topic Classifier')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--predict', type=str, help='Predict topic for given email text')
    parser.add_argument('--sender', type=str, help='Email sender address for prediction')
    parser.add_argument('--test', action='store_true', help='Run evaluation on test data')
    parser.add_argument('--detailed-test', action='store_true', help='Run detailed evaluation on all test sets')
    args = parser.parse_args()
    
    classifier = EmailTopicClassifier()
    
    if args.train:
        print("Training the model...")
        classifier.train()
        print("Model trained and saved successfully!")
        
    elif args.predict:
        if not os.path.exists('email_classifier.pkl'):
            print("Error: Model not found. Please train the model first using --train")
            return
            
        classifier.load('email_classifier.pkl')
        prediction = classifier.predict((args.predict, args.sender if args.sender else "unknown"))
        probabilities = classifier.predict_proba((args.predict, args.sender if args.sender else "unknown"))
        
        print(f"\nPredicted Topic: {prediction}")
        print("\nProbabilities for each topic:")
        for topic, prob in zip(classifier.pipeline.classes_, probabilities):
            print(f"{topic}: {prob:.2%}")
            
    elif args.test:
        if not os.path.exists('email_classifier.pkl'):
            print("Error: Model not found. Please train the model first using --train")
            return
            
        classifier.load('email_classifier.pkl')
        classifier.evaluate_on_test_data()
        
    elif args.detailed_test:
        if not os.path.exists('email_classifier.pkl'):
            print("Error: Model not found. Please train the model first using --train")
            return
            
        classifier.load('email_classifier.pkl')
        metrics = classifier.evaluate_all_test_sets()
        
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 