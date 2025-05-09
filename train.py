import pandas as pd
import argparse
from email_classifier import EmailClassifier
import os

def create_sample_data():
    """Create a sample dataset for demonstration purposes."""
    data = {
        'text': [
            "Meeting scheduled for tomorrow at 2 PM to discuss project timeline",
            "Please review the attached quarterly financial report",
            "Your order #12345 has been shipped and will arrive in 3-5 business days",
            "New software update available for your device",
            "Thank you for your job application, we'd like to schedule an interview",
            "Your subscription will renew automatically next month",
            "Security alert: unusual login attempt detected",
            "Your payment of $50 has been processed successfully",
            "Team building event this Friday at 4 PM",
            "Please complete the customer satisfaction survey"
        ],
        'topic': [
            'meeting',
            'report',
            'shipping',
            'software',
            'job',
            'subscription',
            'security',
            'payment',
            'event',
            'survey'
        ]
    }
    return pd.DataFrame(data)

def main():
    parser = argparse.ArgumentParser(description='Train the email topic classifier')
    parser.add_argument('--data_path', type=str, help='Path to the training data CSV file')
    parser.add_argument('--model_path', type=str, default='model.pkl', help='Path to save the trained model')
    args = parser.parse_args()

    # Load or create sample data
    if args.data_path and os.path.exists(args.data_path):
        df = pd.read_csv(args.data_path)
    else:
        print("No data file provided or file not found. Using sample data...")
        df = create_sample_data()
        # Save sample data for reference
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/sample_data.csv', index=False)
        print("Sample data saved to data/sample_data.csv")

    # Initialize and train the classifier
    classifier = EmailClassifier()
    classifier.train(df['text'], df['topic'])
    
    # Save the trained model
    classifier.save_model(args.model_path)
    print(f"Model trained and saved to {args.model_path}")

    # Test the model with a few examples
    print("\nTesting the model with some examples:")
    test_emails = [
        "Can we schedule a meeting to discuss the project?",
        "Your payment of $100 has been received",
        "New security update available for your account"
    ]
    
    for email in test_emails:
        prediction = classifier.predict(email)
        print(f"\nEmail: {email}")
        print(f"Predicted topic: {prediction}")

if __name__ == "__main__":
    main() 