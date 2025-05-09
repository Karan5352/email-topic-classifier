from email_classifier import EmailClassifier

def main():
    # Initialize the classifier
    classifier = EmailClassifier()
    
    # Load the trained model (make sure to run train.py first)
    try:
        classifier.load_model('model.pkl')
    except FileNotFoundError:
        print("Model file not found. Please run train.py first to train the model.")
        return

    # Example emails to classify
    example_emails = [
        "Hi team, let's have a quick meeting tomorrow at 10 AM to discuss the project progress.",
        "Your monthly subscription payment of $29.99 has been processed successfully.",
        "Important: Please update your password as we've detected suspicious activity on your account.",
        "The new software version 2.0 is now available for download.",
        "Your order #54321 has been shipped and will arrive in 2-3 business days."
    ]

    print("Classifying example emails:\n")
    for email in example_emails:
        prediction = classifier.predict(email)
        probabilities = classifier.predict_proba(email)
        print(f"Email: {email}")
        print(f"Predicted topic: {prediction}")
        print("-" * 80)

if __name__ == "__main__":
    main() 