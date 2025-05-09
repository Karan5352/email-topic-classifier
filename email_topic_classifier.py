import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample data
SAMPLE_EMAILS = [
    ("Let's meet tomorrow to discuss the project.", "meeting"),
    ("Your order has shipped and will arrive soon.", "shipping"),
    ("Please review the attached financial report.", "report"),
    ("Your payment was received.", "payment"),
    ("Security alert: new login detected.", "security"),
    ("Join us for a team event this Friday.", "event"),
    ("Your subscription will renew next month.", "subscription"),
    ("Thank you for your job application.", "job"),
    ("Software update available for your device.", "software"),
    ("Please complete the customer satisfaction survey.", "survey"),
]

class EmailTopicClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.model = MultinomialNB()
        self.is_trained = False

    def train(self, texts, labels):
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        self.is_trained = True

    def predict(self, text):
        X = self.vectorizer.transform([text])
        return self.model.predict(X)[0]

    def save(self, path="model.pkl"):
        with open(path, "wb") as f:
            pickle.dump((self.vectorizer, self.model), f)

    def load(self, path="model.pkl"):
        with open(path, "rb") as f:
            self.vectorizer, self.model = pickle.load(f)
        self.is_trained = True

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