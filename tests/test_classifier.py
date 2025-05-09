import unittest
from email_topic_classifier import EmailTopicClassifier, SAMPLE_EMAILS

class TestEmailClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = EmailTopicClassifier()
        texts, labels = zip(*SAMPLE_EMAILS)
        self.classifier.train(texts, labels)

    def test_training(self):
        """Test if the model can be trained."""
        self.assertTrue(self.classifier.is_trained)

    def test_prediction(self):
        """Test if the model can make predictions."""
        test_email = "Let's schedule a meeting for tomorrow"
        prediction = self.classifier.predict(test_email)
        self.assertIsInstance(prediction, str)
        self.assertIn(prediction, [label for _, label in SAMPLE_EMAILS])

    def test_save_load(self):
        """Test if the model can be saved and loaded."""
        # Save the model
        self.classifier.save("test_model.pkl")
        
        # Create a new classifier and load the model
        new_classifier = EmailTopicClassifier()
        new_classifier.load("test_model.pkl")
        
        # Test if predictions match
        test_email = "Your payment has been processed"
        original_prediction = self.classifier.predict(test_email)
        loaded_prediction = new_classifier.predict(test_email)
        self.assertEqual(original_prediction, loaded_prediction)

if __name__ == '__main__':
    unittest.main() 