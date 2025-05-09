# Email Topic Classifier 📧

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

A powerful email topic classifier that uses machine learning to categorize emails into 10 distinct topics. Perfect for organizing your inbox and managing email workflows!

## ✨ Features

- 🎯 10 distinct email topics
- 🚀 High-accuracy classification
- 📊 Interactive HTML report
- 💾 Model persistence
- 🔧 Easy to extend and customize
- 📝 Well-documented code
- 🧪 Includes test suite

## 🎯 Supported Topics

The classifier can categorize emails into these topics:
- Work
- Shipping
- Finance
- Travel
- Promotions
- Social
- Updates
- Support
- Spam
- Events

## 🚀 Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/email-topic-classifier.git
   cd email-topic-classifier
   ```

2. **Set up the environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Train the model:**
   ```bash
   python email_topic_classifier.py --train
   ```

4. **Process your emails:**
   ```bash
   python email_processor.py --input "your_emails" --output "organized_emails"
   ```

5. **View the report:**
   Open `organized_emails/email_report.html` in your browser

## 📚 Usage Examples

### Basic Classification
```python
from email_topic_classifier import EmailTopicClassifier

# Initialize and train
classifier = EmailTopicClassifier()
classifier.train(texts, labels)

# Make predictions
topic = classifier.predict("Your payment has been processed")
print(f"Predicted topic: {topic}")
```

### Process Multiple Emails
```bash
# Export emails from Gmail
python export_gmail.py --email "your.email@gmail.com" --password "your-app-password" --output "emails"

# Process and organize them
python email_processor.py --input "emails" --output "organized_emails"
```

## 🔧 Customization

### Adding Custom Training Data
```python
custom_emails = [
    ("Your meeting is scheduled for tomorrow", "work"),
    ("Family dinner this weekend", "personal"),
    # Add more examples...
]

texts, labels = zip(*custom_emails)
classifier.train(texts, labels)
```

## 🧪 Running Tests

```bash
python -m unittest discover tests
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [scikit-learn](https://scikit-learn.org/)
- Inspired by real-world email organization needs
