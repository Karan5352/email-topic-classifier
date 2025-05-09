import os
import email
from email_topic_classifier import EmailTopicClassifier
from datetime import datetime
import shutil
import html
from email.header import decode_header
import chardet

class EmailProcessor:
    def __init__(self, model_path="model.pkl"):
        self.classifier = EmailTopicClassifier()
        if os.path.exists(model_path):
            self.classifier.load(model_path)
        else:
            print("Model not found. Please train the model first using email_topic_classifier.py --train")
            return

    def decode_email_header(self, header):
        """Decode email header properly."""
        decoded_header = ""
        for part, encoding in decode_header(header):
            if isinstance(part, bytes):
                if encoding:
                    decoded_header += part.decode(encoding)
                else:
                    # Try to detect encoding
                    detected = chardet.detect(part)
                    decoded_header += part.decode(detected['encoding'] or 'utf-8', errors='replace')
            else:
                decoded_header += part
        return decoded_header

    def extract_email_content(self, email_path):
        """Extract text content from an email file."""
        try:
            with open(email_path, 'rb') as f:  # Open in binary mode
                msg = email.message_from_binary_file(f)
                
            # Get subject
            subject = self.decode_email_header(msg.get('subject', ''))
            
            # Get body
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        try:
                            charset = part.get_content_charset() or 'utf-8'
                            body = part.get_payload(decode=True).decode(charset, errors='replace')
                            break
                        except:
                            continue
            else:
                try:
                    charset = msg.get_content_charset() or 'utf-8'
                    body = msg.get_payload(decode=True).decode(charset, errors='replace')
                except:
                    body = "Could not decode email body"
            
            return {
                'subject': subject,
                'body': body,
                'from': self.decode_email_header(msg.get('from', '')),
                'date': msg.get('date', ''),
                'path': email_path
            }
        except Exception as e:
            print(f"Error processing {email_path}: {str(e)}")
            return None

    def process_emails(self, input_dir, output_dir):
        """Process all emails in a directory and organize them by topic."""
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Dictionary to store emails by topic
        emails_by_topic = {}

        # Process each email file
        for filename in os.listdir(input_dir):
            if filename.endswith('.eml'):  # Process .eml files
                email_path = os.path.join(input_dir, filename)
                email_data = self.extract_email_content(email_path)
                
                if email_data:
                    # Get topic prediction
                    topic = self.classifier.predict(email_data['body'])
                    
                    # Create topic directory if it doesn't exist
                    topic_dir = os.path.join(output_dir, topic)
                    if not os.path.exists(topic_dir):
                        os.makedirs(topic_dir)
                    
                    # Copy email to topic directory
                    shutil.copy2(email_path, os.path.join(topic_dir, filename))
                    
                    # Store email data
                    if topic not in emails_by_topic:
                        emails_by_topic[topic] = []
                    emails_by_topic[topic].append(email_data)
                    
                    print(f"Processed {filename} -> {topic}")

        # Generate HTML report
        self.generate_html_report(emails_by_topic, output_dir)

    def generate_html_report(self, emails_by_topic, output_dir):
        """Generate HTML report with email tables."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Email Topics Report</title>
            <style>
                body { 
                    font-family: Arial, sans-serif; 
                    margin: 20px;
                    background-color: #f5f5f5;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }
                .topic-nav {
                    background: white;
                    padding: 15px;
                    border-radius: 8px;
                    margin-bottom: 20px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .topic-nav h2 {
                    margin-top: 0;
                    color: #2c3e50;
                }
                .topic-buttons {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 10px;
                }
                .topic-button {
                    padding: 8px 16px;
                    background: #e0e0e0;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    transition: background 0.3s;
                }
                .topic-button:hover {
                    background: #d0d0d0;
                }
                .topic-button.active {
                    background: #2c3e50;
                    color: white;
                }
                .topic-section { 
                    display: none;
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    margin-bottom: 20px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .topic-section.active {
                    display: block;
                }
                h2 { 
                    color: #2c3e50;
                    margin-top: 0;
                }
                table { 
                    border-collapse: collapse; 
                    width: 100%;
                    margin-top: 15px;
                }
                th, td { 
                    border: 1px solid #ddd; 
                    padding: 12px 8px; 
                    text-align: left;
                }
                th { 
                    background-color: #f8f9fa;
                    font-weight: 600;
                }
                tr:nth-child(even) { 
                    background-color: #f8f9fa;
                }
                tr:hover {
                    background-color: #f0f0f0;
                }
                .email-count {
                    color: #666;
                    font-size: 0.9em;
                    margin-left: 10px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Email Topics Report</h1>
                
                <div class="topic-nav">
                    <h2>Topics</h2>
                    <div class="topic-buttons">
        """

        # Add topic buttons
        for topic in emails_by_topic.keys():
            count = len(emails_by_topic[topic])
            html_content += f"""
                        <button class="topic-button" onclick="showTopic('{topic}')">
                            {topic.title()} ({count})
                        </button>
            """

        html_content += """
                    </div>
                </div>
        """

        # Add topic sections
        for topic, emails in emails_by_topic.items():
            html_content += f"""
                <div id="{topic}" class="topic-section">
                    <h2>{topic.title()} <span class="email-count">({len(emails)} emails)</span></h2>
                    <table>
                        <tr>
                            <th>From</th>
                            <th>Subject</th>
                            <th>Date</th>
                        </tr>
            """
            
            for email_data in emails:
                html_content += f"""
                        <tr>
                            <td>{html.escape(email_data['from'])}</td>
                            <td>{html.escape(email_data['subject'])}</td>
                            <td>{html.escape(email_data['date'])}</td>
                        </tr>
                """
            
            html_content += """
                    </table>
                </div>
            """

        # Add JavaScript for topic navigation
        html_content += """
                <script>
                    function showTopic(topic) {
                        // Hide all sections
                        document.querySelectorAll('.topic-section').forEach(section => {
                            section.classList.remove('active');
                        });
                        
                        // Remove active class from all buttons
                        document.querySelectorAll('.topic-button').forEach(button => {
                            button.classList.remove('active');
                        });
                        
                        // Show selected section
                        document.getElementById(topic).classList.add('active');
                        
                        // Add active class to clicked button
                        event.target.classList.add('active');
                    }
                    
                    // Show first topic by default
                    document.addEventListener('DOMContentLoaded', function() {
                        const firstTopic = document.querySelector('.topic-button');
                        if (firstTopic) {
                            firstTopic.click();
                        }
                    });
                </script>
            </div>
        </body>
        </html>
        """

        # Save HTML report
        with open(os.path.join(output_dir, 'email_report.html'), 'w', encoding='utf-8') as f:
            f.write(html_content)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Process and organize emails by topic')
    parser.add_argument('--input', required=True, help='Directory containing email files (.eml)')
    parser.add_argument('--output', required=True, help='Directory to store organized emails')
    args = parser.parse_args()

    processor = EmailProcessor()
    processor.process_emails(args.input, args.output)

if __name__ == "__main__":
    main() 