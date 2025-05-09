import os
import imaplib
import email
from email.header import decode_header
import datetime

def connect_to_gmail(email_address, password):
    """Connect to Gmail using IMAP."""
    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    mail.login(email_address, password)
    return mail

def save_email(msg, output_dir, index):
    """Save email as .eml file."""
    filename = f"email_{index}.eml"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(str(msg))
    
    return filename

def export_emails(email_address, password, output_dir, days=30):
    """Export emails from Gmail to .eml files."""
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Connect to Gmail
    mail = connect_to_gmail(email_address, password)
    mail.select("inbox")
    
    # Calculate date
    date = (datetime.date.today() - datetime.timedelta(days=days)).strftime("%d-%b-%Y")
    
    # Search for emails
    _, messages = mail.search(None, f'(SINCE {date})')
    
    # Process each email
    for i, message_number in enumerate(messages[0].split()):
        _, msg_data = mail.fetch(message_number, "(RFC822)")
        email_body = msg_data[0][1]
        email_message = email.message_from_bytes(email_body)
        
        # Save email
        filename = save_email(email_message, output_dir, i)
        print(f"Saved {filename}")
    
    mail.close()
    mail.logout()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Export emails from Gmail')
    parser.add_argument('--email', required=True, help='Gmail address')
    parser.add_argument('--password', required=True, help='Gmail password or app password')
    parser.add_argument('--output', required=True, help='Directory to save emails')
    parser.add_argument('--days', type=int, default=30, help='Number of days to look back')
    args = parser.parse_args()

    export_emails(args.email, args.password, args.output, args.days)

if __name__ == "__main__":
    main() 