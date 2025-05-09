"""
Test data for evaluating the email topic classifier.
This file contains both straightforward and ambiguous test cases.
"""

# Straightforward test cases - clear, single-topic emails
STRAIGHTFORWARD_TESTS = [
    # Work examples
    ("Your job application for Software Engineer position has been received and is under review. We will contact you within 5 business days.", "work"),
    ("Interview scheduled for tomorrow at 2 PM. Please bring your resume and portfolio.", "work"),
    ("Welcome to your new role! Your start date is confirmed for next Monday.", "work"),
    
    # Shipping examples
    ("Your package #12345 has been delivered to your doorstep. Please confirm receipt.", "shipping"),
    ("Order #67890 has been shipped and is expected to arrive in 3-5 business days.", "shipping"),
    ("Your return shipping label has been generated. Please use this for your return.", "shipping"),
    
    # Finance examples
    ("Your monthly bank statement is now available. Please log in to view.", "finance"),
    ("Payment of $500 has been successfully processed for invoice #123.", "finance"),
    ("Your credit card payment of $200 is due in 3 days.", "finance"),
    
    # Travel examples
    ("Your flight from New York to London has been confirmed for tomorrow.", "travel"),
    ("Hotel reservation at Grand Hotel confirmed for your stay next week.", "travel"),
    ("Your car rental at LAX has been processed. Pick up at Terminal 4.", "travel"),
    
    # Promotions examples
    ("Flash sale: 50% off all electronics this weekend only.", "promotions"),
    ("Special offer: Buy one get one free on all clothing items.", "promotions"),
    ("Limited time deal: 30% off your next purchase with code SUMMER30.", "promotions"),
    
    # Social examples
    ("New connection request from John Smith on LinkedIn.", "social"),
    ("Sarah has invited you to her birthday party next Saturday.", "social"),
    ("Your friend Mike commented on your recent post.", "social"),
    
    # Updates examples
    ("System maintenance scheduled for tonight at 2 AM EST.", "updates"),
    ("New software update available for your device.", "updates"),
    ("Your account settings have been updated successfully.", "updates"),
    
    # Support examples
    ("Your support ticket #456 has been resolved. Please confirm if you need anything else.", "support"),
    ("Technical support is available 24/7 for any issues you encounter.", "support"),
    ("Your recent issue has been fixed. Please try again and let us know if you need help.", "support"),
    
    # Spam examples
    ("You've won a million dollars! Click here to claim your prize.", "spam"),
    ("Urgent: Your account will be suspended unless you verify now.", "spam"),
    ("Congratulations! You've been selected for a special offer.", "spam"),
    
    # Events examples
    ("Conference registration is now open. Early bird tickets available.", "events"),
    ("Webinar on machine learning scheduled for next Thursday at 3 PM.", "events"),
    ("Your event ticket for Tech Summit 2024 has been confirmed.", "events")
]

# Ambiguous test cases - could belong to multiple categories
AMBIGUOUS_TESTS = [
    # Work/Promotions
    ("Join our exclusive webinar on career development and get a 50% discount on our premium membership. Limited time offer for our valued members.", "promotions"),
    ("Your job application has been received. As a special welcome offer, you'll get 20% off on our professional development courses.", "work"),
    
    # Updates/Security
    ("Important: Your account security update and new features announcement. Please review the changes and update your preferences.", "updates"),
    
    # Shipping/Promotions
    ("Your package has been delivered! Don't forget to check out our weekend sale with up to 70% off on new arrivals.", "shipping"),
    
    # Events/Promotions
    ("Conference registration is now open! Early bird tickets available with 30% discount for the first 100 registrants.", "events"),
    
    # Travel/Promotions
    ("Your flight has been delayed. As compensation, we're offering a 25% discount on your next booking. Plus, enjoy exclusive access to our airport lounge.", "travel"),
    
    # Events/Work/Promotions
    ("Join our professional networking event! Early registration includes a free workshop on career development and a 20% discount on premium membership.", "events"),
    
    # Finance/Promotions
    ("Your investment portfolio has been updated. New opportunities available with special rates for existing customers. Limited time offer on selected funds.", "finance"),
    
    # Support/Promotions
    ("Technical support update: New features added to improve your experience. Try them now and get 50 bonus points in our rewards program.", "support"),
    
    # Social/Support
    ("Your social media account has been compromised. Please update your security settings immediately. We're offering a free security audit to all affected users.", "social")
]

# Combined test set (for backward compatibility)
TEST_EMAILS = STRAIGHTFORWARD_TESTS + AMBIGUOUS_TESTS 