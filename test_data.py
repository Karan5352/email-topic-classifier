"""
Test data for evaluating the email topic classifier.
This file contains realistic email examples with some ambiguity between categories.
"""

TEST_EMAILS = [
    # Ambiguous cases that could belong to multiple categories
    ("Join our exclusive webinar on career development and get a 50% discount on our premium membership. Limited time offer for our valued members.", "promotions"),  # Could be work/promotions
    ("Your job application has been received. As a special welcome offer, you'll get 20% off on our professional development courses.", "work"),  # Could be work/promotions
    ("Important: Your account security update and new features announcement. Please review the changes and update your preferences.", "updates"),  # Could be updates/security
    ("Your package has been delivered! Don't forget to check out our weekend sale with up to 70% off on new arrivals.", "shipping"),  # Could be shipping/promotions
    ("Conference registration is now open! Early bird tickets available with 30% discount for the first 100 registrants.", "events"),  # Could be events/promotions
    
    # Longer, more realistic emails
    ("Dear valued customer,\n\nWe are pleased to inform you that your recent order #12345 has been successfully processed and is now being prepared for shipment. You can track your package using the tracking number provided below.\n\nAs a token of our appreciation, we're offering you an exclusive 15% discount on your next purchase. This offer is valid for the next 7 days.\n\nBest regards,\nCustomer Service Team", "shipping"),
    
    ("Hello,\n\nThank you for your interest in the Senior Software Engineer position at our company. We have received your application and are impressed with your background.\n\nWe would like to invite you for a technical interview next week. The interview will include a coding assessment and a discussion about your previous projects.\n\nPlease find attached the interview preparation guide and company information packet.\n\nBest regards,\nHR Department", "work"),
    
    ("Important Update: System Maintenance\n\nDear users,\n\nWe will be performing scheduled maintenance on our platform this weekend. During this time, some features may be temporarily unavailable. We apologize for any inconvenience this may cause.\n\nMaintenance Schedule:\n- Start: Saturday, 2:00 AM EST\n- End: Sunday, 6:00 AM EST\n\nPlease ensure you save any important work before the maintenance period.\n\nThank you for your understanding.\nSystem Administration Team", "updates"),
    
    ("Exclusive Offer Inside!\n\nDear valued customer,\n\nWe're excited to announce our biggest sale of the year! Enjoy up to 70% off on selected items, plus free shipping on orders over $50.\n\nSpecial deals include:\n- Electronics: 50% off\n- Home goods: 40% off\n- Fashion: 30% off\n\nUse code SUMMER2024 at checkout.\n\nThis offer is valid until Sunday midnight.\n\nHappy shopping!\nThe Store Team", "promotions"),
    
    ("Security Alert: Unusual Login Activity\n\nDear account holder,\n\nWe detected a login attempt from an unrecognized device. If this was you, please verify your account. If not, please secure your account immediately.\n\nTo protect your account:\n1. Change your password\n2. Enable two-factor authentication\n3. Review recent activity\n\nIf you need assistance, our support team is available 24/7.\n\nBest regards,\nSecurity Team", "support"),
    
    # Mixed category examples
    ("Your flight has been delayed. As compensation, we're offering a 25% discount on your next booking. Plus, enjoy exclusive access to our airport lounge.", "travel"),  # travel/promotions
    
    ("Join our professional networking event! Early registration includes a free workshop on career development and a 20% discount on premium membership.", "events"),  # events/work/promotions
    
    ("Your investment portfolio has been updated. New opportunities available with special rates for existing customers. Limited time offer on selected funds.", "finance"),  # finance/promotions
    
    ("Technical support update: New features added to improve your experience. Try them now and get 50 bonus points in our rewards program.", "support"),  # support/promotions
    
    ("Your social media account has been compromised. Please update your security settings immediately. We're offering a free security audit to all affected users.", "social"),  # social/support
    
    # Real-world style emails
    ("Re: Your Recent Application\n\nDear Applicant,\n\nThank you for your interest in our company. We have reviewed your application for the Data Scientist position and would like to proceed with the next steps.\n\nPlease complete the technical assessment within the next 48 hours. The assessment will include:\n- Data analysis problems\n- Machine learning concepts\n- Coding challenges\n\nAfter reviewing your assessment, we will schedule a technical interview with our team.\n\nBest regards,\nRecruitment Team", "work"),
    
    ("Your Order Status Update\n\nDear Customer,\n\nWe're writing to inform you about your recent order #98765.\n\nYour package is currently in transit and scheduled for delivery tomorrow between 2:00 PM and 5:00 PM. You can track your shipment using the link below.\n\nAs a valued customer, we're offering you an exclusive 15% discount on your next purchase. Use code THANKYOU15 at checkout.\n\nIf you have any questions, our support team is available 24/7.\n\nBest regards,\nShipping Department", "shipping"),
    
    ("Important: Account Security Update\n\nDear User,\n\nWe're implementing new security measures to protect your account. Please review and update your security settings by following the link below.\n\nNew features include:\n- Two-factor authentication\n- Login alerts\n- Device management\n\nFor a limited time, we're offering a free security audit to all users who complete their security update.\n\nIf you need assistance, contact our support team.\n\nBest regards,\nSecurity Team", "updates"),
    
    ("Special Weekend Offer\n\nDear Valued Customer,\n\nWe're excited to announce our weekend flash sale! Enjoy incredible discounts on our most popular items:\n\n- Electronics: Up to 50% off\n- Home & Kitchen: 40% off\n- Fashion: 30% off\n\nPlus, get free shipping on all orders over $75.\n\nThis offer is valid until Sunday midnight. Don't miss out!\n\nHappy shopping!\nThe Store Team", "promotions"),
    
    ("Your Investment Portfolio Update\n\nDear Investor,\n\nYour portfolio has been updated with the latest market data. We've identified some new investment opportunities that match your risk profile.\n\nKey updates:\n- Market performance summary\n- New investment recommendations\n- Portfolio rebalancing suggestions\n\nAs a premium member, you have exclusive access to our new investment tools. Try them now and earn 100 bonus points.\n\nBest regards,\nInvestment Team", "finance")
] 