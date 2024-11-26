from .base_classifier import BaseClassifier
from src.extraction.text_extractor import extract_text
from werkzeug.datastructures import FileStorage
import logging

logger = logging.getLogger(__name__)

class RuleBasedClassifier(BaseClassifier):
    def classify(self, file: FileStorage) -> str:
        text = extract_text(file).lower()
        logger.debug(f"Extracted text from {file.filename}: {text}") 
        
        # Check for invoice keywords first
        invoice_keywords = ['invoice', 'bill', 'amount due', 'total', 'invoice date', 'invoice to']
        if any(keyword in text for keyword in invoice_keywords):
            logger.debug(f"Found 'invoice' related keywords in {file.filename}")
            return "invoice"
        
        # Check for bank statement keywords
        bank_keywords = ['bank statement', 'bank', 'account', 'balance']
        if any(keyword in text for keyword in bank_keywords):
            logger.debug(f"Found bank statement keywords in {file.filename}")
            return "bank_statement"
        
        # Check for driver's license keywords
        license_keywords = ["driver's license", "drivers license", "license", "licence", "id"]
        if any(keyword in text for keyword in license_keywords):
            logger.debug(f"Found driver's license keywords in {file.filename}")
            return "drivers_license"
        
        logger.warning(f"Could not classify {file.filename}")
        return "unknown file"