import re

def preprocess_text(text: str) -> str:
    """
    Basic text preprocessing.
    """
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Extract alphabetic characters from alphanumeric strings
    text = ' '.join(''.join(c for c in word if c.isalpha()) 
                   for word in text.split())
    # Remove any remaining special characters
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text