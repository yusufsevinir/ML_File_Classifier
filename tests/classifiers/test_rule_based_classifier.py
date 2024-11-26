import pytest
from io import BytesIO
from werkzeug.datastructures import FileStorage
from src.classifiers.rule_based_classifier import RuleBasedClassifier

@pytest.fixture
def classifier():
    return RuleBasedClassifier()

@pytest.fixture
def create_file():
    def _create_file(content: str, filename: str = "test.txt"):
        return FileStorage(
            stream=BytesIO(content.encode()),
            filename=filename,
            content_type="text/plain"
        )
    return _create_file

def test_classify_invoice(classifier, create_file):
    # Test various invoice-related content
    test_cases = [
        "This is an invoice for services rendered",
        "Total amount due: $500",
        "Bill to: John Doe",
        "Invoice date: 2024-03-20"
    ]
    
    for content in test_cases:
        file = create_file(content)
        assert classifier.classify(file) == "invoice"

def test_classify_bank_statement(classifier, create_file):
    test_cases = [
        "Monthly bank statement",
        "Account balance: $1000",
        "Your bank transactions for March 2024",
        "Current balance as of"
    ]
    
    for content in test_cases:
        file = create_file(content)
        assert classifier.classify(file) == "bank_statement"

def test_classify_drivers_license(classifier, create_file):
    test_cases = [
        "Driver's license number: XYZ123",
        "State drivers license",
        "Official ID document",
        "Driver licence renewal"
    ]
    
    for content in test_cases:
        file = create_file(content)
        assert classifier.classify(file) == "drivers_license"

def test_classify_unknown(classifier, create_file):
    test_cases = [
        "Random document content",
        "This is some text without any keywords",
        ""  # Empty document
    ]
    
    for content in test_cases:
        file = create_file(content)
        assert classifier.classify(file) == "unknown file"

def test_case_insensitive_classification(classifier, create_file):
    # Test that classification works regardless of case
    test_cases = [
        ("INVOICE TO: JOHN DOE", "invoice"),
        ("Bank STATEMENT", "bank_statement"),
        ("DRIVER'S LICENSE", "drivers_license")
    ]
    
    for content, expected in test_cases:
        file = create_file(content)
        assert classifier.classify(file) == expected 