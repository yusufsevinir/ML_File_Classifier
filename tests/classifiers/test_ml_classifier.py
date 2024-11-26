import pytest
from unittest.mock import Mock, patch
import torch
from src.classifiers.ml_classifier import MLClassifier
from werkzeug.datastructures import FileStorage

@pytest.fixture
def mock_model():
    model = Mock()
    model.config.id2label = {
        0: "bank_statement",
        1: "invoice",
        2: "drivers_license"
    }
    return model

@pytest.fixture
def mock_tokenizer():
    tokenizer = Mock()
    # Mock tokenizer output
    tokenizer.return_value = {
        'input_ids': torch.tensor([[1, 2, 3]]),
        'attention_mask': torch.tensor([[1, 1, 1]])
    }
    return tokenizer

@pytest.fixture
def classifier(mock_model, mock_tokenizer):
    with patch('src.classifiers.ml_classifier.AutoModelForSequenceClassification') as mock_model_cls, \
         patch('src.classifiers.ml_classifier.AutoTokenizer') as mock_tokenizer_cls:
        
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        
        return MLClassifier()

def test_initialization(classifier):
    assert classifier.model is not None
    assert classifier.tokenizer is not None
    assert classifier.device in [torch.device('cuda'), torch.device('cpu')]

def test_classify_success(classifier, mock_model):
    # Mock the model output
    mock_output = Mock()
    mock_output.logits = torch.tensor([[0.7, 0.2, 0.1]])
    mock_model.return_value = mock_output

    # Create a mock file
    mock_file = Mock(spec=FileStorage)
    
    with patch('src.classifiers.ml_classifier.extract_text') as mock_extract, \
         patch('src.classifiers.ml_classifier.preprocess_text') as mock_preprocess:
        
        mock_extract.return_value = "Sample text"
        mock_preprocess.return_value = "Processed sample text"
        
        result = classifier.classify(mock_file)
        
        assert isinstance(result, dict)
        assert "category" in result
        assert "confidence" in result
        assert "debug" in result
        assert isinstance(result["debug"], dict)
        assert len(result["debug"]["all_predictions"]) <= 3

def test_classify_handles_long_input(classifier, mock_model, mock_tokenizer):
    mock_output = Mock()
    mock_output.logits = torch.tensor([[0.7, 0.2, 0.1]])
    mock_model.return_value = mock_output
    
    # Mock tokenizer to return max length tokens
    mock_tokenizer.return_value = {
        'input_ids': torch.tensor([[1] * 512]),
        'attention_mask': torch.tensor([[1] * 512])
    }
    
    mock_file = Mock(spec=FileStorage)
    
    with patch('src.classifiers.ml_classifier.extract_text') as mock_extract, \
         patch('src.classifiers.ml_classifier.preprocess_text') as mock_preprocess:
        
        mock_extract.return_value = "Long text" * 1000
        mock_preprocess.return_value = "Processed " + ("long text " * 1000)
        
        result = classifier.classify(mock_file)
        
        assert result["debug"]["token_length"] == 512

def test_classify_handles_empty_input(classifier, mock_model):
    mock_output = Mock()
    mock_output.logits = torch.tensor([[0.7, 0.2, 0.1]])
    mock_model.return_value = mock_output
    
    mock_file = Mock(spec=FileStorage)
    
    with patch('src.classifiers.ml_classifier.extract_text') as mock_extract, \
         patch('src.classifiers.ml_classifier.preprocess_text') as mock_preprocess:
        
        mock_extract.return_value = ""
        mock_preprocess.return_value = ""
        
        result = classifier.classify(mock_file)
        
        assert isinstance(result, dict)
        assert result["debug"]["text_length"] == 0

def test_classify_returns_top_predictions(classifier, mock_model):
    mock_output = Mock()
    # Create logits that will give clear top 3 predictions
    mock_output.logits = torch.tensor([[5.0, 3.0, 1.0]])
    mock_model.return_value = mock_output
    
    mock_file = Mock(spec=FileStorage)
    
    with patch('src.classifiers.ml_classifier.extract_text') as mock_extract, \
         patch('src.classifiers.ml_classifier.preprocess_text') as mock_preprocess:
        
        mock_extract.return_value = "Sample text"
        mock_preprocess.return_value = "Processed sample text"
        
        result = classifier.classify(mock_file)
        
        predictions = result["debug"]["all_predictions"]
        assert len(predictions) == 3
        # Verify predictions are in descending order of confidence
        confidences = [float(p["confidence"]) for p in predictions]
        assert confidences == sorted(confidences, reverse=True) 

def test_classify_bank_statement(classifier, mock_model):
    mock_output = Mock()
    # Simulate high confidence for bank statement
    mock_output.logits = torch.tensor([[4.5, 1.2, 0.8]])  # High score for bank statement
    mock_model.return_value = mock_output
    
    mock_file = Mock(spec=FileStorage)
    
    with patch('src.classifiers.ml_classifier.extract_text') as mock_extract, \
         patch('src.classifiers.ml_classifier.preprocess_text') as mock_preprocess:
        
        # Simulate typical bank statement text
        mock_extract.return_value = """
        MONTHLY STATEMENT
        Account Number: XXXX-XXXX-1234
        Statement Period: 01/01/2024 - 31/01/2024
        
        TRANSACTION DETAILS
        01/01/2024    Opening Balance             $1,234.56
        02/01/2024    GROCERY STORE              -$52.10
        03/01/2024    SALARY DEPOSIT             $2,500.00
        """
        mock_preprocess.return_value = mock_extract.return_value
        
        result = classifier.classify(mock_file)
        
        assert result["category"] == "bank_statement"
        assert float(result["confidence"]) > 0.8
        assert "MONTHLY STATEMENT" in mock_extract.return_value

def test_classify_invoice(classifier, mock_model):
    mock_output = Mock()
    mock_output.logits = torch.tensor([[1.2, 4.8, 0.5]])  # High score for invoice
    mock_model.return_value = mock_output
    
    mock_file = Mock(spec=FileStorage)
    
    with patch('src.classifiers.ml_classifier.extract_text') as mock_extract, \
         patch('src.classifiers.ml_classifier.preprocess_text') as mock_preprocess:
        
        # Simulate typical invoice text
        mock_extract.return_value = """
        INVOICE
        Invoice #: INV-2024-001
        Date: January 15, 2024
        
        Bill To:
        John Doe
        123 Main Street
        
        Items:
        1. Consulting Services    $500.00
        2. Software License       $299.99
        
        Total Amount Due: $799.99
        """
        mock_preprocess.return_value = mock_extract.return_value
        
        result = classifier.classify(mock_file)
        
        assert result["category"] == "invoice"
        assert float(result["confidence"]) > 0.8
        assert "INVOICE" in mock_extract.return_value

def test_classify_drivers_license(classifier, mock_model):
    mock_output = Mock()
    mock_output.logits = torch.tensor([[0.8, 1.1, 4.6]])  # High score for driver's license
    mock_model.return_value = mock_output
    
    mock_file = Mock(spec=FileStorage)
    
    with patch('src.classifiers.ml_classifier.extract_text') as mock_extract, \
         patch('src.classifiers.ml_classifier.preprocess_text') as mock_preprocess:
        
        # Simulate typical driver's license text
        mock_extract.return_value = """
        DRIVER LICENSE
        DL NO: D1234-5678-9012
        DOB: 01/15/1990
        ISS: 01/01/2020
        EXP: 01/01/2028
        
        SMITH
        JOHN ROBERT
        123 MAIN STREET
        ANYTOWN, ST 12345
        """
        mock_preprocess.return_value = mock_extract.return_value
        
        result = classifier.classify(mock_file)
        
        assert result["category"] == "drivers_license"
        assert float(result["confidence"]) > 0.8
        assert "DRIVER LICENSE" in mock_extract.return_value

def test_mixed_document_features(classifier, mock_model):
    """Test handling of documents with mixed features"""
    mock_output = Mock()
    # Even with mixed features, model should still make a clear decision
    mock_output.logits = torch.tensor([[4.2, 2.1, 1.5]])
    mock_model.return_value = mock_output
    
    mock_file = Mock(spec=FileStorage)
    
    with patch('src.classifiers.ml_classifier.extract_text') as mock_extract, \
         patch('src.classifiers.ml_classifier.preprocess_text') as mock_preprocess:
        
        # Text containing features from multiple document types
        mock_extract.return_value = """
        MONTHLY STATEMENT
        Account Number: XXXX-XXXX-1234
        
        INVOICE #: 12345
        Amount Due: $500.00
        
        License Number: DL-123456
        """
        mock_preprocess.return_value = mock_extract.return_value
        
        result = classifier.classify(mock_file)
        
        # Verify that a clear decision is made despite mixed features
        assert float(result["confidence"]) > 0.7
        assert len(result["debug"]["all_predictions"]) == 3 