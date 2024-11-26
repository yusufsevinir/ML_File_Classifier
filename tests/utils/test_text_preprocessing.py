import pytest
from src.utils.text_preprocessing import preprocess_text

def test_basic_text_cleaning():
    input_text = "Hello World! This is a Test."
    expected = "hello world this is a test"
    assert preprocess_text(input_text) == expected

def test_url_removal():
    input_text = "Check this link https://example.com and http://test.org/page"
    expected = "check this link and"
    assert preprocess_text(input_text) == expected

def test_special_characters():
    input_text = "Hello! @#$%^&* World?"
    expected = "hello world"
    assert preprocess_text(input_text) == expected

def test_numbers_removal():
    input_text = "Room123 costs 500$ in Building42"
    expected = "room costs in building"
    assert preprocess_text(input_text) == expected

def test_multiple_spaces():
    input_text = "Too    many     spaces    here"
    expected = "too many spaces here"
    assert preprocess_text(input_text) == expected

def test_empty_string():
    input_text = ""
    expected = ""
    assert preprocess_text(input_text) == expected

def test_only_special_characters():
    input_text = "@#$%^&* 123456"
    expected = ""
    assert preprocess_text(input_text) == expected

def test_mixed_case_handling():
    input_text = "MiXeD cAsE tExT"
    expected = "mixed case text"
    assert preprocess_text(input_text) == expected

def test_alphanumeric_extraction():
    input_text = "user123name pass456word"
    expected = "username password"
    assert preprocess_text(input_text) == expected

def test_leading_trailing_spaces():
    input_text = "   spaces   around   text   "
    expected = "spaces around text"
    assert preprocess_text(input_text) == expected

def test_newlines_and_tabs():
    input_text = "Text with\nnewlines\tand\ttabs"
    expected = "text with newlines and tabs"
    assert preprocess_text(input_text) == expected

def test_real_world_example():
    input_text = """
    INVOICE #123
    Date: 2024-01-15
    Amount: $500.00
    https://payment.link
    Contact: support@example.com
    """
    expected = "invoice date amount contact supportexamplecom"
    assert preprocess_text(input_text) == expected

def test_unicode_characters():
    input_text = "Café résumé naïve"
    expected = "caf rsum nave"  # Note: accents are removed
    assert preprocess_text(input_text) == expected

def test_punctuation_in_words():
    input_text = "don't can't won't it's"
    expected = "dont cant wont its"
    assert preprocess_text(input_text) == expected

def test_edge_cases():
    test_cases = [
        (None, pytest.raises(AttributeError)),  # Should raise error for None
        (123, pytest.raises(AttributeError)),   # Should raise error for non-string
        (" ", ""),                             # Single space should become empty
        (".", ""),                             # Single punctuation should become empty
        ("a" * 1000, "a" * 1000),             # Long string handling
    ]
    
    for input_text, expected in test_cases:
        if isinstance(expected, str):
            assert preprocess_text(input_text) == expected
        else:
            with expected:
                preprocess_text(input_text) 