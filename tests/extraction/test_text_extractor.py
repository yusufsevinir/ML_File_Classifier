import pytest
from werkzeug.datastructures import FileStorage
from io import BytesIO
import pypdf
from PIL import Image
from src.extraction.text_extractor import extract_text

@pytest.fixture
def mock_pdf_file():
    # Create a simple PDF file in memory
    buffer = BytesIO()
    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(width=612, height=792)  # Standard letter size
    
    # Create a new PDF with content
    writer.write(buffer)
    buffer.seek(0)
    
    return FileStorage(
        stream=buffer,
        filename="test.pdf",
        content_type="application/pdf"
    )

@pytest.fixture
def mock_image_file():
    # Create a simple image with text
    img = Image.new('RGB', (100, 30), color='white')
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return FileStorage(
        stream=img_byte_arr,
        filename="test.png",
        content_type="image/png"
    )

@pytest.fixture
def mock_text_file():
    text_content = BytesIO(b"Hello, this is a test file.")
    return FileStorage(
        stream=text_content,
        filename="test.txt",
        content_type="text/plain"
    )

def test_extract_text_from_pdf(mock_pdf_file):
    result = extract_text(mock_pdf_file)
    assert isinstance(result, str)
    # Since we can't easily add text with pypdf, we just check if we get a string back
    # and it's empty (since our PDF is blank)
    assert result == ""

def test_extract_text_from_image(mock_image_file):
    result = extract_text(mock_image_file)
    assert isinstance(result, str)
    # Note: OCR results might vary, so we just check if we get a string back

def test_extract_text_from_text_file(mock_text_file):
    result = extract_text(mock_text_file)
    assert isinstance(result, str)
    assert "Hello, this is a test file." in result

def test_unsupported_mime_type():
    unsupported_file = FileStorage(
        stream=BytesIO(b"some content"),
        filename="test.unknown",
        content_type="application/unknown"
    )
    result = extract_text(unsupported_file)
    assert result == ""

def test_handle_empty_file():
    empty_file = FileStorage(
        stream=BytesIO(b""),
        filename="empty.txt",
        content_type="text/plain"
    )
    result = extract_text(empty_file)
    assert isinstance(result, str)
    assert result == ""

@pytest.mark.parametrize("error_file,mime_type", [
    (BytesIO(b"corrupted content"), "application/pdf"),
    (BytesIO(b"corrupted content"), "image/png"),
    (BytesIO(b"corrupted content"), "text/rtf"),
])
def test_handle_corrupted_files(error_file, mime_type):
    corrupted_file = FileStorage(
        stream=error_file,
        filename="corrupted.file",
        content_type=mime_type
    )
    result = extract_text(corrupted_file)
    assert result == "" 