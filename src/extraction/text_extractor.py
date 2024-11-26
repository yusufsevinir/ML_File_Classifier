from werkzeug.datastructures import FileStorage
import io
import pypdf
from PIL import Image
import pytesseract
import docx2txt
import openpyxl
import striprtf
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to see all messages
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def extract_text(file: FileStorage) -> str:
    """
    Extract text from the uploaded file.
    """
    try:
        mime_type = file.content_type
        file.seek(0)
        logger.debug(f"Extracting text from file: {file.filename} (mime type: {mime_type})")

        if mime_type == 'application/pdf':
            logger.debug("Processing PDF file")
            pdf_reader = pypdf.PdfReader(file)
            text = ''
            for i, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text() or ''
                text += page_text
                logger.debug(f"Page {i+1} extracted text length: {len(page_text)}")
            logger.debug(f"Total extracted text length: {len(text)}")
            return text

        elif mime_type in ['image/png', 'image/jpeg']:
            logger.debug("Processing image file")
            try:
                image = Image.open(file.stream)
                text = pytesseract.image_to_string(image)
                logger.debug(f"OCR extracted text length: {len(text), text}")
                return text
            except Exception as e:
                logger.error(f"OCR error: {str(e)}")
                raise

        elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            logger.debug("Processing Word document")
            text = docx2txt.process(file)
            logger.debug(f"Extracted text length: {len(text)}")
            return text

        elif mime_type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
            logger.debug("Processing Excel file")
            wb = openpyxl.load_workbook(file)
            text = ''
            for sheet in wb:
                for row in sheet.iter_rows(values_only=True):
                    text += ' '.join([str(cell) for cell in row if cell is not None]) + '\n'
            logger.debug(f"Extracted text length: {len(text)}")
            return text

        elif mime_type == 'text/rtf':
            logger.debug("Processing RTF file")
            text = striprtf.rtf_to_text(file.read().decode('utf-8', errors='ignore'))
            logger.debug(f"Extracted text length: {len(text)}")
            return text

        elif mime_type in ['text/plain', 'text/csv']:
            logger.debug("Processing plain text file")
            text = file.read().decode('utf-8', errors='ignore')
            logger.debug(f"Extracted text length: {len(text)}")
            return text

        else:
            logger.warning(f"Unsupported mime type: {mime_type}")
            return ''
    except Exception as e:
        logger.error(f"Error extracting text from {file.filename}: {str(e)}", exc_info=True)
        return ''