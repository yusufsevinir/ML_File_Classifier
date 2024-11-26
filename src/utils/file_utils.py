from werkzeug.datastructures import FileStorage
import magic

ALLOWED_MIME_TYPES = [
    'application/pdf',
    'image/png',
    'image/jpeg',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',  # docx
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',        # xlsx
    'text/plain',
    'text/rtf',
    'text/csv',
]

def allowed_file(file: FileStorage) -> bool:
    file.seek(0)
    mime_type = magic.from_buffer(file.read(1024), mime=True)
    file.seek(0)
    return mime_type in ALLOWED_MIME_TYPES