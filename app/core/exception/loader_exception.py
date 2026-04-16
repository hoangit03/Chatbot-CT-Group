from pathlib import Path
from typing import Optional


class LoaderException(Exception):
    """Base exception cho toàn bộ Document Loader module"""
    def __init__(self, message: str, file_path: Optional[Path] = None, original_error: Optional[Exception] = None):
        self.file_path = file_path
        self.original_error = original_error
        full_message = f"[Loader Error] {message}"
        if file_path:
            full_message += f" | File: {file_path}"
        if original_error:
            full_message += f" | Original: {type(original_error).__name__} - {original_error}"
        super().__init__(full_message)


class UnsupportedFileExtensionError(LoaderException):
    """Khi file có đuôi không được hỗ trợ"""
    def __init__(self, file_path: Path):
        super().__init__(
            message=f"Định dạng file không được hỗ trợ: {file_path.suffix}",
            file_path=file_path
        )


class DocumentLoadError(LoaderException):
    """Lỗi khi load nội dung file (PDF, DOCX, Excel, PPTX, MSG...)"""
    def __init__(self, file_path: Path, original_error: Exception):
        super().__init__(
            message=f"Không thể load nội dung file",
            file_path=file_path,
            original_error=original_error
        )


class InvalidFileError(LoaderException):
    """File bị hỏng, không tồn tại hoặc không đọc được"""
    def __init__(self, file_path: Path, original_error: Optional[Exception] = None):
        super().__init__(
            message="File không hợp lệ hoặc không thể đọc",
            file_path=file_path,
            original_error=original_error
        )