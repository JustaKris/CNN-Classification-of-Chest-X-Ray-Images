"""Unit tests for src.exception module."""

from src.exception import CustomException


class TestCustomException:
    """Verify CustomException behavior."""

    def test_custom_exception_message(self):
        """CustomException should store the error message."""
        exc = CustomException("test error")
        assert str(exc) == "test error"

    def test_custom_exception_from_exception(self):
        """CustomException should accept an Exception as the message."""
        original = ValueError("original error")
        exc = CustomException(original)
        assert "original error" in str(exc)

    def test_custom_exception_is_exception(self):
        """CustomException should be a subclass of Exception."""
        exc = CustomException("test")
        assert isinstance(exc, Exception)
