"""Custom exception handling with detailed error messages and logging."""

import sys
import types

from xray_classifier.logger import get_logger

logger = get_logger(__name__)


def error_message_detail(error: str | Exception, error_detail: types.ModuleType) -> str:
    """Construct a detailed error message.

    Args:
        error: The exception object or message string.
        error_detail: The sys module to extract exception details.

    Returns:
        A formatted error message string.
    """
    _, _, exc_tb = error_detail.exc_info()
    if exc_tb is None:
        return f"Error occurred: {error!s}"
    file_name = exc_tb.tb_frame.f_code.co_filename
    return (
        f"Error occurred in python script name [{file_name}] "
        f"line number [{exc_tb.tb_lineno}] error message [{error!s}]"
    )


class CustomException(Exception):
    """Application exception with detailed context and automatic logging."""

    def __init__(
        self,
        error_message: str | Exception,
        error_detail: types.ModuleType | None = None,
    ):
        """Initialize the CustomException with a detailed error message.

        Args:
            error_message: The error message or original exception.
            error_detail: The sys module to extract exception details.
        """
        super().__init__(str(error_message))
        if error_detail is not None:
            self.error_message: str = error_message_detail(error_message, error_detail=error_detail)
        else:
            self.error_message = str(error_message)
        logger.error(self.error_message)

    def __str__(self) -> str:
        """Return the error message string representation."""
        return self.error_message


if __name__ == "__main__":
    from xray_classifier.logger import configure_logging

    configure_logging()
    try:
        a = 1 / 0
    except Exception as e:
        logger.info("Exception Testing Successful - Divide by Zero")
        raise CustomException(str(e), sys) from e
