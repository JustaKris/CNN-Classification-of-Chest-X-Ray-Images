"""Unit tests for src.utils.utils module."""

from xray_classifier.utils.utils import get_date, get_time


class TestDateTimeHelpers:
    """Verify date and time utility functions."""

    def test_get_date_format(self):
        """get_date should return a string in YYYY.MM.DD format."""
        date_str = get_date()
        assert isinstance(date_str, str)
        assert len(date_str) == 10
        assert date_str[4] == "."
        assert date_str[7] == "."

    def test_get_time_format(self):
        """get_time should return a string in HH-MM format."""
        time_str = get_time()
        assert isinstance(time_str, str)
        assert len(time_str) == 5
        assert time_str[2] == "-"
