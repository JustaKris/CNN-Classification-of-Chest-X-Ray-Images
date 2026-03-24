"""Unit tests for src.config module."""

from src.config import CLASSES, COLOR_MODE, IMAGE_SIZE


class TestConfig:
    """Verify configuration constants are correctly defined."""

    def test_classes_defined(self):
        """CLASSES should be a non-empty dict mapping ints to strings."""
        assert isinstance(CLASSES, dict)
        assert len(CLASSES) > 0
        assert all(isinstance(k, int) and isinstance(v, str) for k, v in CLASSES.items())

    def test_expected_classes(self):
        """CLASSES should contain the four expected categories."""
        expected = {"COVID19", "NORMAL", "PNEUMONIA", "TURBERCULOSIS"}
        assert set(CLASSES.values()) == expected

    def test_image_size(self):
        """IMAGE_SIZE should be a tuple of two positive integers."""
        assert isinstance(IMAGE_SIZE, tuple)
        assert len(IMAGE_SIZE) == 2
        assert all(isinstance(dim, int) and dim > 0 for dim in IMAGE_SIZE)

    def test_color_mode(self):
        """COLOR_MODE should be a valid Keras color mode string."""
        assert COLOR_MODE in ("rgb", "grayscale", "rgba")
