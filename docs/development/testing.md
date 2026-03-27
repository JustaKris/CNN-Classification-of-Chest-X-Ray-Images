# Testing Guide

Testing practices using pytest for the chest X-ray classification project.

## Quick Start

```powershell
# Run all tests
uv run pytest -v

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run a specific test file
uv run pytest tests/test_utils.py -v
```

## Test Organisation

```text
tests/
├── conftest.py          # Shared fixtures and configuration
├── test_config.py       # Configuration and constants tests
├── test_data_loader.py  # Data pipeline tests
├── test_evaluate.py     # Evaluation utility tests
├── test_grad_cam.py     # Grad-CAM heatmap tests
├── test_image_classifier.py  # CLIP image type recognition tests
├── test_models.py       # Model architecture tests
└── test_utils.py        # Utility function tests
```

## Running Tests

```powershell
# All tests
uv run pytest -v

# Specific file
uv run pytest tests/test_utils.py -v

# Specific test
uv run pytest tests/test_utils.py::test_get_date -v

# With coverage report
uv run pytest --cov=src --cov-report=term-missing
```

## Writing Tests

### Best Practices

1. **One test, one behaviour** — each test verifies a single thing
2. **Use descriptive names** — `test_preprocess_image_converts_grayscale_to_rgb`
3. **Arrange-Act-Assert** — clear test structure
4. **Mock heavy dependencies** — TensorFlow model loading, CLIP inference
5. **Test edge cases** — invalid images, missing files, empty inputs

### Example

```python
import pytest
from unittest.mock import patch, MagicMock
from xray_classifier.utils.utils import get_date, get_time, preprocess_image


class TestUtilFunctions:
    """Test utility functions."""

    def test_get_date_format(self):
        """Date string matches YYYY.MM.DD format."""
        date = get_date()
        assert len(date) == 10
        assert date[4] == "."

    def test_get_time_format(self):
        """Time string matches HH-MM format."""
        time = get_time()
        assert len(time) == 5
        assert time[2] == "-"

    def test_preprocess_image_invalid_path(self):
        """Invalid path raises an exception."""
        with pytest.raises(Exception):
            preprocess_image("nonexistent.jpg")
```

## CI/CD Integration

Tests run automatically in GitHub Actions:

```yaml
- name: Run tests
  run: |
    uv run pytest -v --cov=src --cov-report=term-missing
```

## Related Documentation

- [Linting Guide](linting.md) — Code quality checks
- [Code Style](code-style.md) — General style guidelines
