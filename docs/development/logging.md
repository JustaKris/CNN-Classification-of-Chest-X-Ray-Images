# Logging Configuration

**Centralized logging setup for development and production environments.**

---

## Overview

The project uses a centralized logging configuration (`src/logger.py`) that provides:

- **JSON logging** for production/cloud environments (Azure, Docker)
- **Colorized console logging** for local development
- **Structured log filtering** to suppress noisy third-party libraries
- **Consistent formatting** across all modules

## Quick Start

### Basic Usage

```python
from src.logger import configure_logging, get_logger

# Configure once at app startup
configure_logging(level="INFO", use_json=False)

# Get a logger for your module
logger = get_logger(__name__)

# Log messages
logger.info("Starting prediction pipeline")
logger.warning("Low confidence score detected")
logger.error("Failed to load model", exc_info=True)
```

### Application Configuration

The Flask app configures logging at startup:

```python
from src.logger import configure_logging, get_logger

configure_logging(level="INFO", use_json=False)
logger = get_logger(__name__)
```

## Configuration Options

### Log Levels

| Level | When to Use |
| ----- | ----------- |
| `DEBUG` | Detailed diagnostic information (variable values, flow control) |
| `INFO` | General operational messages (process started, completed, counts) |
| `WARNING` | Potentially problematic situations (missing optional files, low confidence) |
| `ERROR` | Error events that might still allow the application to continue |
| `CRITICAL` | Severe errors causing premature termination |

### Output Formats

#### Development (Colorized)

```python
configure_logging(level="INFO", use_json=False)
```

**Output:**

```text
2025-12-08 14:32:15 | INFO     | src.utils.utils | Image preprocessed successfully
2025-12-08 14:32:16 | INFO     | app              | Prediction: COVID19 (92.3%)
2025-12-08 14:32:17 | WARNING  | src.utils.image_classifier | Image may not be a chest X-ray
2025-12-08 14:32:18 | ERROR    | src.utils.utils  | Error loading model from ./models/missing.keras
```

**Features:**

- Color-coded log levels (Green=INFO, Yellow=WARNING, Red=ERROR)
- Human-readable timestamps
- Clear module names
- Easy to scan visually

#### Production (JSON)

```python
configure_logging(level="INFO", use_json=True)
```

**Output:**

```json
{"timestamp": "2025-12-08 14:32:15", "level": "INFO", "logger": "app", "message": "Prediction: COVID19", "module": "app", "process": 12345, "thread": 67890}
```

**Features:**

- Structured JSON for log aggregation tools
- Parseable by Azure Monitor, CloudWatch, Datadog, Loki
- Includes metadata (process ID, thread ID, module)
- Easy to query and filter

## Suppressed Libraries

The logging configuration automatically suppresses verbose output from:

- `transformers` — Model loading details and tokenizer warnings
- `tensorflow` — Internal TF logging
- `absl` — TensorFlow abseil logging
- `PIL` — Image processing debug messages
- `urllib3` — HTTP request details

To suppress additional libraries:

```python
import logging
logging.getLogger("some_noisy_library").setLevel(logging.ERROR)
```

## Best Practices

### Do

1. **Use %-style formatting** (deferred evaluation):

   ```python
   logger.info("Processed %d records in %.2fs", count, duration)
   logger.error("File not found: %s", file_path)
   ```

2. **Use appropriate log levels:**

   ```python
   logger.info("Model loaded successfully")
   logger.warning("Using default threshold for classification")
   logger.error("Failed to preprocess image")
   ```

3. **Include exception info for errors:**

   ```python
   try:
       model = load_model(path)
   except Exception as e:
       logger.error("Failed to load model: %s", e, exc_info=True)
       raise
   ```

4. **Configure once at startup:**

   ```python
   if __name__ == "__main__":
       configure_logging(level="INFO", use_json=False)
       main()
   ```

### Don't

1. **Don't use f-strings in log calls** (evaluates even if level is filtered):

   ```python
   # Bad
   logger.debug(f"Processing image {image_path}")

   # Good
   logger.debug("Processing image %s", image_path)
   ```

2. **Don't use print() statements:**

   ```python
   # Bad
   print("Processing started")

   # Good
   logger.info("Processing started")
   ```

3. **Don't log sensitive data:**

   ```python
   # Bad
   logger.info("API key: %s", api_key)

   # Good
   logger.info("Authentication successful")
   ```

## Deployment Considerations

### Local Development

```python
configure_logging(level="DEBUG", use_json=False)
```

### Docker/Azure

```python
import os
use_json = os.getenv("LOG_FORMAT", "json") == "json"
configure_logging(level="INFO", use_json=use_json)
```

## Related Documentation

- [Code Style](code-style.md) — General style guidelines
- [Linting Guide](linting.md) — Code quality checks

---

**Logging Module:** `src/logger.py`
