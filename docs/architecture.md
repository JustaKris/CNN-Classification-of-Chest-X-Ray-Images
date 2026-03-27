# Project Architecture

## Overview

This project is a chest X-ray image classification application that uses deep learning
(CNN / Transfer Learning) to classify chest X-ray images into four categories:
COVID-19, Normal, Pneumonia, and Tuberculosis. It provides a Flask web interface
with Grad-CAM heatmap visualization and an optional Streamlit frontend.

## Package Layout

The project uses a **src-layout** with a named package `xray_classifier` under `src/`.
CLI entry points are defined in `pyproject.toml` under `[project.scripts]`.

```text
.
├── Dockerfile                      # Multi-stage Docker build (CPU-only)
├── pyproject.toml                  # Project metadata, dependencies, tool config
├── uv.lock                        # Locked dependency versions
│
├── src/
│   └── xray_classifier/            # Installable Python package
│       ├── __init__.py
│       ├── config.py                # Global constants (classes, image size, etc.)
│       ├── exception.py             # Custom exception with traceback enrichment
│       ├── logger.py                # Centralized JSON/colored logging
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── data_generator.py    # Keras ImageDataGenerator wrapper
│       │   └── data_loader.py       # Dataset loading and augmentation
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── models.py            # Model architectures (MobileNetV3, EfficientNet)
│       │   └── train.py             # Training pipeline
│       │
│       ├── tuning/
│       │   ├── __init__.py
│       │   └── tune_hyperparameters.py  # Keras Tuner integration
│       │
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── callbacks.py         # Keras callbacks (checkpoints, LR scheduling)
│       │   ├── custom_loss.py       # F1-based loss function
│       │   ├── custom_metrics.py    # Precision/recall metrics
│       │   ├── evaluate.py          # Model evaluation and plotting
│       │   ├── grad_cam.py          # Grad-CAM heatmap generation
│       │   ├── image_classifier.py  # CLIP-based image type validation
│       │   ├── mlflow_utils.py      # MLflow experiment tracking
│       │   ├── tensorflow_gpu_setup.py  # CUDA/GPU configuration
│       │   └── utils.py             # Image preprocessing, model loading helpers
│       │
│       └── web/                     # Web interfaces
│           ├── __init__.py
│           ├── flask_app.py          # Flask application
│           ├── streamlit_app.py     # Streamlit application
│           ├── templates/           # Flask Jinja2 HTML templates
│           │   ├── index.html
│           │   ├── prediction.html
│           │   └── warning.html
│           └── static/              # CSS, JS, images
│               ├── css/
│               ├── images/
│               └── js/
│
├── saved_models/                    # Trained model weights (.keras)
├── data/                            # Chest X-ray image dataset (train/val/test)
├── notebooks/                       # Jupyter research notebooks
├── tests/                           # Pytest test suite
├── reports/                         # Coverage reports
├── docs/                            # Project documentation
│   ├── architecture.md              # This file
│   ├── dev-guide.md                 # Developer guide (apps, Docker, CI, Git)
│   └── development/                 # Development guides
│       ├── code-style.md
│       ├── formatting.md
│       ├── linting.md
│       ├── logging.md
│       ├── markdown-linting.md
│       ├── security.md
│       └── testing.md
└── .github/workflows/               # CI/CD pipelines
    ├── python-tests.yml             # Pytest + coverage
    ├── python-lint.yml              # Ruff linting
    ├── python-typecheck.yml         # Mypy type checking
    ├── security-audit.yml           # Bandit + pip-audit
    ├── markdown-lint.yml            # Markdown linting
    ├── build-push-docker.yml        # Docker image build/push
    ├── deploy-azure.yml             # Azure App Service deployment
    └── deploy-render.yml            # Render deployment
```

## Component Details

### Entry Points

| Command | Script | Description |
| --------- | -------- | ------------- |
| `uv run xray-flask` | `xray_classifier.web.flask_app:main` | Start the Flask web server |
| `uv run xray-streamlit` | `xray_classifier.web.streamlit_app:main` | Start the Streamlit interface |

Entry points are defined in `pyproject.toml` under `[project.scripts]`.
In Docker, the Flask app runs via `python -m xray_classifier.web.flask_app`.

### Web Interfaces (`xray_classifier.web`)

| Module | Responsibility |
| -------- | --------------- |
| `flask_app.py` | Flask application: image upload/URL, CNN prediction, Grad-CAM, CLIP pre-screening |
| `streamlit_app.py` | Streamlit alternative with the same prediction pipeline |
| `templates/` | Flask Jinja2 HTML templates (index, prediction, warning pages) |
| `static/` | CSS, JavaScript, images, favicon |

### Core Package (`xray_classifier`)

| Module | Responsibility |
| -------- | --------------- |
| `config.py` | Central configuration: class labels, image dimensions, batch size, class weights |
| `logger.py` | Structured logging with JSON and colored console formatters |
| `exception.py` | `CustomException` with file/line traceback enrichment |

### Data Pipeline (`xray_classifier.data`)

| Module | Responsibility |
| -------- | --------------- |
| `data_generator.py` | Wraps Keras `ImageDataGenerator` with project defaults |
| `data_loader.py` | Loads train/val/test splits, applies augmentation transforms |

### Models (`xray_classifier.models`)

| Module | Responsibility |
| -------- | --------------- |
| `models.py` | Model factory functions: `MobileNetV3Transfer()`, `EfficientNetTransfer()` |
| `train.py` | End-to-end training script with callbacks, evaluation, and model saving |

### Utilities (`xray_classifier.utils`)

| Module | Responsibility |
| -------- | --------------- |
| `utils.py` | Image preprocessing, model loading, URL image fetching |
| `grad_cam.py` | Grad-CAM heatmap generation over model activation layers |
| `image_classifier.py` | CLIP model for validating uploaded images are chest X-rays |
| `callbacks.py` | Keras callbacks: checkpoints, early stopping, LR plateau, TensorBoard |
| `evaluate.py` | Classification reports, confusion matrices, accuracy/loss plots |
| `custom_loss.py` | F1-based differentiable loss function |
| `custom_metrics.py` | Per-class precision and recall Keras metrics |
| `mlflow_utils.py` | MLflow experiment tracking utilities |
| `tensorflow_gpu_setup.py` | CUDA memory growth and GPU configuration |

### Tuning (`xray_classifier.tuning`)

| Module | Responsibility |
| -------- | --------------- |
| `tune_hyperparameters.py` | Keras Tuner hyperparameter search integration |

## Data Flow

```text
User uploads image / provides URL
         │
         ▼
   ┌─────────────┐
   │  Flask App   │  (flask_app.py)
   └──────┬──────┘
          │
          ▼
   ┌──────────────┐
   │ CLIP Screener │  Validates input is a chest X-ray
   └──────┬───────┘
          │ (passes)
          ▼
   ┌──────────────┐
   │ Preprocessing │  Resize, normalize (ImageNet preprocessing)
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │  CNN Model    │  MobileNetV3 Transfer Learning
   └──────┬───────┘
          │
    ┌─────┴─────┐
    ▼           ▼
┌────────┐ ┌──────────┐
│ Predict │ │ Grad-CAM │  Activation heatmaps
└────┬───┘ └─────┬────┘
     │           │
     ▼           ▼
   ┌───────────────┐
   │  Results Page  │  Prediction + heatmap overlay
   └───────────────┘
```

## Build & Deployment

### Docker

Multi-stage build optimized for CPU-only deployment:

1. **Builder stage** — installs `uv`, exports deps, strips GPU packages,
   installs CPU-only PyTorch and TensorFlow
2. **Runtime stage** — copies the virtual environment and application code

### CI/CD

All workflows run on GitHub Actions triggered by pushes to `main` and pull requests:

- **Quality gates**: Ruff lint/format, mypy type checking, bandit security scan, pip-audit
- **Tests**: pytest with coverage reporting to Codecov
- **Build**: Docker image build and push to registry
- **Deploy**: Azure App Service (primary) and Render (alternative)

## Technology Stack

| Layer | Technology |
| ------- | ----------- |
| Deep Learning | TensorFlow/Keras 2.18+ |
| Image Validation | PyTorch + CLIP (Hugging Face Transformers) |
| Web Framework | Flask 3.x |
| Build System | Hatchling (PEP 517) |
| Package Manager | uv |
| Containerization | Docker (multi-stage, CPU-only) |
| CI/CD | GitHub Actions |
| Deployment | Azure App Service, Render |
| Testing | pytest + pytest-cov |
| Linting | Ruff, mypy, bandit |
