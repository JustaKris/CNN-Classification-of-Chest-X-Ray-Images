# Chest X-Ray Image Classification

[![Tests](https://github.com/JustaKris/CNN-Classification-of-Chest-X-Ray-Images/actions/workflows/python-tests.yml/badge.svg)](https://github.com/JustaKris/CNN-Classification-of-Chest-X-Ray-Images/actions/workflows/python-tests.yml)
[![Lint](https://github.com/JustaKris/CNN-Classification-of-Chest-X-Ray-Images/actions/workflows/python-lint.yml/badge.svg)](https://github.com/JustaKris/CNN-Classification-of-Chest-X-Ray-Images/actions/workflows/python-lint.yml)
[![Type Check](https://github.com/JustaKris/CNN-Classification-of-Chest-X-Ray-Images/actions/workflows/python-typecheck.yml/badge.svg)](https://github.com/JustaKris/CNN-Classification-of-Chest-X-Ray-Images/actions/workflows/python-typecheck.yml)
[![Security](https://github.com/JustaKris/CNN-Classification-of-Chest-X-Ray-Images/actions/workflows/security-audit.yml/badge.svg)](https://github.com/JustaKris/CNN-Classification-of-Chest-X-Ray-Images/actions/workflows/security-audit.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end deep learning application that classifies chest X-ray images into four diagnostic categories — **COVID-19**, **Normal**, **Pneumonia**, and **Tuberculosis** — and explains its predictions through Grad-CAM heatmap visualizations. The project spans the full ML lifecycle: research and experimentation in Jupyter notebooks, a production-ready web application with input validation, containerized deployment, and CI/CD automation.

## Web Application

The core of this project is an interactive Flask web app that turns the trained model into a usable diagnostic tool. Users upload a chest X-ray (or paste a URL), and the app returns a classification along with a Grad-CAM overlay showing which regions of the image influenced the prediction.

### Key Features

- **Image upload or URL input** — accepts chest X-ray images from local files or via URL
- **Real-time classification** — returns one of four diagnoses with a confidence score
- **Grad-CAM visualization** — generates a heatmap overlay highlighting the areas the model focused on, making predictions interpretable
- **CLIP-based input validation** — uses OpenAI's CLIP model (zero-shot) to verify the uploaded image is actually a chest X-ray before running the classifier, preventing nonsensical predictions on unrelated images
- **Alternative Streamlit frontend** — a second interface built with Streamlit for quick prototyping and comparison

### Prediction Pipeline

```text
User Image --> CLIP Validation --> Preprocessing --> MobileNetV3 Prediction --> Grad-CAM Heatmap
                    |                                        |                         |
               Is it an X-ray?                       Class + Confidence          Activation Overlay
                    |                                        |                         |
               (if not: warn)                                +----------+--------------+
                                                             |
                                                       Display Results
```

1. **Input validation** — CLIP zero-shot classification screens the image against several categories (X-ray, photograph, document, etc.) and warns the user if it doesn't appear to be a chest X-ray.
2. **Preprocessing** — the image is resized to 224×224 and processed with TensorFlow's ImageNet preprocessing to match the model's expected input format.
3. **Prediction** — the MobileNetV3 transfer learning model (92% accuracy) outputs class probabilities across the four categories.
4. **Explainability** — Grad-CAM computes gradients against the final convolutional layer to produce a heatmap, which is overlaid on the original image.

### Deployment

The app is containerized with a multi-stage Docker build and has been deployed to Azure App Service via GitHub Actions CI/CD. Deployment workflows for both Azure and Render are included.

```bash
# Run locally with Docker
docker build -t chest-xray-classifier .
docker run -p 5050:5050 chest-xray-classifier
```

## Research & Model Development

All experimentation — data exploration, model architecture comparisons, hyperparameter tuning, and evaluation — is documented in the Jupyter notebooks.

**[View the Research Notebook](./notebooks/CNN%20classification%20of%20chest%20X-Ray%20Images.ipynb)**

### Approach

Several CNN architectures were trained and compared:

| Approach | Description | Best Accuracy |
| --- | --- | --- |
| **CNN from scratch** | Custom convolutional architecture | ~82% |
| **Expanded CNN + Keras Tuner** | Deeper architecture with automated hyperparameter search | ~82% |
| **MobileNetV3 Transfer Learning** | Pre-trained MobileNetV3Small with fine-tuned classification head | **92%** |

The **MobileNetV3 transfer learning model** was selected for the production app. MobileNetV3Small was chosen specifically because its lightweight architecture allows the app to run within the memory constraints of free-tier cloud platforms, while still achieving strong classification performance.

### Training Details

- **Class imbalance handling** — manual class weights compensate for the uneven distribution across categories (COVID-19 and Tuberculosis samples are significantly fewer than Normal and Pneumonia)
- **Custom metrics** — per-class precision and recall, alongside a differentiable F1-based loss function
- **Experiment tracking** — all training runs logged with MLflow, with checkpoints saved for each experiment
- **Visualization** — TensorBoard integration for monitoring training metrics in real time

### Dataset

The dataset combines three Kaggle chest X-ray datasets into **7,135 images** split across train, validation, and test sets. Each image belongs to one of four classes:

| Class | Description |
| --- | --- |
| COVID-19 | Chest X-rays showing COVID-19 indicators |
| Normal | Healthy chest X-rays |
| Pneumonia | Chest X-rays showing pneumonia |
| Tuberculosis | Chest X-rays showing tuberculosis indicators |

[Kaggle Dataset](https://www.kaggle.com/datasets/jtiptj/chest-xray-pneumoniacovid19tuberculosis)

## Technology Stack

| Category | Technologies |
| --- | --- |
| **Deep Learning** | TensorFlow / Keras, PyTorch (CLIP only), Keras Tuner |
| **Web** | Flask, Streamlit, Jinja2, Gunicorn |
| **MLOps** | MLflow, TensorBoard, Docker, GitHub Actions |
| **Code Quality** | Ruff (lint + format), Mypy, Pytest, Bandit, pip-audit |
| **Infrastructure** | Azure App Service, Render, Docker Hub |
| **Package Management** | uv, Hatchling (src-layout) |

## Project Structure

```text
src/xray_classifier/              # Installable Python package
├── config.py                     # Global constants and class labels
├── data/                         # Dataset loading and augmentation
├── models/                       # Model architectures and training pipeline
├── tuning/                       # Keras Tuner hyperparameter search
├── utils/                        # Grad-CAM, CLIP validation, MLflow, evaluation
└── web/                          # Flask & Streamlit apps, templates, static assets

saved_models/                     # Trained model weights (.keras)
notebooks/                        # Research notebooks with MLflow and TensorBoard logs
tests/                            # Pytest test suite
docs/                             # Architecture docs and developer guide
.github/workflows/                # CI/CD: tests, lint, typecheck, security, Docker, deploy
```

For the full architecture breakdown, see [docs/architecture.md](docs/architecture.md).

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/JustaKris/CNN-Classification-of-Chest-X-Ray-Images.git
cd CNN-Classification-of-Chest-X-Ray-Images

# Install dependencies
uv sync --group dev

# Run the Flask app
uv run xray-flask
```

The app will be available at `http://localhost:5050`.

### Other Commands

```bash
# Run tests
uv run pytest

# Lint and format
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Type checking
uv run mypy src/

# Run the Streamlit frontend instead
uv sync --group streamlit
uv run xray-streamlit
```

For Docker setup, CI/CD details, and contribution guidelines, see the [Developer Guide](docs/dev-guide.md).

## License

This project is licensed under the MIT License.
