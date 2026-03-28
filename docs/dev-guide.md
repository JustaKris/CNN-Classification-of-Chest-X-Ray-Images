# Developer Guide

Comprehensive guide covering environment setup, running the applications,
Docker, code quality, CI/CD pipelines, and common Git workflows.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- [Docker](https://www.docker.com/) (for containerised deployment)

### Installing uv

```powershell
# Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex
```

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Environment Setup

Install all dependencies from `pyproject.toml` (creates `.venv` automatically):

```bash
uv sync
```

Install with specific dependency groups:

```bash
# Core + dev tools (linting, testing, type checking)
uv sync --group dev

# Include Streamlit frontend
uv sync --group streamlit

# Include notebook/training dependencies
uv sync --group notebooks

# Everything
uv sync --all-groups
```

Activate the virtual environment manually (usually not needed with `uv run`):

```powershell
# PowerShell
& ".venv\Scripts\Activate.ps1"
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Add a new dependency:

```bash
# Main dependency
uv add some-package

# To a specific group
uv add --group dev some-dev-tool
```

> **Note:** If you want GPU-accelerated PyTorch, follow the official
> [PyTorch install selector](https://pytorch.org/get-started/locally/) and
> install the appropriate `torch` wheel for your CUDA version.

## Running the Flask App

The Flask web server is registered as a CLI entry point via `pyproject.toml`.

```bash
uv run xray-flask
```

The server starts on **<http://localhost:5050>** by default.

### Environment Variables

| Variable | Default | Description |
| ---------- | ------- | ----------- |
| `PORT` | `5050` | Port the Flask server listens on |
| `DEBUG` | `false` | Enable Flask debug mode (`true` / `false`) |

Example with a custom port:

```bash
PORT=8080 uv run xray-flask
```

### Production (Gunicorn)

```bash
uv run gunicorn "xray_classifier.web.flask_app:app" --bind 0.0.0.0:5050
```

## Running the Streamlit App

Make sure the `streamlit` dependency group is installed, then:

```bash
uv run xray-streamlit
```

Streamlit opens a browser tab automatically (default **<http://localhost:8501>**).

## Running Notebooks

Install the notebooks dependency group, then start Jupyter:

```bash
uv sync --group notebooks
uv run jupyter lab
```

Or the classic notebook UI:

```bash
uv run jupyter notebook
```

Notebooks are in the `notebooks/` directory. Run cells top-to-bottom.

## TensorBoard

View training logs with TensorBoard:

```bash
uv run tensorboard --logdir=./notebooks/TensorBoard
```

## Docker

### Build the Image

```bash
docker build -t justakris/chest-xray-classification-app:latest .
```

The multi-stage Dockerfile installs CPU-only versions of TensorFlow and PyTorch
to keep the image size manageable.

### Run the Container

```bash
# Foreground (logs stream to terminal, container removed on exit)
docker run -it --rm -p 5050:5050 --name chest-xray justakris/chest-xray-classification-app:latest

# Background (detached)
docker run -d --rm -p 5050:5050 --name chest-xray justakris/chest-xray-classification-app:latest
```

The Flask app will be available at **<http://localhost:5050>**.

### View Logs and Interact

```bash
# Follow logs of a detached container
docker logs -f chest-xray

# Open a shell inside the running container
docker exec -it chest-xray bash
```

### Stop the Container

```bash
docker stop chest-xray
```

### Publishing to Docker Hub

```bash
docker login

# Tag the image (if not already tagged)
docker tag chest-xray-classification-app justakris/chest-xray-classification-app:latest

# Push to Docker Hub
docker push justakris/chest-xray-classification-app:latest
```

## Code Quality

### Linting

[Ruff](https://docs.astral.sh/ruff/) checks for code issues
(PEP 8, imports, common bugs). See [development/linting.md](development/linting.md)
for configuration details.

```bash
# Check for lint errors
uv run ruff check src/ tests/

# Auto-fix issues (including import sorting)
uv run ruff check --fix src/ tests/
```

### Formatting

Ruff also handles code formatting (Black-compatible). See
[development/formatting.md](development/formatting.md) for standards.

```bash
# Format code
uv run ruff format src/ tests/

# Check formatting without changes
uv run ruff format --check src/ tests/
```

### Type Checking

[mypy](https://mypy.readthedocs.io/) performs static type analysis.

```bash
uv run mypy src/ tests/
```

### Security Scanning

[Bandit](https://bandit.readthedocs.io/) scans for code vulnerabilities and
[pip-audit](https://github.com/pypa/pip-audit) checks for known dependency CVEs.
See [development/security.md](development/security.md) for details.

```bash
uv run bandit -r src/ -c pyproject.toml
uv run pip-audit
```

### Markdown Linting

[pymarkdownlnt](https://github.com/jackdewinter/pymarkdown) checks documentation
formatting. See [development/markdown-linting.md](development/markdown-linting.md).

```bash
uv run pymarkdown --config pyproject.toml scan docs/ README.md
```

### Testing

[pytest](https://docs.pytest.org/) with coverage reporting. See
[development/testing.md](development/testing.md) for writing tests and conventions.

```bash
# Run all tests with coverage
uv run pytest

# Verbose output
uv run pytest tests/ -v

# Run a specific test file
uv run pytest tests/test_utils.py
```

Coverage reports are generated in `reports/coverage/`.

## CI/CD Pipelines

All CI workflows are defined in `.github/workflows/` and run on GitHub Actions.

### Quality Gates (on every push and PR)

| Workflow | File | What It Does |
| ---------- | ------ | -------------- |
| Lint | `python-lint.yml` | Runs `ruff check` and `ruff format --check` on Python 3.11 and 3.12 |
| Type Check | `python-typecheck.yml` | Runs `mypy` (non-blocking — allowed to fail) |
| Tests | `python-tests.yml` | Runs `pytest` with coverage, uploads reports to Codecov |
| Security | `security-audit.yml` | Runs `bandit` and `pip-audit` (weekly schedule + on push to main) |
| Markdown | `markdown-lint.yml` | Runs `pymarkdown` on docs and README (non-blocking) |

### Build and Deploy (on push to main)

| Workflow | File | What It Does |
| ---------- | ------ | -------------- |
| Docker Build | `build-push-docker.yml` | Builds the Docker image and pushes to Docker Hub |
| Azure Deploy | `deploy-azure.yml` | Deploys to Azure App Service (manual trigger) |
| Render Deploy | `deploy-render.yml` | Deploys to Render via webhook (manual trigger) |

The Docker workflow tags images as `prod` on main and `dev` on other branches.
Azure and Render deployments are configured for manual dispatch to conserve
free-tier resources.

### Required Secrets

| Secret | Used By |
| -------- | -------- |
| `DOCKERHUB_USERNAME` | Docker Build |
| `DOCKERHUB_TOKEN` | Docker Build |
| `AZURE_TENANT_ID` | Azure Deploy |
| `AZURE_CLIENT_ID` | Azure Deploy |
| `AZURE_SUBSCRIPTION_ID` | Azure Deploy |
| `RENDER_DEPLOY_WEBHOOK` | Render Deploy |
| `RENDER_SERVICE_URL` | Render Deploy |

## Git Workflows

### Getting Started

```bash
git clone https://github.com/JustaKris/CNN-Classification-of-Chest-X-Ray-Images.git
cd CNN-Classification-of-Chest-X-Ray-Images
uv sync --group dev
```

### Daily Workflow

```bash
git pull origin main
git checkout -b feature/your-feature
# make changes
git add .
git commit -m "Describe changes"
git push -u origin feature/your-feature
```

### Undo Local Changes

```bash
git checkout -- <path/to/file>
git reset HEAD <path/to/file>
```

## Quick Reference

| Task | Command |
| ------ | --------- |
| Install deps | `uv sync` |
| Flask app | `uv run xray-flask` |
| Streamlit app | `uv run xray-streamlit` |
| Jupyter Lab | `uv run jupyter lab` |
| Lint | `uv run ruff check src/ tests/` |
| Format | `uv run ruff format src/ tests/` |
| Type check | `uv run mypy src/ tests/` |
| Tests | `uv run pytest` |
| Docker build | `docker build -t justakris/chest-xray-classification-app:latest .` |
| Docker run | `docker run -it --rm -p 5050:5050 justakris/chest-xray-classification-app:latest` |
