# Project commands — x-ray-image-classifier-app

This file contains the most relevant commands for working with this repository. It focuses on local development: uv environment setup, running Jupyter notebooks, testing and code-quality commands, and common Git workflows.

## uv / environment

Install uv (if not already installed):

```powershell
# Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install all dependencies from `pyproject.toml` (creates `.venv` automatically):

```powershell
uv sync
```

Install with specific dependency groups:

```powershell
# Production only (no dev tools)
uv sync --no-dev

# With notebook/training dependencies
uv sync --group notebooks

# With Streamlit frontend
uv sync --group streamlit

# With documentation tools
uv sync --group docs

# Everything
uv sync --all-groups
```

Activate the virtual environment (PowerShell):

```powershell
& ".venv\Scripts\Activate.ps1"
```

Add a new dependency:

```powershell
# Add to main dependencies
uv add some-package

# Add to a specific group
uv add --group dev some-dev-tool
uv add --group notebooks some-notebook-tool
```

Notes:

- If you want GPU-accelerated PyTorch, follow the official PyTorch install selector and install the appropriate `torch` wheel for your CUDA version.
- Transformers will download pretrained models on first run — make sure you have disk space.

## Jupyter / notebooks

Start Jupyter Lab (requires notebooks group):

```powershell
uv run --group notebooks jupyter lab
```

Or the classic notebook UI:

```powershell
uv run --group notebooks jupyter notebook
```

Open the notebooks in `notebooks/` and run cells top-to-bottom.

## Testing & code quality

Run tests (if/when you add them):

```powershell
uv run pytest
```

Lint and format:

```powershell
uv run ruff check src/
uv run ruff format src/
```

Type checking:

```powershell
uv run mypy src/
```

Security audit:

```powershell
uv run bandit -r src/ -c pyproject.toml
uv run pip-audit
```

Markdown linting:

```powershell
uv run pymarkdown --config pyproject.toml scan docs/ README.md
```

## Docker

Build and run the Docker image:

```powershell
docker build -t justakris/chest-xray-classification-app:latest .
```

Run attached (logs stream to terminal, container removed on exit):

```powershell
docker run -it --rm -p 5050:5050 --name chest-xray-classification justakris/chest-xray-classification-app:latest
```

Run detached in the background:

```powershell
docker run -d --rm -p 5050:5050 --name chest-xray-classification justakris/chest-xray-classification-app:latest
```

Follow logs of a detached container:

```powershell
docker logs -f chest-xray-classification
```

Stop and remove the container:

```powershell
docker stop chest-xray-classification
```

## Common Git workflows

Clone and start working:

```powershell
git clone <repo-url>
cd CNN-Classification-of-Chest-X-Ray-Images
uv sync
```

Daily workflow:

```powershell
git pull origin main
git checkout -b feature/your-feature
# make changes
git add .
git commit -m "Describe changes"
git push -u origin feature/your-feature
```

Undo local changes:

```powershell
git checkout -- <path/to/file>
git reset HEAD <path/to/file>
```

## Quick reference

- Install deps: `uv sync`
- Start Jupyter Lab: `uv run --group notebooks jupyter lab`
- Run tests: `uv run pytest`
- Lint code: `uv run ruff check src/`
- Format code: `uv run ruff format src/`
