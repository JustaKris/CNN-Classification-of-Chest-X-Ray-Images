# Environment setup - uv (recommended)
uv sync
uv sync --group notebooks    # Include Jupyter/training deps
uv sync --group streamlit    # Include Streamlit frontend
uv sync --all-groups          # Install everything

# Activate virtual environment
& ".venv\Scripts\Activate.ps1"

# Run app locally
uv run python app.py  # localhost:5050

# Tensorboard
uv run tensorboard --logdir=./logs/tensorboard

# Linting & formatting
uv run ruff check src/
uv run ruff format src/

# Type checking
uv run mypy src/

# Security audit
uv run bandit -r src/ -c pyproject.toml
uv run pip-audit

# Run unit tests
uv run pytest

# Docker
docker build -t justakris/chest-xray-classification-app:latest .
# Run attached — logs stream to terminal, container removed on exit
docker run -it --rm -p 5050:5050 --name chest-xray-classification justakris/chest-xray-classification-app:latest
# Run detached in the background
docker run -d --rm -p 5050:5050 --name chest-xray-classification justakris/chest-xray-classification-app:latest
# Follow logs of a detached container
docker logs -f chest-xray-classification
docker stop chest-xray-classification

# Docker Hub
docker login
docker tag chest-xray-classification-app justakris/chest-xray-classification-app:latest
docker push justakris/chest-xray-classification-app:latest

# Interacting with container
docker exec -it chest-xray-classification bash

# GitHub Actions and deployment to Render 
# Render Key -> rnd_FTBxo7rYaRqdbL9frCXpSyFuBfsg
# Render ID (from hook in settings) -> srv-cpmjffg8fa8c73ajaajg
# Render ID Docker version -> srv-cpooujmehbks73enrqlg