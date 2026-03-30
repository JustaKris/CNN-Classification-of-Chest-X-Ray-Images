# Use an official Python runtime as a parent image
FROM python:3.12-slim-bookworm AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends gcc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files for layer caching
COPY pyproject.toml uv.lock /app/

# Create a venv for the app dependencies
RUN uv venv /opt/venv

# Export production deps, strip GPU packages, install CPU-only alternatives
RUN uv export --frozen --no-dev --no-hashes --no-emit-project > requirements.txt && \
    # Remove GPU packages, torch (install CPU version separately), and packages not needed in Docker
    sed -i -E '/^(torch|torchvision|tensorflow|nvidia[-_]|triton|streamlit|opencv-python)/d' requirements.txt && \
    # Install CPU-only PyTorch (no torchvision needed - only torch is used by CLIP)
    uv pip install --python /opt/venv/bin/python --no-cache \
        torch --index-url https://download.pytorch.org/whl/cpu && \
    # Install CPU-only TensorFlow
    uv pip install --python /opt/venv/bin/python --no-cache tensorflow-cpu && \
    # Install remaining production dependencies
    uv pip install --python /opt/venv/bin/python --no-cache -r requirements.txt

# ====================
# Stage 2: Runtime
# ====================
FROM python:3.12-slim-bookworm

WORKDIR /app

# Copy the venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# Allow Python to find the xray_classifier package under src/
ENV PYTHONPATH="/app/src"

# Copy application code
COPY . /app

# Expose port 5050
EXPOSE 5050

# Run the app with gunicorn for production
# 1 worker: ML inference is CPU-bound, multiple workers cause TF thread contention
# 2 threads: allows serving static pages while inference runs
# preload: loads models once in master before forking
CMD ["gunicorn", "xray_classifier.web.flask_app:app", "--bind", "0.0.0.0:5050", "--workers", "1", "--threads", "2", "--preload", "--timeout", "120"]
