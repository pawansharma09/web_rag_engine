# --- STAGE 1: Build ---
# This stage installs all dependencies
FROM python:3.10-slim AS builder

WORKDIR /app

# Install system dependencies that might be needed for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# --- STAGE 2: Final ---
# This stage copies the installed dependencies and the app code
FROM python:3.10-slim

WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the application code
COPY . .

# --- Notes on Running ---
# This Dockerfile can build an image with all code and dependencies.
# In a real production setup, you would use Docker Compose to run the
# api, worker, and streamlit app as separate services from this same image.

# Example CMD for running all services (for demo only, not production-robust)
# This is fragile as it doesn't manage process failure.
# CMD sh -c "python rag_core.py && \
#            uvicorn api:app --host 0.0.0.0 --port 8000 & \
#            python worker.py & \
#            streamlit run app.py --server.port 8501 --server.enableCORS false"

# Recommended CMD for running just the API (if using Docker Compose)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

# To run the worker, you'd override the command:
# docker run -d <image_name> python worker.py

# To run streamlit, you'd override the command:
# docker run -d -p 8501:8501 <image_name> streamlit run app.py --server.port 8501 --server.enableCORS false
