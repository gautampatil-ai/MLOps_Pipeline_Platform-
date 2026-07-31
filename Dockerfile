# Base Python image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and models
COPY src/ ./src/
COPY models/ ./models/

# Run the training script during docker build to ensure updated artifact exists
RUN python src/train.py

# Expose Render default port
EXPOSE 10000

# Command to launch production ASGI server
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "10000"]
