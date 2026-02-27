# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set non-interactive timezone and environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies required for audio processing and building
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libsndfile1 \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Install PyTorch with CUDA 12.1 support first (to leverage Docker layer caching)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create output directories so they exist when the container runs
RUN mkdir -p svaram_outputs expresso_tests expresso_intense_laughs

# Ensure the entry points are executable
RUN chmod +x train.py inference_svaram.py

# Default command: show help or just run the inference script to generate the default samples
CMD ["python", "inference_svaram.py"]
