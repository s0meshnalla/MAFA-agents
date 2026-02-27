# Use the official Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements first to cache the installation step
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Suppress TensorFlow warnings (equivalent to your PS script)
ENV TF_ENABLE_ONEDNN_OPTS="0"

# Cloud Run injects a $PORT environment variable (usually 8080). 
# This CMD uses $PORT if available, otherwise defaults to your local 5001.
CMD uvicorn API:app --host 0.0.0.0 --port ${PORT:-5001}