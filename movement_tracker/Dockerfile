# Use official slim Python 3.13 runtime image
FROM python:3.13-slim

# Prevent Python from writing pyc files to disk and buffer outputs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set container working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications
COPY requirements.txt .

# Install project python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source codebase
COPY . .

# Expose FastAPI REST / WebSocket server port
EXPOSE 8000

# Execute Uvicorn web server instance
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
