# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Use Railway's PORT environment variable (Railway sets this automatically)
# Railway typically uses port 8080, not 3000
EXPOSE $PORT

# Command to run the application
# Use the PORT environment variable that Railway provides
CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 bot:app
