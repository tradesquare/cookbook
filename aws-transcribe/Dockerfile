# Use explicit platform for ECS compatibility
FROM --platform=linux/amd64 python:3.11-slim

# Set Python environment variables for production
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONHASHSEED=random
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies including audio processing libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    curl \
    portaudio19-dev \
    alsa-utils \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements and install Python dependencies
COPY requirements-aws.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-aws.txt

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Copy application files (ensure proper ownership)
COPY --chown=app:app app-streaming.py .

# Chainlit will work without these optional config files
# Create empty .chainlit directory for any future config
RUN mkdir -p .chainlit

# Expose port
EXPOSE 8000

# Set environment variables
ENV CHAINLIT_HOST=0.0.0.0
ENV CHAINLIT_PORT=8000

# Health check for container (check main Chainlit endpoint)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Run the application
CMD ["python", "-m", "chainlit", "run", "app-streaming.py", "--host", "0.0.0.0", "--port", "8000"]