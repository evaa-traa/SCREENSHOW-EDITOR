# Use an official lightweight Python image
FROM python:3.10-slim

# Install system dependencies (FFmpeg is critical)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face requires running as a non-root user
RUN useradd -m -u 1000 user

# Set environment variables for the new user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

# Switch to the non-root user
USER user

# Set working directory
WORKDIR $HOME/app

# Copy requirements and install them
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy the rest of the application code
COPY --chown=user:user . .

# Hugging Face Spaces exposes port 7860 by default
EXPOSE 7860

# Use Gunicorn to run the Flask app on port 7860 with an extended timeout for video processing
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "2", "--timeout", "600", "app:app"]
