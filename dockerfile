# Base image with Python
FROM python:3.9-slim

# Set a working directory inside the container
WORKDIR /app

# Copy the application files to the working directory
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Gradio runs on
EXPOSE 7860

# Set the command to run the application
CMD ["python", "app.py"]
