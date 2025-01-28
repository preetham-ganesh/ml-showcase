# Base Image
FROM python:3.9-slim-buster

# Set Working Directory
WORKDIR /api

# Copy Application Files
COPY . /api

# Install System Dependencies
RUN apt-get update && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade Pip and Install Python Dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose Port
EXPOSE 8100

# Start Application
CMD ["python3", "app.py"]