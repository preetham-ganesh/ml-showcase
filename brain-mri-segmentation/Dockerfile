# Uses Python 3.9 Slim Buster as Base image.
FROM python:3.9-slim-buster

# Sets working directory.
WORKDIR /brain-mri-segmentation

# Copys application files to working directory.
COPY . /brain-mri-segmentation

# Installs system dependencies.
RUN apt-get update && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrades Pip and install python dependencies.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Exposes Port 3002.
EXPOSE 3002

# Start application in production mode.
CMD ["python3", "app.py", "-dt", "prod"]
