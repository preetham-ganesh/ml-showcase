# Uses Python 3.9 Slim Buster as Base image.
FROM python:3.9-slim-buster

# Sets working directory.
WORKDIR /digit-recognizer

# Copys application files to working directory.
COPY . /digit-recognizer

# Installs system dependencies.
RUN apt-get update && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrades Pip and install python dependencies.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Exposes Port 3001.
EXPOSE 3001

# Start application in production mode.
CMD ["python3", "app.py", "-dt", "prod"]
