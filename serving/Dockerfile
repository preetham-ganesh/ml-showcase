# Uses an Alpine-based Debian image.
FROM debian:bullseye-slim

# Sets the working directory inside the container.
WORKDIR /

# Updates packages and install dependencies.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    unzip \
    wget \
    curl \
    ca-certificates \
    gnupg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Adds TensorFlow Model Server repository and key
RUN echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" > /etc/apt/sources.list.d/tensorflow-serving.list && \
    curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | apt-key add -

# Updates packages and install TensorFlow Model Server Universal.
RUN apt-get update && \
apt-get install -y tensorflow-model-server-universal && \
apt-get clean && \
rm -rf /var/lib/apt/lists/*

# Copies all files to the working directory.
COPY . .

# Makes the script executable.
RUN chmod +x download_models.sh

# Runs the download models script
RUN ./download_models.sh

# Exposes TensorFlow Model Server's default port.
EXPOSE 8500
EXPOSE 8501

# Starts the tensorflow model server.
CMD ["tensorflow_model_server", "--port=8500", "--rest_api_port=8501", "--model_config_file=/configs/models.config"]
