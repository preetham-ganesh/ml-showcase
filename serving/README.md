# ML Showcase - Serving

This directory houses trained and serialized machine learning models for various projects. These models are optimized for serving and deployment using TensorFlow Serving, Docker, and other deployment tools.

## Contents

- [Usage](https://github.com/preetham-ganesh/ml-showcase/serving#usage)
- [Models Information](https://github.com/preetham-ganesh/ml-showcase/serving#models-information)
- [Support](https://github.com/preetham-ganesh/ml-showcase/serving#support)

## Usage

Requires: [Docker](https://www.docker.com)

Use the following code snippet to deploy the docker container locally:

```bash
docker build --no-cache -t ml-showcase-models .
docker run -d -p 8500:8500 -p 8501:8501 --name ml-showcase-models ml-showcase-models
```
