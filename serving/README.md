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

## Models Information

| Project                | Model Name                       | Model Version | Description                                                                                     | API Endpoint                                                                         |
| ---------------------- | -------------------------------- | ------------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| Digit Recognizer       | Digit Recognizer                 | v1.0.0        | A CNN model that recognizes digit in an image.                                                  | http://172.17.0.1:8501/v1/models/digit_recognizer_v1.0.0:predict                     |
| Brain MRI Segmentation | FLAIR Abnormality Classification | v1.2.0        | A CNN model that classifies whether a given Brain MRI image has abnormality.                    | http://172.17.0.1:8501/v1/models/bms_flair_abnormality_classification_v1.2.0:predict |
| Brain MRI Segmentation | FLAIR Abnormality Segmentation   | v1.0.0        | A U-Net model with MobileNetV2 pretrained on ImageNet as Encoder, and custom layers as decoder. | http://172.17.0.1:8501/v1/models/bms_flair_abnormality_segmentation_v1.0.0:predict   |
