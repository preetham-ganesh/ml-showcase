# ML Showcase - Serving

This directory houses trained and serialized machine learning models for various projects. These models are optimized for serving and deployment using TensorFlow Serving, Docker, and other deployment tools.

## Contents

- [Docker](https://github.com/preetham-ganesh/ml-showcase/serving#docker)
- [Models Information](https://github.com/preetham-ganesh/ml-showcase/serving#models-information)

## Docker

Requires: [Docker](https://www.docker.com)

Use the following code snippet to deploy the docker container locally:

```bash
docker build --no-cache -t ml-showcase-models .
docker run -d -p 8500:8500 -p 8501:8501 --name ml-showcase-models ml-showcase-models
```

## Models Information

### Digit Recognizer v1.0.0

| Attribute Name        | Details                                                          |
| --------------------- | ---------------------------------------------------------------- |
| **Project Name**      | Digit Recognizer                                                 |
| **Model Description** | A CNN model that recognizes digit in an image.                   |
| **Hugging Face URL**  | https://huggingface.co/preethamganesh/digit-recognizer-v1.0.0    |
| **API Endpoint**      | http://172.17.0.1:8501/v1/models/digit_recognizer_v1.0.0:predict |

#### Example API Request

```python
import requests
import json
import numpy as np

response = requests.post(
    "http://172.17.0.1:8501/v1/models/digit_recognizer_v1.0.0:predict",
    data=json.dumps({"inputs": np.zeros((1, 28, 28, 1)).tolist()}),
    headers={"content-type": "application/json"},
)

prediction = np.array(json.loads(response.text)["outputs"], dtype=np.float32)
```

### FLAIR Abnormality Classification v1.2.0

| Attribute Name        | Details                                                                              |
| --------------------- | ------------------------------------------------------------------------------------ |
| **Project Name**      | Brain MRI Segmentation                                                               |
| **Model Description** | A CNN model that classifies whether a given Brain MRI image has abnormality.         |
| **Hugging Face URL**  | https://huggingface.co/preethamganesh/bms-flair-abnormality-classification-v1.2.0    |
| **API Endpoint**      | http://172.17.0.1:8501/v1/models/bms_flair_abnormality_classification_v1.2.0:predict |

```python
import requests
import json
import numpy as np

response = requests.post(
    "http://172.17.0.1:8501/v1/models/bms_flair_abnormality_classification_v1.2.0:predict",
    data=json.dumps({"inputs": np.zeros((1, 256, 256, 3)).tolist()}),
    headers={"content-type": "application/json"},
)

prediction = np.array(json.loads(response.text)["outputs"], dtype=np.float32)
```

### FLAIR Abnormality Segmentation v1.0.0

| Attribute Name        | Details                                                                                         |
| --------------------- | ----------------------------------------------------------------------------------------------- |
| **Project Name**      | Brain MRI Segmentation                                                                          |
| **Model Description** | A U-Net model with MobileNetV2 pretrained on ImageNet as Encoder, and custom layers as decoder. |
| **Hugging Face URL**  | https://huggingface.co/preethamganesh/bms-flair-abnormality-segmentation-v1.0.0                 |
| **API Endpoint**      | http://172.17.0.1:8501/v1/models/bms_flair_abnormality_segmentation_v1.0.0:predict              |

```python
import requests
import json
import numpy as np

response = requests.post(
    "http://172.17.0.1:8501/v1/models/bms_flair_abnormality_segmentation_v1.0.0:predict",
    data=json.dumps({"inputs": np.zeros((1, 256, 256, 3)).tolist()}),
    headers={"content-type": "application/json"},
)

prediction = np.array(json.loads(response.text)["outputs"], dtype=np.float32)
```
