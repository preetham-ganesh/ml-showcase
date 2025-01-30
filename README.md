# ML Showcase

This repository showcases my machine learning projects, each with its own front-end interface, a shared API, and a unified ML model server. The applications are containerized using Docker and deployed on a Homelab server, securely routed through Cloudflare.

## Installation

### Download the repo

```bash
git clone https://github.com/preetham-ganesh/ml-showcase.git
cd ml-showcase
```

## Project Components

### API Service (api/)

- Serves as the backend for managing ML workflows, handling requests, and interacting with the database.
- It provides a RESTful interface for model inference, training, and data retrieval while managing submission storage and completion status.
- Built with Flask, the API is containerized using Docker for scalable deployment.

### Brain MRI Segmentation (brain-mri-segmentation/)

- Contains the front-end interface for the [Brain MRI Segmentation](https://github.com/preetham-ganesh/brain-mri-segmentation) project.
- It provides a web-based UI built with Flask, HTML, and CSS to interact with the backend ML models.
- Users can upload images and view the model's predictions.

### Digit Recognizer (digit-recognizer/)

- Contains the front-end interface for the [Digit Recognizer](https://github.com/preetham-ganesh/digit-recognizer) project.
- It provides a web-based UI built with Flask, HTML, and CSS to interact with the backend ML models.
- Users can upload images and view the model's predictions.

### Serving (serving/)

- Contains trained and serialized machine learning models for various projects.
- Models are optimized for serving and deployment using TensorFlow Serving, Docker, and other tools.
- It manages the lifecycle of ML models, including downloading pre-trained models and serving predictions.
