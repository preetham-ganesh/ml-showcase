# Brain MRI Segmentation

🚀 **Live Demo**: [Click here to access the application](https://brain-mri-segmentation.preethamganesh.com)

This directory container contains the front-end interface for the [Brain MRI Segmentation](https://github.com/preetham-ganesh/brain-mri-segmentation) project. It provides a web-based UI built with Flask, HTML, and CSS to interact with the ML models deployed in the backend. Users can upload images and view the predictions from the model.

## Docker

Requires: [Docker](https://www.docker.com)

Use the following code snippet to deploy the docker container locally:

```bash
docker build --no-cache -t brain-mri-segmentation .
docker run -d -p 3002:3002 --name brain-mri-segmentation brain-mri-segmentation
```
