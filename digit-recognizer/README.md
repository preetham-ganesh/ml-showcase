# Digit Recognizer

🚀 **Live Demo**: [Click here to access the application](https://digit-recognizer.preethamganesh.com)

This directory container contains the front-end interface for the [Digit Recognizer](https://github.com/preetham-ganesh/digit-recognizer) project. It provides a web-based UI built with Flask, HTML, and CSS to interact with the ML models deployed in the backend. Users can upload images and view the predictions from the model.

## Docker

Requires: [Docker](https://www.docker.com)

Use the following code snippet to deploy the docker container locally:

```bash
docker build --no-cache -t digit-recognizer .
docker run -d -p 3001:3001 --name digit-recognizer digit-recognizer
```

## UI Routes

| Route     | Method | Description                                                                              |
| --------- | ------ | ---------------------------------------------------------------------------------------- |
| `/`       | `GET`  | Redirects to Upload page.                                                                |
| `/upload` | `POST` | Handles user selection of an image from a randomized list and submits it for prediction. |
| `/error`  | `GET`  | Displays error messages if an issue occurs during prediction.                            |
| `/result` | `GET`  | Shows the predicted result after processing the uploaded image.                          |
