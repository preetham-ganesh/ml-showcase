# ML Showcase - API

This directory contains the core ML Showcase API, which serves as the backend for managing machine learning workflows, handling requests, and interacting with the database. It provides RESTful endpoints for model inference, training, and data retrieval. The API is built using Flask and is designed to be containerized with Docker for scalable deployment.

## Docker

Requires: [Docker](https://www.docker.com)

Use the following code snippet to deploy the docker container locally:

```bash
docker build --no-cache -t ml-showcase-api .
docker run -d -p 8100:8100 --name ml-showcase-api ml-showcase-api
```

## API Endpoints

| Endpoint                             | Method | Description                                                                                               |
| ------------------------------------ | ------ | --------------------------------------------------------------------------------------------------------- |
| /api/v1/submit_image                 | POST   | Submits image to the API. Accepts file and workflow_name as inputs. Validates the inputs & workflow_name. |
| /api/v1/fetch_result/<submission_id> | GET    | Checks if prediction output is ready. If yes, then returns the output, else returns current status.       |
