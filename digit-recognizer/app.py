import uuid
import os
import argparse

from flask import Flask, send_from_directory, request, render_template, redirect
import requests
from dotenv import load_dotenv
import numpy as np
import skimage


# Creates a flask application.
app = Flask(__name__)


@app.route("/send_image")
def send_image():
    """Sends a saved image from the directory based on the provided file path.

    Args:
        file_path: The relative path to the image file to send.

    Returns:
        The requested file if it exists, or a 400 error if the file path is invalid.
    """
    file_path = request.args.get("file_path")

    if not file_path:
        return "File path is required", 400

    try:
        # Send the file using send_from_directory
        directory, file_name = os.path.split(file_path)
        return send_from_directory(directory, file_name)
    except Exception as e:
        return f"Error: {str(e)}", 400


@app.route("/error")
def error(submission_id: str, message: str) -> str:
    """Renders template for error message.

    Renders template for error message.

    Args:
        submission_id: A string for the unique id of the submission.
        message: A string for the error message.

    Returns:
        A string for the rendered template for error.
    """
    return render_template("error.html", submission_id=submission_id, message=message)


@app.route("/in_progress", methods=["GET", "POST"])
def in_progress(image_file_path: str, submission_id: str) -> str:
    """"""
    if request.method == "POST":
        _ = ""
    else:
        return render_template(
            "in_progress.html",
            input_file_path=image_file_path,
            submission_id=submission_id,
        )


@app.route("/upload", methods=["GET", "POST"])
def upload() -> str:
    """Renders template for uploading image to Digit Recognizer. Recognizes digit in image.

    Renders template for uploading image to Digit Recognizer. Recognizes digit in image.

    Args:
        None.

    Returns:
        A string for the rendered template for upload or complete.
    """
    # Checks if request method is POST, then predicts digit for uploaded image.
    if request.method == "POST":
        # Extracts image file path from request.
        image_file_path = request.form["selected_image"]
        image_file_path = image_file_path.lstrip("../")

        # Generates a unique id for this submission to ensure traceability.
        submission_id = str(uuid.uuid4())

        # Defines the API URL for submitting the image to the ML model.
        submit_image_api_url = f"{host_url}/api/v1/submit_image"

        #


if __name__ == "__main__":
    print()

    # Parses the arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-dt",
        "--deployment_type",
        type=str,
        required=True,
        help="Type of the deployment.",
    )
    args = parser.parse_args()

    # Based on the type of deployment, the host URL is set.
    if args.deployment_type == "local":
        host_url = "http://localhost:8100"
    else:
        host_url = "http://host.docker.internal:8100"

    # Runs app on specified host & port (For local deployment)
    app.run(host="0.0.0.0", port=3000)
