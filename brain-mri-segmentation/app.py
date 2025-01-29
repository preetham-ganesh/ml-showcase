import uuid
import os
import time

from flask import Flask, send_from_directory, request, render_template, redirect
import requests
from dotenv import load_dotenv
import numpy as np
import skimage


# Creates a flask application.
app = Flask(__name__)


@app.route("/send_image/")
def send_image():
    """Sends a saved image from the directory based on the provided file path.

    Sends a saved image from the directory based on the provided file path.

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
def error() -> str:
    """Renders template for error message.

    Renders template for error message.

    Args:
        None.

    Returns:
        A string for the rendered template for error.
    """
    # Extracts the query parameters from the request
    image_file_path = request.args.get("image_file_path")
    message = request.args.get("message")
    return render_template(
        "error.html", input_file_path=image_file_path, message=message
    )


@app.route("/positive")
def positive() -> str:
    """Renders template for viewing positive result.

    Renders template for viewing positive esult.

    Args:
        None.

    Returns:
        A string for the rendered template for viewing result.
    """
    # Extracts the query parameters from the request
    input_file_path = request.args.get("input_file_path")
    output_file_path = request.args.get("output_file_path")
    score = request.args.get("score")
    return render_template(
        "positive.html",
        input_file_path=input_file_path,
        output_file_path=output_file_path,
        score=round(float(score) * 100, 3),
    )
