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
