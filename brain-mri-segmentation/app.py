import os
import time
import argparse

from flask import (
    Flask,
    send_from_directory,
    request,
    render_template,
    redirect,
    url_for,
)
import requests
from PIL import Image
import numpy as np


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
        A string for the rendered template for viewing positive result.
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


@app.route("/negative")
def negative() -> str:
    """Renders template for viewing negative result.

    Renders template for viewing negative esult.

    Args:
        None.

    Returns:
        A string for the rendered template for viewing negative result.
    """
    # Extracts the query parameters from the request
    input_file_path = request.args.get("input_file_path")
    score = request.args.get("score")
    return render_template(
        "negative.html",
        input_file_path=input_file_path,
        score=round(float(score) * 100, 3),
    )


def process_result(image_file_path: str, submission_id: str) -> str:
    """Processes the result of the prediction.

    Processes the result of the prediction.

    Args:
        image_file_path: A string for the file path to the image.
        submission_id: A string for the submission ID of the image.

    Returns:
        A string for the rendered template for the result or error.
    """
    fetch_result_api_url = f"{API_HOST}/api/v1/fetch_result/{submission_id}"

    # Sets polling interval (in seconds) and maximum iterations for polling.
    polling_interval = 2
    max_iterations = 30

    # Iterates over the maximum iterations to fetch the result.
    for _ in range(max_iterations):
        result_response = requests.get(fetch_result_api_url)

        # Based on the status code of response, redirects to appropriate page.
        if result_response.status_code == 200:
            result = result_response.json()

            # If the prediction is an abnormality, saves the predicted image.
            if result["prediction"]["label"] == "abnormality":
                # Converts the predicted image data to a NumPy array.
                predicted_image = np.array(result["prediction"]["image"])
                print(predicted_image.shape)

                # Ensures the predicted image is in the correct format for saving.
                if predicted_image.max() <= 1:
                    predicted_image = (predicted_image * 255).astype("uint8")
                predicted_image = predicted_image.astype("uint8")

                # Converts NumPy array to a PIL image, saves it to a file.
                predicted_image = Image.fromarray(predicted_image)
                predicted_image.save(
                    os.path.join("data", "out", f"{submission_id}.png")
                )
                return redirect(
                    url_for(
                        "positive",
                        input_file_path=image_file_path,
                        output_file_path=os.path.join(
                            "data", "out", f"{submission_id}.png"
                        ),
                        score=result["prediction"]["score"],
                    )
                )

            # If the prediction is not an abnormality, redirects to negative result page.
            else:
                return redirect(
                    url_for(
                        "negative",
                        input_file_path=image_file_path,
                        score=result["prediction"]["score"],
                    )
                )

        # If the result is not found, redirects to error page.
        elif result_response.status_code == 404:
            return redirect(
                url_for(
                    "error",
                    message=result_response.text,
                    image_file_path=image_file_path,
                )
            )

        # Wait before the next polling attempt
        time.sleep(polling_interval)

    # If processing is still not completed after timeout, returns timeout as error message.
    return redirect(
        url_for(
            "error",
            message="Timeout. Server is busy. Please try again after some time.",
            image_file_path=image_file_path,
        )
    )


@app.route("/upload", methods=["GET", "POST"])
def upload() -> str:
    """Renders template for uploading image to Brain MRI Segmentation. Predicts FLAIR abnormality in uploaded MRI image.

    Renders template for uploading image to Brain MRI Segmentation. Predicts FLAIR abnormality in uploaded MRI image.

    Args:
        None.

    Returns:
        A string for the rendered template for upload or complete.
    """
    # Checks if request method is POST, then predicts digit for uploaded image.
    if request.method == "POST":
        # Extracts image file path from request.
        image_file_path = request.form["selected-image"]
        image_file_path = image_file_path.lstrip("../")

        # Defines the API URL for submitting the image to the ML model.
        submit_image_api_url = f"{API_HOST}/api/v1/submit_image"

        # Opens the image file and submit it to the ML Showcase API for prediction.
        try:
            with open(image_file_path, "rb") as image_file:
                submission_response = requests.post(
                    submit_image_api_url,
                    files={"image": image_file},
                    data={"workflow_name": "workflow_001"},
                )

            # Based on the status code of response, redirects to appropriate page.
            if submission_response.status_code == 200:
                submission_id = submission_response.json().get("submission_id")
                return process_result(image_file_path, submission_id)
            else:
                return redirect(
                    url_for(
                        "error",
                        message=submission_response.text,
                        input_file_path=image_file_path,
                    )
                )
        except Exception as e:
            return redirect(
                url_for("error", message=str(e), input_file_path=image_file_path)
            )
    else:
        # Renders the upload template if the request method is GET.
        return render_template("upload.html")


@app.route("/")
def index():
    """Redirects to upload page.

    Redirects to upload page.

    Args:
        None.

    Returns:
        A response for the redirected upload page.
    """
    return redirect("/upload")


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

    # Sets API host.
    host_name = "localhost" if args.deployment_type == "dev" else "api"
    API_HOST = f"http://{host_name}:8100"

    # Runs app on specified host & port (For local deployment)
    app.run(host="0.0.0.0", port=3002)
