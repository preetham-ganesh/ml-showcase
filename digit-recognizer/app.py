import os
import argparse
import time

from flask import (
    Flask,
    send_from_directory,
    request,
    render_template,
    redirect,
    url_for,
)
import requests


# Creates a flask application.
app = Flask(__name__)


@app.route("/send_image")
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


@app.route("/result")
def result() -> str:
    """Renders template for viewing result.

    Renders template for viewing result.

    Args:
        None.

    Returns:
        A string for the rendered template for viewing result.
    """
    # Extracts the query parameters from the request
    image_file_path = request.args.get("image_file_path")
    predicted_digit = request.args.get("predicted_digit")
    score = request.args.get("score")
    return render_template(
        "result.html",
        input_file_path=image_file_path,
        predicted_digit=predicted_digit,
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
    fetch_result_api_url = f"{host_url}/api/v1/fetch_result/{submission_id}"

    # Sets polling interval (in seconds) and maximum iterations for polling.
    polling_interval = 2
    max_iterations = 30

    # Iterates over the maximum iterations to fetch the result.
    for _ in range(max_iterations):
        result_response = requests.get(fetch_result_api_url)

        # Based on the status code of response, redirects to appropriate page.
        if result_response.status_code == 200:
            result = result_response.json()

            if result["status"] == "Success":
                return redirect(
                    url_for(
                        "result",
                        image_file_path=image_file_path,
                        predicted_digit=result["prediction"]["digit"],
                        score=result["prediction"]["score"],
                    )
                )

            elif result["status"] == "Failure":
                return redirect(
                    url_for(
                        "error",
                        message=result["message"],
                        image_file_path=image_file_path,
                    )
                )

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
        image_file_path = request.form["selected-image"]
        image_file_path = image_file_path.lstrip("../")

        # Defines the API URL for submitting the image to the ML model.
        submit_image_api_url = f"{host_url}/api/v1/submit_image"

        # Opens the image file and submit it to the ML Showcase API for prediction.
        try:
            with open(image_file_path, "rb") as image_file:
                submission_response = requests.post(
                    submit_image_api_url,
                    files={"image": image_file},
                    data={"workflow_name": "workflow_000"},
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

    # Based on the type of deployment, the host URL is set.
    if args.deployment_type == "local":
        host_url = "http://localhost:8100"
    else:
        host_url = "http://172.17.0.1:8100"

    # Runs app on specified host & port (For local deployment)
    app.run(host="0.0.0.0", port=3001)
