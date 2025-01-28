import os
import sys
import uuid
import io
import sqlite3
import time


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_PATH)


from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from PIL import Image
import numpy as np

from src.utils import (
    load_json_file,
    check_directory_path_existence,
    generate_time_stamp,
)
from src.workflows.workflow_000 import Workflow000

from typing import Dict


# Creates flask app & enables CORS.
app = Flask(__name__)
CORS(app)

# Creates an empty dictionary to store loaded workflows.
workflows = dict()


def load_workflows(models_base_url: str) -> None:
    """Loads all the models & utility files for all workflows.

    Loads all the models & utility files for all workflows.

    Args:
        models_base_url: A string for the base URL where the model's in the workflow are served.

    Returns:
        None.
    """
    # Checks types & values of arguments.
    assert isinstance(
        models_base_url, str
    ), "Variable models_base_url should be of type 'str'."

    # Loads the version numbers for all workflows.
    home_directory_path = os.getcwd()
    workflow_versions = load_json_file(
        "configuration", os.path.join(home_directory_path, "configs", "workflows")
    )

    # Extracts the available workflow names from the version numbers dictionary.
    available_workflow_names = list(workflow_versions.keys())

    # Iterates across available workflow names, to initialize each workflow.
    for name in available_workflow_names:
        # Creates on object for the Workflow 000.
        if name == "workflow_000":
            workflows[name] = Workflow000(workflow_versions[name], models_base_url)

        # Loads the workflow configuration file for current version.
        workflows[name].load_workflow_configuration()

        # Loads each model & utility files in the workflow.
        workflows[name].load_workflow_models()


def initialize_databases():
    """Initializes SQLite3 databases used to track the progress of submissions to the ML showcase API.

    Initializes SQLite3 databases used to track the progress of submissions to the ML showcase API.

    Args:
        None.

    Returns:
        None.
    """
    global cursor, connection

    # Establish a connection to the SQLite3 database and create a cursor.
    connection = sqlite3.connect("ml_showcase_db.sqlite3", check_same_thread=False)
    cursor = connection.cursor()

    # Check if the 'submissions_info' table exists, if not, create it.
    try:
        cursor.execute("SELECT * FROM submissions_info LIMIT 1")
    except sqlite3.OperationalError:
        cursor.execute(
            """
            CREATE TABLE submissions_info (
                submission_id TEXT PRIMARY KEY NOT NULL,
                workflow_name TEXT NOT NULL,
                submission_time_stamp TEXT NOT NULL,
                file_extension TEXT NOT NULl
            )
            """
        )
        print("'submissions_info' does not exist. Creating a new one.")
        print()

    # Check if the 'submissions_completion_info' table exists, if not, create it.
    try:
        cursor.execute("SELECT * FROM submissions_completion_info LIMIT 1")
    except sqlite3.OperationalError:
        cursor.execute(
            """
            CREATE TABLE submissions_completion_info (
                submission_id TEXT PRIMARY KEY NOT NULL,
                workflow_name TEXT NOT NULL,
                submission_time_stamp TEXT NOT NULL,
                completion_time_stamp TEXT NOT NULL
            )
            """
        )
        print("'submissions_completion_info' does not exist. Creating a new one.")
        print()

    # Commit changes and reset the cursor.
    connection.commit()
    print("Databases initialized successfully.")
    print()


@app.route("/api/v1/submit_image/", methods=["POST"])
@cross_origin()
def submit_image() -> Dict[str, str]:
    """Submits the image to the API, validates the input, and stores the submission information in the database.

    Submits the image to the API, validates the input, and stores the submission information in the database.

    Args:
        None.

    Returns:
        A dictionary which contains the status of the image & its validity.
    """
    # Checks if the request contains an image.
    if "file" not in request.files:
        return (
            jsonify({"status": "Failure", "message": "Image was not submitted."}),
            400,
        )

    # Gets the uploaded file.
    file = request.files["file"]

    # Checks if the file has a valid extension.
    if file.filename.endswith(".png"):
        file_type = "image/png"
    elif file.filename.endswith(".jpg"):
        file_type = "image/jpg"
    elif file.filename.endswith(".jpeg"):
        file_type = "image/jpeg"
    else:
        return (
            jsonify(
                {
                    "status": "Failure",
                    "message": "File should have '.png', '.jpg', or '.jpeg' as extension.",
                }
            ),
            400,
        )

    # Checks if the request contains workflow id.
    if "workflow_name" not in request.form:
        return (
            jsonify(
                {
                    "status": "Failure",
                    "message": "Request should include 'workflow_name'.",
                }
            ),
            400,
        )
    else:
        workflow_name = request.form.get("workflow_name")

    # Checks if the workflow name entered is correct.
    try:
        workflows[workflow_name]
    except KeyError:
        return (
            jsonify(
                {
                    "status": "Failure",
                    "message": "Incorrect 'workflow_name' included in the request.",
                }
            ),
            400,
        )

    # Read the content in the file as bytes.
    file_content = file.read()

    # Generates a unique id for the submission.
    submission_id = str(uuid.uuid4())

    # Converts bytes into a NumPy array.
    try:
        image = Image.open(io.BytesIO(file_content))
    except Exception as e:
        return jsonify({"status": "Failure", "message": str(e)})

    # Checks if the following directory path exists.
    uploaded_data_directory_path = check_directory_path_existence("data/in")

    # Saves the image as a PNG image.
    image.save(os.path.join(uploaded_data_directory_path, f"{submission_id}.png"))

    # Updates image submissions info table, with the uploaded image information.
    cursor.execute(
        """
        INSERT INTO submissions_info (submission_id, workflow_name, submission_time_stamp, file_extension) 
        VALUES (?, ?, ?, ?)
        """,
        (submission_id, workflow_name, generate_time_stamp(), "png"),
    )
    connection.commit()

    # Returns the success message along with the unique id.
    return (
        jsonify(
            {
                "status": "Success",
                "submission_id": submission_id,
                "file_type": file_type,
                "message": "File submitted.",
            }
        ),
        200,
    )


def prediction() -> None:
    """Performs prediction for the uploaded input based on the workflow name.

    Performs prediction for the uploaded input based on the workflow name.

    Args:
        None.

    Returns:
        None.
    """
    wait_time = 2
    while True:
        # Checks if an image has been uploaded to the API.
        cursor.execute(
            """
            SELECT submission_id, workflow_name, submission_time_stamp, file_extension 
            FROM submissions_info 
            ORDER BY submission_time_stamp ASC 
            LIMIT 1
            """
        )
        row = cursor.fetchone()

        # If no new image has been uploaded, then checks after wait time seconds.
        if row is None:
            print(f"No new data was uploaded. Will check again after {wait_time} secs.")
            print()
            time.sleep(wait_time)
            continue

        # Extracts submission id, workflow name & submission time stamp from row.
        submission_id, workflow_name, submission_time_stamp, file_extension = row

        # Checks if the following directory path exists.
        uploaded_data_directory_path = check_directory_path_existence("data/in")

        # Generates parameters required for workflow result.
        workflows[workflow_name].generate_prediction_parameters(
            submission_id,
            f"{uploaded_data_directory_path}/{submission_id}.{file_extension}",
        )

        # Executes workflow to complete the prediction task.
        workflows[workflow_name].workflow_prediction()

        # Deletes submission id from submissions info table.
        cursor.execute(
            "DELETE FROM submissions_info WHERE submission_id=?",
            (submission_id,),
        )

        # Updates image submissions completion info table, with the latest prediction.
        cursor.execute(
            """
            INSERT INTO submissions_completion_info 
            (submission_id, workflow_name, submission_time_stamp, completion_time_stamp) 
            VALUES (?, ?, ?, ?)
            """,
            (
                submission_id,
                workflow_name,
                submission_time_stamp,
                generate_time_stamp(),
            ),
        )
        connection.commit()


@app.route("/api/v1/recognize_digit", methods=["POST"])
@cross_origin()
def recognize_digit() -> Dict[str, str]:
    """Recognizes digit in an image given as input.

    Recognizes digit in an image given as input.

    Args:
        None.

    Returns:
        None.
    """
    # Checks if the request contains an image.
    if "image" not in request.files:
        return (
            jsonify({"status": "Failure", "message": "Image was not submitted."}),
            400,
        )

    # Read the content in the file as bytes.
    image_content = request.files["image"].read()

    # Generates a unique id for the submission.
    submission_id = str(uuid.uuid4())

    # Converts bytes into a NumPy array.
    try:
        image = np.array(Image.open(io.BytesIO(image_content)))
    except Exception as e:
        return jsonify({"status": "Failure", "message": str(e)}), 400

    # Generates parameters required for workflow result.
    workflows["workflow_000"].generate_prediction_parameters(submission_id, image)

    # Executes workflow to recgonize digit in an image.
    workflows["workflow_000"].workflow_prediction()

    if workflows["workflow_000"].output["status"] == "Success":
        return jsonify(workflows["workflow_000"].output), 200
    else:
        return jsonify(workflows["workflow_000"].output), 400


if __name__ == "__main__":
    print()

    # Loads all the models & utility files for all workflows.
    load_workflows("http://host.docker.internal:8501")

    # Runs the app on host '0.0.0.0' and port 8100.
    app.run(host="0.0.0.0", port=8100)
