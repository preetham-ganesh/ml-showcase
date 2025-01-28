import os
import sys
import uuid
import io
import sqlite3


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_PATH)


from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from PIL import Image
import numpy as np

from src.utils import load_json_file
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
