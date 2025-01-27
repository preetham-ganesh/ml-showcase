import os
import sys


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_PATH)


from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

from src.utils import load_json_file
from src.workflows.workflow_000 import Workflow000


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
