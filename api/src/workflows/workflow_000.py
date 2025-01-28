import os
import time

from PIL import Image
import numpy as np

from src.utils import load_json_file, save_json_file
from src.models.digit_recognizer import DigitRecognizer


class Workflow000(object):
    """Recognizes digit in an image."""

    def __init__(self, workflow_version: str, models_base_url: str) -> None:
        """Creates object attributes for the Workflow000 class.

        Creates object attributes for the Workflow000 class.

        Args:
            workflow_version: A string for the version of the workflow.
            models_base_url: A string for the base URL where the model's in the workflow are served.

        Returns:
            None.
        """
        # Checks types & values of arguments.
        assert isinstance(
            workflow_version, str
        ), "Variable workflow_version should be of type 'str'."
        assert isinstance(
            models_base_url, str
        ), "Variable models_base_url should be of type 'str'."

        # Initializes class variables.
        self.workflow_version = workflow_version
        self.models_base_url = models_base_url

    def load_workflow_configuration(self) -> None:
        """Loads the workflow configuration file for current version.

        Loads the workflow configuration file for current version.

        Args:
            None.

        Returns:
            None.
        """
        self.home_directory_path = os.getcwd()
        workflow_configuration_directory_path = os.path.join(
            self.home_directory_path, "configs", "workflows", "workflow_000"
        )
        self.workflow_configuration = load_json_file(
            f"v{self.workflow_version}", workflow_configuration_directory_path
        )

    def load_workflow_models(self) -> None:
        """Loads each model & utility files in the workflow.

        Loads each model & utility files in the workflow.

        Args:
            None.

        Returns:
            None.
        """
        start_time = time.time()

        # Creates objects for models in workflow.
        self.digit_recognizer = DigitRecognizer(
            self.workflow_configuration["digit_recognizer"]["version"],
            f"{self.models_base_url}/v1/models/digit_recognizer_"
            + f"v{self.workflow_configuration['digit_recognizer']['version']}:predict",
        )

        # Loads model configuration as dictionary for all the models.
        self.digit_recognizer.load_model_configuration()

        # Checks if the model's TensorFlow Serving URL is working as expected.
        self.digit_recognizer.test_model_api()
        print()
        print(
            "Finished loading serialized models for Workflow000 in {} sec.".format(
                round(time.time() - start_time, 3)
            )
        )
        print()

    def generate_prediction_parameters(
        self, submission_id: str, image_file_path: str
    ) -> None:
        """Generates parameters required for workflow result.

        Generates parameters required for workflow result.

        Args:
            submission_id: A string for the unique id of the submission.
            image_file_path: A string for the location of the image.

        Returns:
            None.
        """
        # Checks types & values of arguments.
        assert isinstance(
            submission_id, str
        ), "Variable submission_id should be of type 'str'."
        assert isinstance(
            image_file_path, str
        ), "Variable image_file_path should be of type 'str'."

        # Adds submission id to the class's variable.
        self.submission_id = submission_id

        # Adds image to the class's variable.
        self.image = np.asarray(Image.open(image_file_path))

        # A dictionary for storing result extracted by the workflow.
        self.output = {
            "submission_id": submission_id,
            "workflow_id": "workflow_000",
            "configuration_version": f"v{self.workflow_version}",
        }

    def save_results(self) -> None:
        """Saves extracted result as a JSON file.

        Saves extracted result as a JSON file.

        Args:
            None.

        Returns:
            None.
        """
        # Saves the extracted document dictionary as a JSON file.
        save_json_file(
            self.output,
            self.submission_id,
            "data/out",
        )
        print()

    def workflow_prediction(self) -> None:
        """Executes workflow to recgonize digit in an image.

        Executes workflow to recgonize digit in an image.

        Args:
            None.

        Returns:
            None.
        """
        start_time = time.time()
        print()

        # Recognizes digit in the image.
        task_start_time = time.time()
        prediction = self.digit_recognizer.predict(self.image)
        print(
            f"Finished recognizing digit in image for submission id "
            + f"{self.submission_id} in {(time.time() - task_start_time):.3f} sec."
        )
        print()

        # Adds prediction to the output.
        if prediction["status"] == "Success":
            self.output["status"] = "Success"
            self.output["prediction"] = {
                "digit": prediction["digit"],
                "score": prediction["score"],
            }
        else:
            self.output["status"] = "Failure"
            self.output["message"] = prediction["message"]

        # Saves extracted result as a JSON file.
        self.save_results()
        print(
            f"Finished predicting output for submission id {self.submission_id} in "
            + f"{(time.time() - start_time):.3f} sec."
        )
        print()
