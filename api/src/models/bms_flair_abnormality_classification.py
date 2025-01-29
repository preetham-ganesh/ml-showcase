import os
import json

import requests
import numpy as np

from src.utils import load_json_file


class FlairAbnormalityClassification(object):
    """Predicts whether is FLAIR abnormality in brain MRI images."""

    def __init__(self, model_version: str, model_api_url: str) -> None:
        """Creates object attributes for the FlairAbnormalityClassification class.

        Creates object attributes for the FlairAbnormalityClassification class.

        Args:
            model_version: A string for the version of the model should be used for prediction.
            model_api_url: A string for the URL of the model's API.

        Returns:
            None.
        """
        # Asserts type & value of the arguments.
        assert isinstance(model_version, str), "Variable model_version of type 'str'."
        assert isinstance(model_api_url, str), "Variable model_api_url of type 'str'."

        # Initalizes class variables.
        self.model_version = model_version
        self.model_api_url = model_api_url
        self.id_to_class = {0: "no_abnormality", 1: "abnormality"}

    def load_model_configuration(self) -> None:
        """Loads the model configuration file for model version.

        Loads the model configuration file for model version.

        Args:
            None.

        Returns:
            None.
        """
        self.home_directory_path = os.getcwd()
        model_configuration_directory_path = os.path.join(
            self.home_directory_path,
            "configs",
            "models",
            "bms_flair_abnormality_classification",
        )
        self.model_configuration = load_json_file(
            f"v{self.model_version}", model_configuration_directory_path
        )

    def test_model_api(self) -> None:
        """Checks if the model's TensorFlow Serving URL is working as expected.

        Checks if the model's TensorFlow Serving URL is working as expected.

        Args:
            None.

        Returns:
            None.
        """
        # Checks if the model's TensorFlow Serving URL is working as expected.
        try:
            response = requests.post(
                self.model_api_url,
                data=json.dumps(
                    {
                        "inputs": np.zeros(
                            (
                                2,
                                self.model_configuration["model"]["final_image_height"],
                                self.model_configuration["model"]["final_image_width"],
                                self.model_configuration["model"]["n_channels"],
                            )
                        ).tolist()
                    }
                ),
                headers={"content-type": "application/json"},
            )
            print(
                f"FLAIR Abnormality Classification model v{self.model_version} status: {response.status_code}"
            )

            # Checks if response code is 200.
            if response.status_code != 200:
                print(response.text)
                exit()
        except requests.exceptions.ConnectionError:
            print(
                f"URL: {self.model_api_url} does not exist. Received 'requests.exceptions.ConnectionError' error."
            )
            exit()
