import os
import json

import requests
import numpy as np

from src.utils import load_json_file

from typing import Dict, Any


class DigitRecognizer(object):
    """Recognizes digit in an image."""

    def __init__(self, model_version: str, model_api_url: str) -> None:
        """Creates object attributes for DigitRecognizer class.

        Creates object attributes for DigitRecognizer class.

        Args:
            model_version: A string for the version of the model.
            model_api_url: A string for the URL of the model's API.

        Returns:
            None.
        """
        # Asserts type of input arguments.
        assert isinstance(
            model_version, str
        ), "Variable model_version should be of type 'str'."
        assert isinstance(
            model_api_url, str
        ), "Variable model_api_url should be of type 'str'."

        # Initializes class variables.
        self.model_version = model_version
        self.model_api_url = model_api_url

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
            self.home_directory_path, "configs", "models", "digit_recognizer"
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
                f"Digit Recognizer model v{self.model_version} status: {response.status_code}"
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

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocesses the image for prediction.

        Preprocesses the image for prediction.

        Args:
            image: A NumPy array for the image.

        Returns:
            A numpy array for the preprocessed image.
        """
        # Asserts type & value of the arguments.
        assert isinstance(image, np.ndarray), "Variable image of type 'np.ndarray'."

        # If image is in RGB format, converts it to grayscale.
        if len(image.shape) == 3:
            image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

        # Expands the dimensions of the image in the first axis.
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=-1)

        # Casts input image to float32 and normalizes the image from [0, 255] range to [0, 1] range.
        image = np.float32(image)
        image = image / 255.0
        return image

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """Preprocesses image based on model requirements. Predicts digit recognized from image.

        Preprocesses image based on model requirements. Predicts digit recognized from image.

        Args:
            image: A NumPy array for the image.

        Returns:
            A dictionary for status of the prediction, along with predicted digit & prediction's confidence score.
        """
        # Asserts type & value of the arguments.
        assert isinstance(
            image, np.ndarray
        ), "Variable image should be of type 'np.ndarray'."

        # Preprocesses image based on model requirements.
        model_input_image = self.preprocess_image(image)

        # Sends model input image as input to Model using URL.
        try:
            response = requests.post(
                self.model_api_url,
                data=json.dumps({"inputs": model_input_image.tolist()}),
                headers={"content-type": "application/json"},
            )
        except requests.exceptions.ConnectionError:
            return {
                "status": "Failure",
                "message": "Serving URL does not exist. Received 'requests.exceptions.ConnectionError' error.",
            }

        # If status is 200, then extracts the prediction from the response.
        if response.status_code == 200:
            prediction = np.array(
                json.loads(response.text)["outputs"], dtype=np.float32
            )

            # Computes the digit predicted by the model, & extracts the confidence score.
            predicted_digit = int(np.argmax(prediction[0]))
            score = float(prediction[0][predicted_digit])
            return {"status": "Success", "digit": predicted_digit, "score": score}

        # Else returns the text from response.
        else:
            return {"status": "Failure", "message": response.text}
