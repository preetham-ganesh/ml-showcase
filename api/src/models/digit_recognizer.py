import os


from src.utils import load_json_file


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
