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
