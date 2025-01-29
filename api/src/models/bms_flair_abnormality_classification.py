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
