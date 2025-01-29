class Workflow001(object):
    """Predicts if a brain MRI image has FLAIR abnormality and predicts the segmentation mask."""

    def __init__(self, workflow_version: str, models_base_url: str) -> None:
        """Creates object attributes for the Workflow001 class.

        Creates object attributes for the Workflow001 class.

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
