import os

from src.utils import load_json_file


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
