import os
import time

from src.utils import load_json_file
from src.models.bms_flair_abnormality_classification import (
    FlairAbnormalityClassification,
)
from src.models.bms_flair_abnormality_segmentation import FlairAbnormalitySegmentation


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
            self.home_directory_path, "configs", "workflows", "workflow_001"
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

        # Creates objects for Predict class in corresponding models.
        self.flair_abnormality_classification = FlairAbnormalityClassification(
            self.workflow_configuration["bms_flair_abnormality_classification"][
                "version"
            ],
            f"{self.models_base_url}/v1/models/bms_flair_abnormality_classification_"
            + f"v{self.workflow_configuration['bms_flair_abnormality_classification']['version']}:predict",
        )
        self.flair_abnormality_segmentation = FlairAbnormalitySegmentation(
            self.workflow_configuration["bms_flair_abnormality_segmentation"][
                "version"
            ],
            f"{self.models_base_url}/v1/models/bms_flair_abnormality_segmentation_"
            + f"v{self.workflow_configuration['bms_flair_abnormality_segmentation']['version']}:predict",
        )

        # Loads model configuration as dictionary for all the models.
        self.flair_abnormality_classification.load_model_configuration()
        self.flair_abnormality_segmentation.load_model_configuration()

        # Checks if the model's TensorFlow Serving URL is working as expected.
        self.flair_abnormality_classification.test_model_api()
        self.flair_abnormality_segmentation.test_model_api()
        print()
        print(
            "Finished loading serialized models for Workflow001 in {} sec.".format(
                round(time.time() - start_time, 3)
            )
        )
        print()
