from imblearn.under_sampling import RepeatedEditedNearestNeighbours  # type: ignore
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler  # type: ignore
from harness import KNNHarness


class RENNHarness(KNNHarness):

    def __init__(
            self,
            regressor_or_classifier: str,
            dataset_file_path: str,
            target_column_name: str,
            test_size: float = 0.2,
            missing_values: list[str] = ['?']
    ):
        '''Initialises a kNN Harness that applies RENN to training data.

        Keyword arguments:
        regressor_or_classifier -- what kNN it runs 'regressor' | 'classifier'.
        dataset_file_path -- file path to the dataset to run the kNN on.
        target_column_name -- name of the column we are predicting.
        test_size -- what percentage of the dataset to reserve for testing.
        missing_values -- strings denoting missing values in the dataset.
        '''

        super().__init__(regressor_or_classifier, dataset_file_path,
                         target_column_name, missing_values)

    def _preprocess_dataset(
        self,
        dataset: pd.DataFrame,
        training_cols: pd.Index | None = None,
        scaler: StandardScaler | None = None,
        training_targets: pd.Series = pd.Series(),
    ) -> tuple[np.ndarray, np.ndarray, pd.Index, StandardScaler]:
        '''Preprocesses data, returns cols & scaler used. Also applies RENN if training.

        Keyword arguments:
        dataset -- the dataset we wish to preprocess.
        training_cols -- defaults to None; used to account for missing cols post split.
        scaler -- the StandardScaler used to scale the data, init if None is passed.
        training_targets -- target series for training data (empty if val or test).
        '''

        # Check if this is a training set that is being preprocessed.
        if scaler is None and training_cols is None:
            is_training_set: bool = True
        else:
            is_training_set = False

        dataset_np: np.ndarray
        training_targets_np: np.ndarray

        # Preprocess the dataset using the super class' method.
        (
            dataset_np,
            training_targets_np,
            training_cols,
            scaler
        ) = super()._preprocess_dataset(
            dataset,
            training_cols,
            scaler,
            training_targets=training_targets
        )

        # If it's a training set, apply RENN to it and update dataset and targets.
        if is_training_set and self.regressor_or_classifier == 'classifier':

            renn = RepeatedEditedNearestNeighbours()

            dataset_np, training_targets_np = renn.fit_resample(
                dataset_np, training_targets_np)

        return (dataset_np, training_targets_np, training_cols, scaler)


# test = RENNHarness('classifier', 'datasets/zoo.data', 'type')
# test = KNNHarness('classifier', 'datasets/custom_cleveland.data', 'num')
# print(test.evaluate())