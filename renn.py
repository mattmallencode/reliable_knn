from imblearn.under_sampling import (
    RepeatedEditedNearestNeighbours,  # type: ignore
)
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
        missing_values: list[str] = ["?"],
        theta: float = 2,
        agree_func: str = "sd_neighbors",
        kind_sel: str = "mode",
    ):
        """Initialises a kNN Harness that applies RENN to training data.

        Keyword arguments:
        regressor_or_classifier -- what kNN it runs 'regressor' | 'classifier'.
        dataset_file_path -- file path to the dataset to run the kNN on.
        test_size -- what percentage of the dataset to reserve for testing.
        missing_values -- strings denoting missing values in the dataset.
        """

        super().__init__(
            regressor_or_classifier,
            dataset_file_path,
            target_column_name,
            missing_values,
        )

        self._theta = theta
        self._agree_func = agree_func
        self._kind_sel = kind_sel

    @property
    def theta(self) -> float:
        return self._theta

    @property
    def agree_func(self) -> str:
        return self._agree_func

    @property
    def kind_sel(self) -> str:
        return self._kind_sel

    def _preprocess_dataset(
        self,
        dataset: pd.DataFrame,
        training_targets: pd.Series,
        training_cols: pd.Index | None = None,
        scaler: StandardScaler | None = None,
    ) -> tuple[np.ndarray, np.ndarray, pd.Index, StandardScaler]:
        """Preprocesses data, returns cols & scaler. Applies RENN if train.

        Keyword arguments:
        dataset -- the dataset we wish to preprocess.
        training_targets -- the targets associated with the training dataset.
        training_cols -- defaults to None; used to account for missing cols.
        scaler -- the StandardScaler used to scale the data, init if None.
        """

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
            scaler,
        ) = super()._preprocess_dataset(
            dataset,
            training_targets,
            training_cols,
            scaler,
        )

        # If it's a training set, apply RENN to it & update dataset & targets.
        if is_training_set:
            try:
                renn = RepeatedEditedNearestNeighbours(
                    training_targets=training_targets_np,
                    kind_sel=self.kind_sel,
                    agree_func=self.agree_func,
                    theta=self.theta,
                    n_neighbors=self.curr_k,
                    regressor_or_classifier=self.regressor_or_classifier,
                )
            except Exception as e:
                print(e)

            try:
                dataset_np, training_targets_np = renn.fit_resample(
                    dataset_np, training_targets_np
                )
            except Exception as e:
                print(e)

        return (dataset_np, training_targets_np, training_cols, scaler)


# test = RENNHarness("classifier", "datasets/heart.data", "num")
# test = KNNHarness('classifier', 'datasets/custom_cleveland.data', 'num')
test = RENNHarness("regressor", "datasets/abalone.data", "Rings", agree_func="sd_whole")
print(test.evaluate())
