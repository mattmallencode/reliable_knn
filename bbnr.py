import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler  # type: ignore
from harness import KNNHarness
from collections import defaultdict


class BBNRHarness(KNNHarness):

    def __init__(
            self,
            regressor_or_classifier: str,
            dataset_file_path: str,
            target_column_name: str,
            test_size: float = 0.2,
            missing_values: list[str] = ['?']
    ):
        '''Initialises a kNN Harness that applies BBNR to training data.

        Keyword arguments:
        regressor_or_classifier -- what kNN it runs 'regressor' | 'classifier'.
        dataset_file_path -- file path to the dataset to run the kNN on.
        target_column_name -- name of the column we are predicting.
        test_size -- what percentage of the dataset to reserve for testing.
        missing_values -- strings denoting missing values in the dataset.
        '''

        super().__init__(regressor_or_classifier, dataset_file_path,
                         target_column_name, test_size, missing_values)

    def _preprocess_dataset(
        self,
        dataset: pd.DataFrame,
        training_cols: pd.Index | None = None,
        scaler: StandardScaler | None = None,
        training_targets: pd.Series = pd.Series(),
    ) -> tuple[np.ndarray, np.ndarray, pd.Index, StandardScaler]:
        '''Preprocesses data, returns cols & scaler used. Also applies BBNR if training.

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

        # If it's a training set, apply BBNR to it and update dataset and targets.
        if is_training_set and self.regressor_or_classifier == 'classifier':
            pass

        return (dataset_np, training_targets_np, training_cols, scaler)

    def _build_bbnr_sets(
            self,
            training_set: np.ndarray,
            training_targets: np.ndarray
    ) -> tuple[dict, dict, dict]:
        '''Returns dicts of coverage sets, liability sets, dissim sets for training_set.

        Keyword arguments:
        training_set -- the training dataset we are running BBNR on.
        training_targets -- the target labels / values of the training_set.
        '''

        # Each integer in these dictionaries represents the index of a training example.

        # Each member of a set are the examples that the key example helps classify.
        coverage_dict: dict[int, set] = defaultdict(set)
        # Each member of a set are the examples that the key example helps misclassify.
        liability_dict: dict[int, set] = defaultdict(set)
        # Each member of a set are the neighbors of the key example that misclassify it.
        dissimilarity_dict: dict[int, set] = defaultdict(set)

        example_index: int
        example: np.ndarray

        for example_index, example in enumerate(training_set):

            # Get the actual label and predicted label for this example.
            actual_label: float = training_targets[example_index]

            # Get closest neighbors (k+1 and [1:] to not include the example itself).
            neighbor_indices: np.ndarray = self.get_k_nearest_neighbors(
                example, training_set, self.curr_k+1)[1:]
            
            # Get the prediction for the example.
            predicted_label: float = self.get_most_common_class(
                training_targets, neighbor_indices)

            neighbor_index: int

            # For each neighbor of the example.
            for neighbor_index in neighbor_indices:

                neighbor_label: float = training_targets[neighbor_index]

                # If the neighbor label misclassifies the example, update dissim.
                if neighbor_label != actual_label:
                    dissimilarity_dict[example_index].add(neighbor_index)

                # If prediction is correct and neighbor contributed, update coverage.
                if predicted_label == actual_label:
                    if neighbor_label == predicted_label:
                        coverage_dict[neighbor_index].add(example_index)

                # If prediction is wrong and neighbor contributed, update liability.
                elif neighbor_label != actual_label:
                    liability_dict[neighbor_index].add(example_index)
        
        # Return the coverage dictionaries containing the sets.
        return (coverage_dict, liability_dict, dissimilarity_dict)


# test = KNNHarness('regressor', 'datasets/abalone.data', 'Rings')
test = BBNRHarness('classifier', 'datasets/iris.data', 'class')
print(test.evaluate())
