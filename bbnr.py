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
            missing_values: list[str] = ['?']
    ):
        '''Initialises a kNN Harness that applies BBNR to training data.

        Keyword arguments:
        regressor_or_classifier -- what kNN it runs 'regressor' | 'classifier'.
        dataset_file_path -- file path to the dataset to run the kNN on.
        target_column_name -- name of the column we are predicting.
        missing_values -- strings denoting missing values in the dataset.
        '''

        super().__init__(regressor_or_classifier, dataset_file_path,
                         target_column_name, missing_values)

    def _preprocess_dataset(
        self,
        dataset: pd.DataFrame,
        training_targets: pd.Series,
        training_cols: pd.Index | None = None,
        scaler: StandardScaler | None = None,
    ) -> tuple[np.ndarray, np.ndarray, pd.Index, StandardScaler]:
        '''Preprocesses data, returns cols & scaler used. BBNR if training.

        Keyword arguments:
        dataset -- the dataset we wish to preprocess.
        training_targets -- the targets associated with the training dataset.
        training_cols -- defaults to None; used to account for missing cols.
        scaler -- the StandardScaler used to scale the data, init if None.
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
            training_targets,
            training_cols,
            scaler,
        )

        # If it's a training set, apply BBNR to it & update dataset & targets.
        if is_training_set and self.regressor_or_classifier == 'classifier':

            # First, build the case-base competence model.
            coverage_dict: dict[int, set]
            liability_dict: dict[int, set]
            dissimilarity_dict: dict[int, set]

            (
                coverage_dict,
                liability_dict,
                dissimilarity_dict
            ) = self._build_bbnr_sets(dataset_np, training_targets_np)

            # The size of the liability sets of each example with liabilities.
            lset_sizes: np.ndarray = np.array(
                [
                    (key, len(liability_dict[key]))
                    for key in liability_dict.keys()
                ]
            )

            # If there aren't liabilities, return the data as is.
            if len(liability_dict) == 0:
                return (dataset_np, training_targets_np, training_cols, scaler)

            # The indices of the examples (sorted by desc liability size).
            indxs_by_desc_lset_size: np.ndarray = lset_sizes[
                lset_sizes[:, 1].argsort()[
                    ::-1]][:, 0]
            curr_example_index: int = indxs_by_desc_lset_size[0]
            removed_examples = np.zeros(dataset_np.shape[0], dtype=np.bool_)

            # While we haven't reached examples without liabilities.
            while (liability_dict[curr_example_index]):

                # Replace examples with nans (so we don't affect indexing).
                removed_examples[curr_example_index] = np.bool_(True)

                # Flag to check if removing example leads to misclassification.
                misclassified_flag: bool = False

                # For each example the removed example correctly classified.
                for classified_example_index in coverage_dict[
                    curr_example_index
                ]:

                    classified_example = dataset_np[classified_example_index]
                    actual_class: int = training_targets_np[
                        classified_example_index
                    ]

                    # Get the predicted class after noise reduction.
                    (
                        neighbor_indices,
                        noise_reduced_class
                    ) = self._predict_on_same_dataset(
                        classified_example,
                        dataset_np[~removed_examples],
                        training_targets_np[~removed_examples]
                    )

                    # If misclassify one of covered classes post removal, flag.
                    if actual_class != noise_reduced_class:
                        misclassified_flag = True
                        break

                # Insert removed example back at its old index.
                if misclassified_flag:
                    removed_examples[curr_example_index] = np.bool_(False)
                    curr_example_index += 1

                # If the removal didn't lead to a missclassifcation.
                else:

                    # For each example the removed example missclassified.
                    for liability_index in list(
                        liability_dict[curr_example_index]
                    ):

                        misclassified_example = dataset_np[liability_index]
                        actual_class = training_targets_np[liability_index]

                        # Get a new prediction for the previously wrong class.
                        (
                            neighbor_indices,
                            noise_reduced_class
                        ) = self._predict_on_same_dataset(
                            misclassified_example,
                            dataset_np,
                            training_targets_np
                        )

                        # If prediction is now correct after removing noise.
                        # We remove example as liability from liable examples.
                        if actual_class == noise_reduced_class:
                            for disim_index in dissimilarity_dict[
                                liability_index
                            ]:
                                if disim_index in liability_dict:
                                    liability_dict[disim_index].remove(
                                        liability_index)

                    del liability_dict[curr_example_index]

                    if len(liability_dict) == 0:
                        continue

                    # The size of liability sets of examples w/ liabilities.
                    lset_sizes = np.array(
                        [
                            (
                                key, len(liability_dict[key]))
                            for key in liability_dict.keys()
                        ]
                    )

                    # The indices of examples (sorted by desc liability size).
                    indxs_by_desc_lset_size = lset_sizes[
                        lset_sizes[:, 1].argsort()[
                            ::-1]][:, 0]

                    curr_example_index = indxs_by_desc_lset_size[0]

            dataset_np = dataset_np[~removed_examples]
            training_targets_np = training_targets_np[~removed_examples]

        return (dataset_np, training_targets_np, training_cols, scaler)

    def _build_bbnr_sets(
            self,
            training_set: np.ndarray,
            training_targets: np.ndarray
    ) -> tuple[dict, dict, dict]:
        '''Returns dicts: coverage, liability, and dissim sets of training_set.

        Keyword arguments:
        training_set -- the training dataset we are running BBNR on.
        training_targets -- the target labels / values of the training_set.
        '''

        # Each integer in these dicts represents index of a training example.

        # Each member of a set are examples that key example helps classify.
        coverage_dict: dict[int, set] = defaultdict(set)
        # Each member of a set are examples that key example helps misclassify.
        liability_dict: dict[int, set] = defaultdict(set)
        # Each member of set are neighbors of example that help misclassify it.
        dissimilarity_dict: dict[int, set] = defaultdict(set)

        example_index: int
        example: np.ndarray

        for example_index, example in enumerate(training_set):

            # Get the actual label and predicted label for this example.
            actual_label: float = training_targets[example_index]

            # Get the prediction for the example.
            neighbor_indices, predicted_label = self._predict_on_same_dataset(
                example, training_set, training_targets)

            neighbor_index: int

            # For each neighbor of the example.
            for neighbor_index in neighbor_indices:

                neighbor_label: float = training_targets[neighbor_index]

                # If prediction correct & neighbor helped, update coverage.
                if predicted_label == actual_label:
                    if neighbor_label == predicted_label:
                        coverage_dict[neighbor_index].add(example_index)

                # If prediction wrong & neighbor helped, update liab & dissim.
                elif neighbor_label != actual_label:
                    liability_dict[neighbor_index].add(example_index)
                    dissimilarity_dict[example_index].add(neighbor_index)

        # Return the coverage dictionaries containing the sets.
        return (coverage_dict, liability_dict, dissimilarity_dict)

    def _predict_on_same_dataset(
            self,
            example: np.ndarray,
            dataset: np.ndarray,
            dataset_targets: np.ndarray
    ) -> tuple[np.ndarray, float]:
        '''Returns neighbor_indices of example & kNN prediction (same dataset).

        Keyword arguments:
        example -- the example to run the kNN prediction for.
        dataset -- the dataset the example is a part of (not split from).
        dataset_targets -- targets vector for dataset.
        '''
        # Get closest neighbors (k+1 and [1:] to not include example itself).
        neighbor_indices: np.ndarray = self._get_k_nearest_neighbors(
            example, dataset, self.curr_k+1)[1:]

        if self.regressor_or_classifier == "classifier":
            # Get the prediction for the example.
            predicted_label: float = self._get_most_common_class(
                dataset_targets, neighbor_indices)
            return (neighbor_indices, predicted_label)
        else:
            # Return mean of corresponding target values.
            return (
                neighbor_indices,
                float(dataset_targets[neighbor_indices].mean())
            )

    def _agrees(self, prediction, neighbor_indices, dataset_targets):
        pass


# test = KNNHarness('regressor', 'datasets/abalone.data', 'Rings')
test = BBNRHarness('classifier', 'datasets/heart.data', 'num')
# test = KNNHarness('classifier', 'datasets/custom_cleveland.data', 'num')
print(test.evaluate())
