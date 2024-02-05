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
            missing_values: list[str] = ['?'],
            theta: float = 2,
            agree_func: str = "sd_neighbors",
            kind_sel: str = "majority"
    ):
        '''Initialises a kNN Harness that applies BBNR to training data.

        Keyword arguments:
        regressor_or_classifier -- what kNN it runs 'regressor' | 'classifier'.
        dataset_file_path -- file path to the dataset to run the kNN on.
        target_column_name -- name of the column we are predicting.
        missing_values -- strings denoting missing values in the dataset.
        theta -- the tolerated difference for regression, multiplier if SD.
        agree_func -- how theta is calc'd, 'given' means theta is used as is.
        kind_sel -- determines proportion of neighbors that must fall in theta.
        '''

        self._theta = theta
        self._agree_func = agree_func
        self._kind_sel = kind_sel
        super().__init__(regressor_or_classifier, dataset_file_path,
                         target_column_name, missing_values)

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
        if is_training_set:

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

            # The current position we're @ in indxs_by_desc_lset_size.
            curr_lset_index: int = 0
            # The index of the example we're attempting to remove.
            curr_example_index: int = indxs_by_desc_lset_size[curr_lset_index]

            removed_examples: np.ndarray = np.zeros(
                dataset_np.shape[0], dtype=np.bool_)

            # While we haven't reached examples without liabilities.
            while (liability_dict[curr_example_index]):

                # Replace examples with nans (so we don't affect indexing).
                removed_examples[curr_example_index] = np.bool_(True)

                # Flag to check if removing example leads to misprediction.
                misclassified_flag: bool = False

                # For each example the removed example correctly predicted.
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
                        classified_example_index,
                        dataset_np[~removed_examples],
                        training_targets_np[~removed_examples]
                    )

                    # If incorrect for covered class post removal, flag.
                    if not self._agrees(
                        actual_class,
                        noise_reduced_class,
                        neighbor_indices,
                        training_targets_np
                    ):
                        misclassified_flag = True
                        break

                # Insert removed example back at its old index.
                if misclassified_flag:
                    removed_examples[curr_example_index] = np.bool_(False)
                    # Move to next biggest liability.
                    curr_lset_index += 1
                    # If reached the end of the liability sizes array, break.
                    if curr_lset_index == len(indxs_by_desc_lset_size):
                        break
                    curr_example_index = indxs_by_desc_lset_size[
                        curr_lset_index
                    ]

                # If the removal didn't lead to an incorrect prediction.
                else:

                    # For each example the removed example got wrong.
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
                            liability_index,
                            dataset_np[~removed_examples],
                            training_targets_np[~removed_examples]
                        )

                        # If prediction is now correct after removing noise.
                        # We remove example as liability from liable examples.
                        if self._agrees(
                            actual_class,
                            noise_reduced_class,
                            neighbor_indices,
                            training_targets_np,
                        ):
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
                    
                    # Go back to position 0 of the liability size array.
                    curr_lset_index = 0
                    # The curr_example_index is for the most liable example.
                    curr_example_index = indxs_by_desc_lset_size[
                        curr_lset_index
                    ]

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

        # Each member of a set are examples that key example helps predict.
        coverage_dict: dict[int, set] = defaultdict(set)
        # Each member of a set are examples that key example helps mispredict.
        liability_dict: dict[int, set] = defaultdict(set)
        # Each member of set are neighbors of example that help mispredict it.
        dissimilarity_dict: dict[int, set] = defaultdict(set)

        example_index: int
        example: np.ndarray

        for example_index, example in enumerate(training_set):

            # Get the actual label and predicted label for this example.
            actual_label: float = training_targets[example_index]

            # Get the prediction for the example.
            neighbor_indices, predicted_label = self._predict_on_same_dataset(
                example, example_index, training_set, training_targets)

            neighbor_index: int

            # For each neighbor of the example.
            for neighbor_index in neighbor_indices:

                neighbor_label: float = training_targets[neighbor_index]

                # If prediction correct & neighbor helped, update coverage.
                if self._agrees(
                    actual_label,
                    predicted_label,
                    neighbor_indices,
                    training_targets,
                ):
                    if self._agrees(
                        neighbor_label,
                        predicted_label,
                        neighbor_indices,
                        training_targets,
                    ):
                        coverage_dict[neighbor_index].add(example_index)

                # If prediction wrong & neighbor helped, update liab & dissim.
                elif not self._agrees(
                    neighbor_label,
                    actual_label,
                    neighbor_indices,
                    training_targets
                ):
                    liability_dict[neighbor_index].add(example_index)
                    dissimilarity_dict[example_index].add(neighbor_index)

        if self.agree_func == "sd_whole":
            self.sd_whole = np.std(training_targets)

        # Return the coverage dictionaries containing the sets.
        return (coverage_dict, liability_dict, dissimilarity_dict)

    def _predict_on_same_dataset(
            self,
            example: np.ndarray,
            example_index: int,
            dataset: np.ndarray,
            dataset_targets: np.ndarray
    ) -> tuple[np.ndarray, float]:
        '''Returns neighbor_indices of example & kNN prediction (same dataset).

        Keyword arguments:
        example -- the example to run the kNN prediction for.
        example_index -- the index position of example.
        dataset -- the dataset the example is a part of (not split from).
        dataset_targets -- targets vector for dataset.
        '''

        # Compute euclidean distances using vectorized operations.
        distances: np.ndarray = np.sqrt(
            np.sum((dataset - example)**2, axis=1))

        # Get indices of the k smallest distances
        neighbor_indices: np.ndarray = np.argsort(distances)[:self.curr_k+1]

        # Remove the example itself.
        neighbor_indices = neighbor_indices[neighbor_indices != example_index]

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

    def _agrees(
        self,
        actual_value: float,
        prediction: float,
        neighbor_indices: np.ndarray,
        dataset_targets: np.ndarray
    ) -> bool:
        '''Returns whether an actual value agrees with its prediction.

        Keyword arguments:
        actual_value -- the real target value / class label.
        prediction -- the predicted target value / class label.
        neighbor_indices -- the indices of the example's neighbors.
        dataset_targets -- the target values / class labels for the dataset. 
        '''
        if self.regressor_or_classifier == "classifier":
            return actual_value == prediction
        else:
            abs_diff: float = abs(actual_value - prediction)
            if self.agree_func == "given":
                return abs_diff < self.theta
            elif self.agree_func == "sd_whole":
                return abs_diff < self.theta * self.sd_whole
            elif self.agree_func == "sd_neighbors":
                # Calculate the standard deviation of the neighbors.
                neighbor_targets = dataset_targets[neighbor_indices]
                std_dev_neighbors = np.std(neighbor_targets)
                return abs_diff < self.theta * std_dev_neighbors
            raise ValueError("Invalid agree_func passed for BBNR!")


# test = BBNRHarness('regressor', 'datasets/abalone.data', 'Rings')
test = BBNRHarness('classifier', 'datasets/heart.data', 'num')
print(test.evaluate())
