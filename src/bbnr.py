import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler  # type: ignore
from src.harness import KNNHarness
from collections import defaultdict
from sklearn.model_selection import KFold, StratifiedKFold  # type: ignore


class BBNRHarness(KNNHarness):
    def __init__(
        self,
        regressor_or_classifier: str,
        dataset_file_path: str,
        target_column_name: str,
        noise_level: float,
        missing_values: list[str] = ["?"],
        theta: float = 0.1,
        agree_func: str = "sd_neighbors",
    ):
        """Initialises a kNN Harness that applies BBNR to training data.

        Keyword arguments:
        regressor_or_classifier -- what kNN it runs 'regressor' | 'classifier'.
        dataset_file_path -- file path to the dataset to run the kNN on.
        target_column_name -- name of the column we are predicting.
        noise_level -- the percentage of the dataset to inject noise into.
        missing_values -- strings denoting missing values in the dataset.
        theta -- the tolerated difference for regression, multiplier if SD.
        agree_func -- how theta is calc'd, 'given' means theta is used as is.
        """

        super().__init__(
            regressor_or_classifier,
            dataset_file_path,
            target_column_name,
            noise_level,
            missing_values,
        )

        self._theta: float = theta
        self._agree_func: str = agree_func
        agree_func = "sd_neighbors"
        theta = 0.1
        self._curr_theta: float | None = theta
        self._curr_agree_func: str | None = agree_func
        self._best_agree_func: str | None = agree_func
        self._best_theta: float | None = theta
        self._agree_funcs: list[str] = ["sd_neighbors", "sd_whole"]
        self._sd_whole: float | None = None
        self._thetas: list[float] = [
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1,
            1.1,
            1.2,
            1.3,
            1.4,
            1.5,
            1.6,
            1.7,
            1.8,
            1.9,
            2,
            2.1,
            2.2,
            2.3,
            2.4,
            2.5,
        ]

    @property
    def curr_theta(self) -> float | None:
        """Getter for the curr_theta property."""
        return self._curr_theta

    @curr_theta.setter
    def curr_theta(self, theta: float | None) -> None:
        """Setter for the curr_theta property."""
        self._curr_theta = theta

    @property
    def curr_agree_func(self) -> str | None:
        """Getter for the curr_agree_func property."""
        return self._curr_agree_func

    @curr_agree_func.setter
    def curr_agree_func(self, agree_func: str | None) -> None:
        """Setter for the curr_agree_func property."""
        self._curr_agree_func = agree_func

    @property
    def agree_funcs(self) -> list[str]:
        """Getter for the agree_funcs property."""
        return self._agree_funcs

    @property
    def thetas(self) -> list[float]:
        """Getter for the thetas property."""
        return self._thetas

    @property
    def best_theta(self) -> float | None:
        """Getter for the best_theta property."""
        return self._best_theta

    @best_theta.setter
    def best_theta(self, theta: float | None) -> None:
        """Setter for the best_theta property."""
        self._best_theta = theta

    @property
    def best_agree_func(self) -> str | None:
        """Getter for the best_agree_func property."""
        return self._best_agree_func

    @best_agree_func.setter
    def best_agree_func(self, agree_func: str | None) -> None:
        """Setter for the best_agree_func property."""
        self._best_agree_func = agree_func

    @property
    def sd_whole(self) -> float | None:
        """Getter for the sd_whole property."""
        return self._sd_whole

    @sd_whole.setter
    def sd_whole(self, sd_whole: float) -> None:
        """Setter for the sd_whole property."""
        self._sd_whole = sd_whole

    def _preprocess_dataset(
        self,
        dataset: pd.DataFrame,
        training_targets: pd.Series,
        training_cols: pd.Index | None = None,
        scaler: StandardScaler | None = None,
    ) -> tuple[np.ndarray, np.ndarray, pd.Index, StandardScaler]:
        """Preprocesses data, returns cols & scaler used. BBNR if training.

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

        self.sd_whole = np.std(training_targets)

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

        # If it's a training set, apply BBNR to it & update dataset & targets.
        if is_training_set:
            # First, build the case-base competence model.
            coverage_dict: dict[int, set]
            liability_dict: dict[int, set]
            dissimilarity_dict: dict[int, set]

            (coverage_dict, liability_dict, dissimilarity_dict) = self._build_bbnr_sets(
                dataset_np, training_targets_np
            )

            # The size of the liability sets of each example with liabilities.
            lset_sizes: np.ndarray = np.array(
                [(key, len(liability_dict[key])) for key in liability_dict.keys()]
            )

            # If there aren't liabilities, return the data as is.
            if len(liability_dict) == 0:
                return (dataset_np, training_targets_np, training_cols, scaler)

            # The indices of the examples (sorted by desc liability size).
            indxs_by_desc_lset_size: np.ndarray = lset_sizes[
                lset_sizes[:, 1].argsort()[::-1]
            ][:, 0]

            # The current position we're @ in indxs_by_desc_lset_size.
            curr_lset_index: int = 0
            # The index of the example we're attempting to remove.
            curr_example_index: int = indxs_by_desc_lset_size[curr_lset_index]

            removed_examples: np.ndarray = np.zeros(dataset_np.shape[0], dtype=np.bool_)

            # While we haven't reached examples without liabilities.
            while liability_dict[curr_example_index]:
                # Replace examples with nans (so we don't affect indexing).
                removed_examples[curr_example_index] = np.bool_(True)

                # Flag to check if removing example leads to misprediction.
                misclassified_flag: bool = False

                # For each example the removed example correctly predicted.
                for classified_example_index in coverage_dict[curr_example_index]:
                    classified_example = dataset_np[classified_example_index]
                    actual_class: int = training_targets_np[classified_example_index]

                    # Get the predicted class after noise reduction.
                    (
                        neighbor_indices,
                        noise_reduced_class,
                    ) = self._predict_on_same_dataset(
                        classified_example,
                        classified_example_index,
                        dataset_np[~removed_examples],
                        training_targets_np[~removed_examples],
                    )

                    # If incorrect for covered class post removal, flag.
                    if not self._agrees(
                        actual_class,
                        noise_reduced_class,
                        neighbor_indices,
                        training_targets_np,
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
                    curr_example_index = indxs_by_desc_lset_size[curr_lset_index]

                # If the removal didn't lead to an incorrect prediction.
                else:
                    # For each example the removed example got wrong.
                    for liability_index in list(liability_dict[curr_example_index]):
                        misclassified_example = dataset_np[liability_index]
                        actual_class = training_targets_np[liability_index]

                        # Get a new prediction for the previously wrong class.
                        (
                            neighbor_indices,
                            noise_reduced_class,
                        ) = self._predict_on_same_dataset(
                            misclassified_example,
                            liability_index,
                            dataset_np[~removed_examples],
                            training_targets_np[~removed_examples],
                        )

                        # If prediction is now correct after removing noise.
                        # We remove example as liability from liable examples.
                        if self._agrees(
                            actual_class,
                            noise_reduced_class,
                            neighbor_indices,
                            training_targets_np,
                        ):
                            for disim_index in dissimilarity_dict[liability_index]:
                                if disim_index in liability_dict:
                                    liability_dict[disim_index].remove(liability_index)

                    del liability_dict[curr_example_index]

                    if len(liability_dict) == 0:
                        continue

                    # The size of liability sets of examples w/ liabilities.
                    lset_sizes = np.array(
                        [
                            (key, len(liability_dict[key]))
                            for key in liability_dict.keys()
                        ]
                    )

                    # The indices of examples (sorted by desc liability size).
                    indxs_by_desc_lset_size = lset_sizes[
                        lset_sizes[:, 1].argsort()[::-1]
                    ][:, 0]

                    # Go back to position 0 of the liability size array.
                    curr_lset_index = 0
                    # The curr_example_index is for the most liable example.
                    curr_example_index = indxs_by_desc_lset_size[curr_lset_index]

            dataset_np = dataset_np[~removed_examples]
            training_targets_np = training_targets_np[~removed_examples]

        return (dataset_np, training_targets_np, training_cols, scaler)

    def _build_bbnr_sets(
        self, training_set: np.ndarray, training_targets: np.ndarray
    ) -> tuple[dict, dict, dict]:
        """Returns dicts: coverage, liability, and dissim sets of training_set.

        Keyword arguments:
        training_set -- the training dataset we are running BBNR on.
        training_targets -- the target labels / values of the training_set.
        """

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
                example, example_index, training_set, training_targets
            )

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
                    neighbor_label, actual_label, neighbor_indices, training_targets
                ):
                    liability_dict[neighbor_index].add(example_index)
                    dissimilarity_dict[example_index].add(neighbor_index)

        if self.curr_agree_func == "sd_whole":
            self.sd_whole = np.std(training_targets)

        # Return the coverage dictionaries containing the sets.
        return (coverage_dict, liability_dict, dissimilarity_dict)

    def _predict_on_same_dataset(
        self,
        example: np.ndarray,
        example_index: int,
        dataset: np.ndarray,
        dataset_targets: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Returns neighbor_indices of example & kNN prediction (same dataset).

        Keyword arguments:
        example -- the example to run the kNN prediction for.
        example_index -- the index position of example.
        dataset -- the dataset the example is a part of (not split from).
        dataset_targets -- targets vector for dataset.
        """

        # Compute euclidean distances using vectorized operations.
        distances: np.ndarray = np.sqrt(np.sum((dataset - example) ** 2, axis=1))

        # Get indices of the k smallest distances
        neighbor_indices: np.ndarray = np.argsort(distances)[: self.curr_k + 1]

        # Remove the example itself.
        neighbor_indices = neighbor_indices[neighbor_indices != example_index]

        if self.regressor_or_classifier == "classifier":
            # Get the prediction for the example.
            predicted_label: float = self._get_most_common_class(
                dataset_targets, neighbor_indices
            )
            return (neighbor_indices, predicted_label)
        else:
            # Return mean of corresponding target values.
            return (neighbor_indices, float(dataset_targets[neighbor_indices].mean()))

    def _agrees(
        self,
        actual_value: float,
        prediction: float,
        neighbor_indices: np.ndarray,
        dataset_targets: np.ndarray,
    ) -> bool:
        """Returns whether an actual value agrees with its prediction.

        Keyword arguments:
        actual_value -- the real target value / class label.
        prediction -- the predicted target value / class label.
        neighbor_indices -- the indices of the example's neighbors.
        dataset_targets -- the target values / class labels for the dataset.
        """
        if self.regressor_or_classifier == "classifier":
            return actual_value == prediction
        else:
            if self.curr_theta is None:
                self.curr_theta = 2.5
            abs_diff: float = abs(actual_value - prediction)
            if self.curr_agree_func == "given":
                return abs_diff < self.curr_theta
            elif self.curr_agree_func == "sd_whole":
                if self.sd_whole is None:
                    self.sd_whole = 1
                return abs_diff < self.curr_theta * self.sd_whole
            elif self.curr_agree_func == "sd_neighbors":
                # Calculate the standard deviation of the neighbors.
                neighbor_targets = dataset_targets[neighbor_indices]
                std_dev_neighbors = np.std(neighbor_targets)
                return abs_diff < self.curr_theta * std_dev_neighbors
            raise ValueError("Invalid agree_func passed for BBNR!")

    def _get_best_k(
        self,
        dev_data: pd.DataFrame,
        dev_targets: pd.Series,
        candidate_k_values: list[int],
        best_avg_score: float | None = None,
        curr_best_k: int = 3,
        tried_k: set[int] = set(),
    ) -> int:
        """Returns best k found using 5-fold cross-validation.

        Keyword arguments:
        dev_data -- the unison of the training and validation datasets.
        dev_targets -- the target values associated with each row of dev_data.
        candidate_k_values -- the candidates for k currently being considered.
        best_avg_score -- the best average accuracy / MAE recorded so far.
        curr_best_k -- the best value for k recorded so far.
        tried_k -- the k values we have tried so far.
        """

        if 1 in candidate_k_values:
            candidate_k_values.remove(1)
            tried_k.add(1)

        # TODO: COMMENT FOR ACTUAL EXPERIMENTS
        # candidate_k_values = [9]

        # If this is the first call on the function.
        if best_avg_score is None:
            # If classifier set the init best_avg_score to negative infinity.
            if self.regressor_or_classifier == "classifier":
                best_avg_score = float("-inf")
            # If regressor set the init best_avg_score to positive infinity.
            else:
                best_avg_score = float("inf")

        if self.regressor_or_classifier == "classifier":
            kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            folds = list(kfold.split(dev_data, dev_targets))
        else:
            kfold = KFold(n_splits=5, shuffle=True, random_state=42)
            folds = list(kfold.split(dev_data))

        candidate_k: int

        if self.regressor_or_classifier == "regressor":
            # For each candidate k value
            for candidate_k in candidate_k_values:
                for agree_func in self.agree_funcs:
                    for theta in self.thetas:
                        curr_best_k, best_avg_score = self._get_best_k_helper(
                            curr_best_k,
                            best_avg_score,
                            folds,
                            dev_data,
                            dev_targets,
                            candidate_k,
                            5,
                            agree_func,
                            theta,
                        )
        else:
            for candidate_k in candidate_k_values:
                curr_best_k, best_avg_score = self._get_best_k_helper(
                    curr_best_k,
                    best_avg_score,
                    folds,
                    dev_data,
                    dev_targets,
                    candidate_k,
                    5,
                )

        # Update tried_k with the candidate_k_values we just tried.
        tried_k.update(candidate_k_values)

        new_candidates: list[int]

        # Get a new list of candidate k values.
        new_candidates = self._expand_k_search_space(
            candidate_k_values, curr_best_k, tried_k
        )

        # TODO: COMMENT FOR ACTUAL EXPERIMENTS
        # new_candidates = [9]

        # If empty list or new_candidates just has curr_best_k end grid search.
        if (
            not new_candidates
            or new_candidates == [curr_best_k]
            or new_candidates[-1] > (len(dev_data) * 0.8)
        ):
            self.curr_theta = self.best_theta
            self.curr_agree_func = self.best_agree_func
            return curr_best_k

        # Recursive call w/ new candidates.
        curr_best_k = self._get_best_k(
            dev_data,
            dev_targets,
            new_candidates,
            best_avg_score,
            curr_best_k,
            tried_k,
        )

        self.curr_theta = self.best_theta
        self.curr_agree_func = self.best_agree_func

        return curr_best_k

    def _get_best_k_helper(
        self,
        curr_best_k: int,
        best_avg_score: float,
        folds: list,
        dev_data: pd.DataFrame,
        dev_targets: pd.Series,
        candidate_k: int,
        n_splits: int,
        agree_func: str | None = None,
        theta: float | None = None,
    ):
        """Helper function for _get_best_k

        Keyword arguments:
        curr_best_k -- the best value for k recorded so far.
        best_avg_score -- the best average accuracy / MAE recorded so far.
        folds -- the train/dev fold provided by the KFold splitter.
        dev_data -- the unison of the training and validation datasets.
        dev_targets -- the target values associated with each row of dev_data.
        candidate_k -- the candidate k currently being considered.
        n_splits -- the number of k fold splits.
        agree_func -- the agreement function to use for BBNR regression.
        theta -- the theta / tolerance to use for BBNR regression.
        """
        self.curr_agree_func = agree_func
        self.curr_theta = theta
        # total_score is either total MAE or total accuracy.

        total_score: float = 0.0

        train_idx: np.ndarray
        val_idx: np.ndarray
        train_targets: pd.Series
        val_targets: pd.Series
        train_targets_np: np.ndarray

        # Each fold, train model w/ 4 folds & val w/ remaining.
        for train_idx, val_idx in folds:
            train_data: pd.DataFrame
            val_data: pd.DataFrame

            # Split data
            train_data, val_data = (
                dev_data.iloc[train_idx].copy(),
                dev_data.iloc[val_idx].copy(),
            )

            train_data.reset_index(drop=True, inplace=True)
            val_data.reset_index(drop=True, inplace=True)

            train_targets, val_targets = (
                dev_targets.iloc[train_idx].copy(),
                dev_targets.iloc[val_idx].copy(),
            )

            train_targets = self._introduce_artificial_noise(
                train_targets,
                self.noise_level,
            )

            train_targets.reset_index(drop=True, inplace=True)
            val_targets.reset_index(drop=True, inplace=True)

            train_data_scaled: np.ndarray
            val_data_scaled: np.ndarray
            training_cols: pd.Index
            scaler: StandardScaler

            self.curr_k = candidate_k

            # Preprocess training data and transform val data.
            (
                train_data_scaled,
                train_targets_np,
                training_cols,
                scaler,
            ) = self._preprocess_dataset(train_data, train_targets)

            val_data_scaled, _, _, _ = self._preprocess_dataset(
                val_data,
                train_targets,
                training_cols=training_cols,
                scaler=scaler,
            )

            score: float

            # Get accuracy for this fold if its a classifier.
            if self.regressor_or_classifier == "classifier":
                score = self._get_accuracy_of_knn_classifier(
                    candidate_k,
                    train_data_scaled,
                    val_data_scaled,
                    train_targets_np,
                    val_targets,
                )

            # Get MAE for this fold if its a regressor.
            else:
                score = self._get_mae_of_knn_regressor(
                    candidate_k,
                    train_data_scaled,
                    val_data_scaled,
                    train_targets_np,
                    val_targets,
                )

            # Add the score for this fold to the total_score.
            total_score += score

        # Get the avg_score by dividing by the number of folds.
        avg_score: float = total_score / n_splits

        # If better than best, update best_score and best_k.
        if self.regressor_or_classifier == "classifier":
            # If classification avg_score is better if higher.
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                curr_best_k = candidate_k
        else:
            # If regression avg_score is better if lower.
            if avg_score < best_avg_score:
                best_avg_score = avg_score
                curr_best_k = candidate_k
                self.best_theta = theta
                self.best_agree_func = agree_func

        if best_avg_score is None:
            # If classifier set the init best_avg_score to negative infinity.
            if self.regressor_or_classifier == "classifier":
                best_avg_score = float("-inf")
            # If regressor set the init best_avg_score to positive infinity.
            else:
                best_avg_score = float("inf")

        return curr_best_k, best_avg_score


# test = BBNRHarness("regressor", "datasets/regression/abalone.data", "Rings")
# test = BBNRHarness("regressor", "datasets/regression/student_portugese.data", "G3")
# test = BBNRHarness("classifier", "datasets/classification/iris.data", "class", 0.5)
# print(test.evaluate())
