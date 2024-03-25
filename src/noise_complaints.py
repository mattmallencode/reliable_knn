import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler  # type: ignore
from src.harness import KNNHarness
from sklearn.model_selection import KFold, StratifiedKFold  # type: ignore
from collections import defaultdict
import sys


class NoiseComplaintsHarness(KNNHarness):
    def __init__(
        self,
        regressor_or_classifier: str,
        dataset_file_path: str,
        target_column_name: str,
        noise_level: float,
        missing_values: list[str] = ["?"],
    ):
        """Initialises a NCkNN harness.

        Keyword arguments:
        regressor_or_classifier -- what kNN it runs 'regressor' | 'classifier'
        dataset_file_path -- file path to the dataset to run the kNN on.
        target_column_name -- name of the column we are predicting.
        noise_level -- fraction of examples in training / val to make noisy.
        missing_values -- strings denoting missing values in the dataset.
        """

        super().__init__(
            regressor_or_classifier,
            dataset_file_path,
            target_column_name,
            noise_level,
            missing_values,
        )
        self._curr_theta: float | None = 0.5
        self._curr_lambda_s: float = 0
        self._curr_lambda_p: float = 0
        self._curr_agree_func: str | None = "sd_neighbors"
        self._best_agree_func: str | None = self._curr_agree_func
        self._best_theta: float | None = self._curr_theta
        self._best_lambda_s: float | None = self._curr_lambda_s
        self._best_lambda_p: float | None = self._curr_lambda_p
        self._agree_funcs: list[str] = ["sd_neighbors", "sd_whole"]
        self._thetas: list[float] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        self._lambda_s_candidates: list[float]

        if self.regressor_or_classifier == "classifier":
            self._lambda_s_candidates = [
                0.0,
                0.05,
                0.1,
                0.15,
                0.2,
                0.25,
                0.3,
                0.35,
                0.4,
                0.45,
                0.5,
                0.55,
                0.6,
                0.65,
                0.7,
                0.75,
                0.8,
                0.85,
                0.9,
                0.95,
                1.0,
            ]
        else:
            self._lambda_s_candidates = [
                0.0,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                1.0,
            ]

        self._lambda_p_candidates: list[float]

        if self.regressor_or_classifier == "classifier":
            self._lambda_p_candidates = [
                -1,
                0.0,
                0.05,
                0.1,
                0.15,
                0.2,
                0.25,
                0.3,
                0.35,
                0.4,
                0.45,
                0.5,
                0.55,
                0.6,
                0.65,
                0.7,
                0.75,
                0.8,
                0.85,
                0.9,
                0.95,
                1.0,
            ]
        else:
            self._lambda_p_candidates = [
                -1,
                0.0,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                1.0,
            ]

        self._best_avg_score: float | None = None
        self._similarities: np.ndarray | None = None
        self._similarity_scaler: StandardScaler | None = None
        self._reliability_scaler: StandardScaler | None = None
        self._reliabilities: np.ndarray | None = None
        self._sd_whole: float = 0

    @property
    def curr_theta(self) -> float | None:
        """Getter for the curr_theta property."""
        return self._curr_theta

    @curr_theta.setter
    def curr_theta(self, theta: float | None) -> None:
        """Setter for the curr_theta property."""
        self._curr_theta = theta

    @property
    def best_avg_score(self) -> float | None:
        """Getter for the best_avg_score property."""
        return self._best_avg_score

    @best_avg_score.setter
    def best_avg_score(self, score: float | None) -> None:
        """Setter for the best_avg_score property."""
        self._best_avg_score = score

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
    def lambda_s_candidates(self) -> list[float]:
        """Getter for the lambda_s_candidates property."""
        return self._lambda_s_candidates

    @lambda_s_candidates.setter
    def lambda_s_candidates(self, lambda_s) -> None:
        """Setter for the lambda_s_candidates property."""
        self._lambda_s_candidates = lambda_s

    @property
    def lambda_p_candidates(self) -> list[float]:
        """Getter for the lambda_p_candidates property."""
        return self._lambda_p_candidates

    @lambda_p_candidates.setter
    def lambda_p_candidates(self, lambda_p) -> None:
        """Setter for the lambda_p_candidates property."""
        self._lambda_p_candidates = lambda_p

    @property
    def best_theta(self) -> float | None:
        """Getter for the best_theta property."""
        return self._best_theta

    @best_theta.setter
    def best_theta(self, theta: float | None) -> None:
        """Setter for the best_theta property."""
        self._best_theta = theta

    @property
    def best_lambda_s(self) -> float | None:
        """Getter for the best_lambda_s property."""
        return self._best_lambda_s

    @best_lambda_s.setter
    def best_lambda_s(self, lambda_s_value: float | None) -> None:
        """Setter for the best_lambda_s property."""
        self._best_lambda_s = lambda_s_value

    @property
    def best_lambda_p(self) -> float | None:
        """Getter for the best_lambda_p property."""
        return self._best_lambda_p

    @best_lambda_p.setter
    def best_lambda_p(self, lambda_p_value: float | None) -> None:
        """Setter for the best_lambda_p property."""
        self._best_lambda_p = lambda_p_value

    @property
    def best_agree_func(self) -> str | None:
        """Getter for the best_agree_func property."""
        return self._best_agree_func

    @best_agree_func.setter
    def best_agree_func(self, agree_func: str | None) -> None:
        """Setter for the best_agree_func property."""
        self._best_agree_func = agree_func

    @property
    def similarities(self) -> np.ndarray | None:
        """Getter for the similarities property."""
        return self._similarities

    @similarities.setter
    def similarities(self, sim: np.ndarray | None) -> None:
        """Setter for the similarities property."""
        self._similarities = sim

    @property
    def similarity_scaler(self) -> StandardScaler | None:
        """Getter for the similarity_scaler property."""
        return self._similarity_scaler

    @similarity_scaler.setter
    def similarity_scaler(self, scaler: StandardScaler | None) -> None:
        """Setter for the similarity_scaler property."""
        self._similarity_scaler = scaler

    @property
    def reliability_scaler(self) -> StandardScaler | None:
        """Getter for the reliability_scaler property."""
        return self._reliability_scaler

    @reliability_scaler.setter
    def reliability_scaler(self, scaler: StandardScaler | None) -> None:
        """Setter for the reliability_scaler property."""
        self._reliability_scaler = scaler

    @property
    def reliabilities(self) -> np.ndarray | None:
        """Getter for the reliabilities property."""
        return self._reliabilities

    @reliabilities.setter
    def reliabilities(self, rel: np.ndarray | None) -> None:
        """Setter for the reliabilities property."""
        self._reliabilities = rel

    @property
    def curr_lambda_s(self) -> float:
        """Getter for the curr_lambda_s property."""
        return self._curr_lambda_s

    @curr_lambda_s.setter
    def curr_lambda_s(self, lambda_s_value: float) -> None:
        """Setter for the curr_lambda_s property."""
        self._curr_lambda_s = lambda_s_value

    @property
    def curr_lambda_p(self) -> float:
        """Getter for the curr_lambda_p property."""
        return self._curr_lambda_p

    @curr_lambda_p.setter
    def curr_lambda_p(self, lambda_p_value: float) -> None:
        """Setter for the curr_lambda_p property."""
        self._curr_lambda_p = lambda_p_value

    @property
    def sd_whole(self) -> float:
        """Getter for the sd_whole property."""
        return self._sd_whole

    @sd_whole.setter
    def sd_whole(self, sd_whole: float) -> None:
        """Setter for the sd_whole property."""
        self._sd_whole = sd_whole

    def _get_k_nearest_neighbors(
        self, example_to_get_neighbors_of: np.ndarray, dataset: np.ndarray, k: int = 3
    ) -> np.ndarray:
        """Returns k nearest neighbors of an example (as an array of indices).

        Keyword arguments:
        example_to_get_neighbors_of -- the example we are interested in.
        dataset -- the example to get the nearest neighbors from.
        k -- the number of nearest neighbors to fetch.
        """

        # Compute euclidean distances using vectorized operations.
        distances: np.ndarray = np.sqrt(
            np.sum((dataset - example_to_get_neighbors_of) ** 2, axis=1)
        )

        # Invert the distances to get similarities, avoiding division by zero
        distances = distances + sys.float_info.epsilon
        similarities = 1.0 / distances

        similarities = similarities.reshape(-1, 1)

        # Scale these similarities using the precomputed scaler
        if self.similarity_scaler is not None:
            scaled_similarities = self.similarity_scaler.transform(similarities)
        else:
            raise TypeError("similarity_scaler is None!")

        if self.reliabilities is not None:
            influence_scores = (self.curr_lambda_s * scaled_similarities) + (
                (1 - self.curr_lambda_s) * self.reliabilities
            )
        else:
            raise TypeError("reliabilities ndarray is None!")

        # Get indices of the k highest influence scores (descending order).
        indices = np.argsort(-(influence_scores.flatten()))[:k]

        self.similarities = scaled_similarities

        return indices

    def _knn_classifier(
        self,
        example_to_predict: np.ndarray,
        dataset: np.ndarray,
        target_column: np.ndarray,
        k: int = 3,
    ) -> float:
        """Predicts the class label of an example using influence weights.

        Keyword arguments:
        example_to_predict -- the example we are running the classification on.
        dataset -- the dataset to get the nearest neighbors from.
        target_column -- column with the class labels of the examples in dataset.
        k -- the number of closest neighbors to use in the mode calculation.
        """

        # If lambda_p is -1, do normal kNN classification.
        if self.curr_lambda_p == -1:
            return super()._knn_classifier(
                example_to_predict, dataset, target_column, k
            )

        indices = self._get_k_nearest_neighbors(example_to_predict, dataset, k)

        if self.similarities is None:
            raise TypeError("similarities is None when running classifier")

        if self.reliabilities is None:
            raise TypeError("reliabilities is None when running classifier")

        # Calc weights for each neighbor based on influence.
        weights = (self.curr_lambda_p * self.similarities[indices]) + (
            (1 - self.curr_lambda_p) * self.reliabilities[indices]
        )

        # Calculate weighted frequency of each class in k neighbors.
        class_weights: dict[float, float] = {}
        for index, weight in zip(indices, weights.flatten()):
            label = target_column[index]
            class_weights[label] = class_weights.get(label, 0) + weight

        # Determine the class with the highest weighted frequency
        most_frequent = max(class_weights, key=lambda x: class_weights[x])

        return most_frequent

    def _knn_regressor(
        self,
        example_to_predict: np.ndarray,
        dataset: np.ndarray,
        target_column: np.ndarray,
        k: int = 3,
    ) -> float:
        """Predicts the target value of an example using influence weights.

        Keyword arguments:
        example_to_predict -- the example we are running the regression on.
        dataset -- the dataset to get the nearest neighbors from.
        target_column -- column with target values of examples in the dataset.
        k -- the number of closest neighbors to use in weighted mean calculation.
        """

        # If lambda_p is -1, do normal kNN regression.
        if self.curr_lambda_p == -1:
            return super()._knn_regressor(example_to_predict, dataset, target_column, k)

        if self.similarities is None:
            raise TypeError("similarities is None when running classifier")

        if self.reliabilities is None:
            raise TypeError("reliabilities is None when running classifier")

        indices = self._get_k_nearest_neighbors(example_to_predict, dataset, k)

        # Calc weights for each neighbor based on similarity & reliability.
        weights = (self.curr_lambda_p * self.similarities[indices]) + (
            (1 - self.curr_lambda_p) * self.reliabilities[indices]
        )

        # Check if the sum of weights is zero to avoid division by zero.

        sum_weights = np.sum(weights)
        if sum_weights == 0:
            # If sum of weights is zero, distribute equal weights among all examples.
            num_neighbors = len(indices)
            weights = np.full_like(weights, fill_value=1.0 / num_neighbors)

        # Compute the weighted mean of the corresponding target values.
        weighted_mean = np.sum(weights.flatten() * target_column[indices]) / np.sum(
            weights
        )

        return float(weighted_mean)

    def _precompute_and_scale(
        self, dataset: np.ndarray, dataset_targets: np.ndarray
    ) -> None:
        """Precomputes dataset reliabilities + NxN similarity, then scales.

        Keyword arguments:
        dataset -- training / development set.
        dataset_targets -- training / development example targets.
        """
        self._precompute_similarities(dataset)
        self._precompute_reliabilities(dataset, dataset_targets)
        self._scale_similarity_and_reliability()

    def _scale_similarity_and_reliability(self):
        """Fits scaler to similarity scores; fits & transforms reliability."""
        self.similarity_scaler = StandardScaler()
        self.reliability_scaler = StandardScaler()
        self.similarity_scaler = self.similarity_scaler.fit(self.similarities)
        self.reliabilities = self.reliability_scaler.fit_transform(self.reliabilities)

    def _precompute_similarities(self, dataset: np.ndarray):
        """Computes pairwise inverted euclid for all examples in dataset."""
        # Compute the pairwise squared differences between all points.
        squared_diff = (
            np.sum(dataset**2, axis=1, keepdims=True)
            - 2 * dataset @ dataset.T
            + np.sum(dataset**2, axis=1)
        )

        # Compute the pairwise Euclidean distances (sqrt of squared diffs).
        distances = np.sqrt(np.maximum(squared_diff, 0))
        distances = distances + sys.float_info.epsilon

        # Invert the distances to compute similarities.
        self.similarities = 1.0 / distances

        if self.similarities is not None:
            # Set the diagonal to the highest similarity.
            np.fill_diagonal(self.similarities, 1.0)
            self.similarities = self.similarities.reshape(-1, 1)

    def _precompute_reliabilities(
        self, dataset: np.ndarray, dataset_targets: np.ndarray
    ):
        """Computes the reliability of each example in dataset.

        Keyword arguments:
        dataset -- training / development set.
        dataset_targets -- training / development example targets.
        """

        # Initialize the coverage dict using the logic provided
        coverage_dict = self._build_coverage_sets(dataset, dataset_targets)

        # Initialize reliabilities array with zeros
        num_examples = len(dataset)
        self.reliabilities = np.zeros(num_examples)

        # Populate the reliabilities array
        for example_index, coverage_set in coverage_dict.items():
            # The reliability of an example is the length of its coverage set
            self.reliabilities[example_index] = len(coverage_set)

        self.reliabilities = self.reliabilities.reshape(-1, 1)

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
                return abs_diff < self.curr_theta * self.sd_whole
            elif self.curr_agree_func == "sd_neighbors":
                # Calculate the standard deviation of the neighbors.
                neighbor_targets = dataset_targets[neighbor_indices]
                std_dev_neighbors = np.std(neighbor_targets)
                return abs_diff < self.curr_theta * std_dev_neighbors
            raise ValueError("Invalid agree_func passed for BBNR!")

    def _build_coverage_sets(
        self, training_set: np.ndarray, training_targets: np.ndarray
    ) -> dict[int, set]:
        """Returns dicts of coverage sets for training_set.

        Keyword arguments:
        training_set -- the training dataset we are running BBNR on.
        training_targets -- the target labels / values of the training_set.
        """

        # Each integer in the dict represents index of a training example.

        # Each member of a set are examples that key example helps predict.
        coverage_dict: dict[int, set] = defaultdict(set)

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

        if self.curr_agree_func == "sd_whole":
            self.sd_whole = np.std(training_targets)

        # Return the coverage dict containing the sets.
        return coverage_dict

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

        if is_training_set:
            self._precompute_and_scale(dataset_np, training_targets_np)

        return (dataset_np, training_targets_np, training_cols, scaler)

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

        if self.regressor_or_classifier == "classifier":
            kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            folds = list(kfold.split(dev_data, dev_targets))
        else:
            kfold = KFold(n_splits=5, shuffle=True, random_state=42)
            folds = list(kfold.split(dev_data))

        best_k = self._get_best_k_helper(
            dev_data,
            dev_targets,
            candidate_k_values,
            folds,
            best_avg_score,
            curr_best_k,
            tried_k,
        )

        with open(f"results/{self.dataset_file_path.split('/')[-1]}_lambda", "a") as f:
            print(
                f"LAMBDA_S: {self.best_lambda_s}, LAMBDA_P: {self.best_lambda_p}\n",
                file=f,
            )

        return best_k

    def _get_best_k_helper(
        self,
        dev_data: pd.DataFrame,
        dev_targets: pd.Series,
        candidate_k_values: list[int],
        folds: list,
        best_avg_score: float | None = None,
        curr_best_k: int = 3,
        tried_k: set[int] = set(),
    ) -> int:
        """Helper function for _get_best_k

        Keyword arguments:
        dev_data -- the unison of the training and validation datasets.
        dev_targets -- the target values associated with each row of dev_data.
        candidate_k_values -- the candidates for k currently being considered.
        folds -- the folds provided by the KFold object.
        best_avg_score -- the best average accuracy / MAE recorded so far.
        curr_best_k -- the best value for k recorded so far.
        tried_k -- the k values we have tried so far.
        """
        if 1 in candidate_k_values:
            candidate_k_values.remove(1)
            tried_k.add(1)

        # TODO: COMMENT FOR ACTUAL EXPERIMENTS
        # candidate_k_values = [9]

        candidate_k: int

        if self.regressor_or_classifier == "regressor":
            # For each candidate k value
            for candidate_k in candidate_k_values:
                for lambda_s_value in self.lambda_s_candidates:
                    self.curr_lambda_s = lambda_s_value
                    for lambda_p_value in self.lambda_p_candidates:
                        self.curr_lambda_p = lambda_p_value
                        for agree_func in self.agree_funcs:
                            for theta in self.thetas:
                                curr_best_k, best_avg_score = (
                                    self._get_best_k_sub_helper(
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
                                )
        else:
            for candidate_k in candidate_k_values:
                for lambda_s_value in self.lambda_s_candidates:
                    self.curr_lambda_s = lambda_s_value
                    for lambda_p_value in self.lambda_p_candidates:
                        self.curr_lambda_p = lambda_p_value
                        curr_best_k, best_avg_score = self._get_best_k_sub_helper(
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

        if self.best_lambda_s is None:
            raise TypeError("best_lambda_s is None when running _get_best_k")

        if self.best_lambda_p is None:
            raise TypeError("best_lambda_p is None when running _get_best_k")

        # If empty list or new_candidates just has curr_best_k end grid search.
        if (
            not new_candidates
            or new_candidates == [curr_best_k]
            or new_candidates[-1] > (len(dev_data) * 0.8)
        ):
            self.curr_theta = self.best_theta
            self.curr_lambda_s = self.best_lambda_s
            self.curr_lambda_p = self.best_lambda_p
            self.curr_agree_func = self.best_agree_func
            return curr_best_k

        # Recursive call w/ new candidates.
        curr_best_k = self._get_best_k_helper(
            dev_data,
            dev_targets,
            new_candidates,
            folds,
            best_avg_score,
            curr_best_k,
            tried_k,
        )

        self.curr_theta = self.best_theta
        self.curr_lambda_s = self.best_lambda_s
        self.curr_lambda_p = self.best_lambda_p
        self.curr_agree_func = self.best_agree_func

        return curr_best_k

    def _get_best_k_sub_helper(
        self,
        curr_best_k: int,
        best_avg_score: float | None,
        folds: list,
        dev_data: pd.DataFrame,
        dev_targets: pd.Series,
        candidate_k: int,
        n_splits: int,
        agree_func: str | None = None,
        theta: float | None = None,
    ):
        """Helper function for _get_best_k. Bit redudnant; may fix up later.

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

        # If this is the first call on the function.
        if best_avg_score is None:
            # If classifier set the init best_avg_score to negative infinity.
            if self.regressor_or_classifier == "classifier":
                best_avg_score = float("-inf")
            # If regressor set the init best_avg_score to positive infinity.
            else:
                best_avg_score = float("inf")

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

            train_targets.reset_index(drop=True, inplace=True)
            val_targets.reset_index(drop=True, inplace=True)

            train_targets = self._introduce_artificial_noise(
                train_targets,
                self.noise_level,
            )

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
                self.best_avg_score = best_avg_score
                curr_best_k = candidate_k
                self.best_lambda_s = self.curr_lambda_s
                self.best_lambda_p = self.curr_lambda_p
        else:
            # If regression avg_score is better if lower.
            if avg_score < best_avg_score:
                best_avg_score = avg_score
                self.best_avg_score = best_avg_score
                curr_best_k = candidate_k
                self.best_theta = theta
                self.best_lambda_s = self.curr_lambda_s
                self.best_lambda_p = self.curr_lambda_p
                self.best_agree_func = agree_func

        return curr_best_k, best_avg_score
