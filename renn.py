from imblearn.under_sampling import (  # type: ignore
    RepeatedEditedNearestNeighbours,  # type: ignore
)  # type: ignore
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler  # type: ignore
from harness import KNNHarness
from sklearn.model_selection import KFold, StratifiedKFold  # type: ignore


class RENNHarness(KNNHarness):
    def __init__(
        self,
        regressor_or_classifier: str,
        dataset_file_path: str,
        target_column_name: str,
        missing_values: list[str] = ["?"],
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

        agree_func: str = "sd_neighbors"
        kind_sel: str = "all"
        theta = 0.5
        self._curr_theta: float | None = theta
        self._curr_agree_func: str | None = agree_func
        self._curr_kind_sel: str | None = kind_sel
        self._best_agree_func: str | None = agree_func
        self._best_kind_sel: str | None = kind_sel
        self._best_theta: float | None = theta
        self._agree_funcs = ["sd_neighbors", "sd_whole"]
        self._thetas = [0.5, 1, 1.5, 2, 2.5]
        self._kind_sels = ["mode", "all"]

    @property
    def curr_theta(self) -> float | None:
        return self._curr_theta

    @curr_theta.setter
    def curr_theta(self, theta: float | None) -> None:
        self._curr_theta = theta

    @property
    def curr_agree_func(self) -> str | None:
        return self._curr_agree_func

    @curr_agree_func.setter
    def curr_agree_func(self, agree_func: str | None) -> None:
        self._curr_agree_func = agree_func

    @property
    def curr_kind_sel(self) -> str | None:
        return self._curr_kind_sel

    @curr_kind_sel.setter
    def curr_kind_sel(self, curr_kind_sel: str | None) -> None:
        self._curr_kind_sel = curr_kind_sel

    @property
    def agree_funcs(self) -> list[str]:
        return self._agree_funcs

    @property
    def thetas(self) -> list[float]:
        return self._thetas

    @property
    def best_theta(self) -> float | None:
        return self._best_theta

    @best_theta.setter
    def best_theta(self, theta: float | None) -> None:
        self._best_theta = theta

    @property
    def best_agree_func(self) -> str | None:
        return self._best_agree_func

    @best_agree_func.setter
    def best_agree_func(self, agree_func: str | None) -> None:
        self._best_agree_func = agree_func

    @property
    def best_kind_sel(self) -> str | None:
        return self._best_kind_sel

    @best_kind_sel.setter
    def best_kind_sel(self, kind_sel: str | None) -> None:
        self._best_kind_sel = kind_sel

    @property
    def kind_sels(self) -> list[str]:
        return self._kind_sels

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
                    kind_sel=self.curr_kind_sel,
                    agree_func=self.curr_agree_func,
                    theta=self.curr_theta,
                    n_neighbors=self.curr_k,
                    regressor_or_classifier=self.regressor_or_classifier,
                )
                dataset_np, training_targets_np = renn.fit_resample(
                    dataset_np, training_targets_np
                )
            except Exception as e:
                print(e)

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

        if 1 in candidate_k_values:
            candidate_k_values.remove(1)
            tried_k.add(1)

        # TODO: COMMENT FOR ACTUAL EXPERIMENTS
        candidate_k_values = [3]

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
                    for kind_sel in self.kind_sels:
                        for theta in self.thetas:
                            curr_best_k = self._get_best_k_helper(
                                curr_best_k,
                                best_avg_score,
                                folds,
                                dev_data,
                                dev_targets,
                                candidate_k,
                                5,
                                agree_func,
                                kind_sel,
                                theta,
                            )
        else:
            for candidate_k in candidate_k_values:
                for kind_sel in self.kind_sels:
                    curr_best_k = self._get_best_k_helper(
                        curr_best_k,
                        best_avg_score,
                        folds,
                        dev_data,
                        dev_targets,
                        candidate_k,
                        5,
                        kind_sel=kind_sel,
                    )

        # Update tried_k with the candidate_k_values we just tried.
        tried_k.update(candidate_k_values)

        new_candidates: list[int]

        # Get a new list of candidate k values.
        new_candidates = self._expand_k_search_space(
            candidate_k_values, curr_best_k, tried_k
        )

        # TODO: COMMENT FOR ACTUAL EXPERIMENTS
        new_candidates = [3]

        # If empty list or new_candidates just has curr_best_k end grid search.
        if not new_candidates or new_candidates == [curr_best_k]:
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
        self.curr_kind_sel = self.best_kind_sel

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
        kind_sel: str | None = None,
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
        agree_func -- the agreement function to use for RENN.
        kind_sel -- the selection function to use for RENN.
        theta -- the theta / tolerance to use for RENN regression.
        """
        self.curr_agree_func = agree_func
        self.curr_kind_sel = kind_sel
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
                self.best_kind_sel = kind_sel
        else:
            # If regression avg_score is better if lower.
            if avg_score < best_avg_score:
                best_avg_score = avg_score
                curr_best_k = candidate_k
                self.best_theta = theta
                self.best_agree_func = agree_func
                self.best_kind_sel = kind_sel

        return curr_best_k


# test = RENNHarness("regressor", "datasets/regression/student_portugese.data", "G3")
test = RENNHarness("classifier", "datasets/classification/heart.data", "num")
# test = RENNHarness("regressor", "datasets/regression/automobile.data", "symboling")
print(test.evaluate())
