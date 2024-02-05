import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold  # type: ignore
from sklearn.metrics import mean_absolute_error, accuracy_score  # type: ignore
from sklearn.preprocessing import StandardScaler, LabelEncoder  # type: ignore
from tqdm import tqdm


class KNNHarness:

    def __init__(
            self,
            regressor_or_classifier: str,
            dataset_file_path: str,
            target_column_name: str,
            missing_values: list[str] = ['?']
    ):
        '''Initialises a kNN Harness.

        Keyword arguments:
        regressor_or_classifier -- what kNN it runs 'regressor' | 'classifier'.
        dataset_file_path -- file path to the dataset to run the kNN on.
        target_column_name -- name of the column we are predicting.
        missing_values -- strings denoting missing values in the dataset.
        '''

        # Set random seed to make the outcomes of experiments deterministic.
        np.random.seed(42)

        # Raise Exception if user passed invalid regressor_or_classifier val.
        if regressor_or_classifier.lower() not in ['classifier', 'regressor']:
            raise ValueError(
                'regressor_or_classifier must be "regressor" | "classifier" ' +
                f'not "{regressor_or_classifier}"'
            )

        self._regressor_or_classifier: str = regressor_or_classifier.lower()
        self._dataset_file_path: str = dataset_file_path
        self._target_column_name: str = target_column_name
        self._missing_values: list[str] = missing_values
        # The curr step size used for the grid search of k (calculated later).
        self._step_size_k: int = 10
        # The current (not best) value for k (important for editing algos).
        self._curr_k: int = 3
        # Dataset as a Dataframe.
        self._dataset: pd.DataFrame = pd.DataFrame()
        # Target values as a Series.
        self._dataset_targets: pd.Series = pd.Series()
        # Populate the Dataframe and Series; label encode the target column.
        self._load_dataset()

    @property
    def regressor_or_classifier(self) -> str:
        '''Getter for the regressor_or_classifier property.'''

        return self._regressor_or_classifier

    @property
    def dataset_file_path(self) -> str:
        '''Getter for the dataset_file_path property.'''

        return self._dataset_file_path

    @property
    def target_column_name(self) -> str:
        '''Getter for the target_column_name property.'''

        return self._target_column_name

    @property
    def missing_values(self) -> list[str]:
        '''Getter for the missing_values property.'''

        return self._missing_values

    @property
    def step_size_k(self) -> int:
        '''Getter for the step_size_k property.'''

        return self._step_size_k

    @step_size_k.setter
    def step_size_k(self, step: int) -> None:
        '''Setter for the step_size_k property.'''

        self._step_size_k = step

    @property
    def curr_k(self) -> int:
        '''Getter for the curr_k property.'''

        return self._curr_k

    @curr_k.setter
    def curr_k(self, k: int) -> None:
        '''Setter for the curr_k property.'''

        self._curr_k = k

    @property
    def dataset(self) -> pd.DataFrame:
        '''Getter for the dataset property.'''

        return self._dataset

    @dataset.setter
    def dataset(self, data: pd.DataFrame) -> None:
        '''Setter for the dataset property'''

        self._dataset = data

    @property
    def dataset_targets(self) -> pd.Series:
        '''Getter for the dataset_targets property.'''

        return self._dataset_targets

    @dataset_targets.setter
    def dataset_targets(self, targets: pd.Series) -> None:
        '''Setter for the dataset_targets property'''

        self._dataset_targets = targets

    def _load_dataset(self) -> None:
        '''Loads dataset as DF; extracts target Series; encodes targets.'''

        # First the load the dataset.
        dataset: pd.DataFrame = pd.read_csv(self.dataset_file_path)

        # Replace specified missing values with NaN.
        dataset.replace(self.missing_values, np.nan, inplace=True)

        # Drop rows containing NaN.
        dataset.dropna(inplace=True)

        # Exclude the target column from the dataset.
        self.dataset = dataset.drop(columns=[self.target_column_name])
        # Put the excluded target column into a Series.
        self.dataset_targets = dataset[self.target_column_name]

        # If it's a classification task, label encode the dataset targets.
        if self.regressor_or_classifier == 'classifier':
            label_encoder: LabelEncoder = LabelEncoder()
            label_encoder.fit(self.dataset_targets)
            self.dataset_targets = pd.Series(
                label_encoder.transform(self.dataset_targets))

    def _get_initial_candidate_k_values(self, num_examples: int) -> list[int]:
        '''Returns init list of 5 k candidates for KNN based on num_examples.

        Keyword arguments:
        num_examples -- the num examples in the dataset the KNN will be run on.
        '''

        # Get a good initial value for k by getting the sqrt of num_examples.
        initial_k: int = int(np.sqrt(num_examples))

        # Ensure the initial k value is odd.
        if initial_k % 2 == 0:
            initial_k -= 1

        # Get a good step size based on magnitude of initial_k.

        if initial_k >= 1000:
            self.step_size_k = 1000
        elif initial_k >= 100:
            self.step_size_k = 100
        elif initial_k >= 10:
            self.step_size_k = 10
        else:
            self.step_size_k = 1

        candidate_k_values: list[int] = [
            # 2 steps back.
            initial_k - (2 * self.step_size_k),
            # 1 step back.
            initial_k - self.step_size_k,
            # Initial value based on sqrt(n).
            initial_k,
            # 1 step forward.
            initial_k + self.step_size_k,
            # 2 steps forward.
            initial_k + (2 * self.step_size_k)
        ]

        # Ensure all k values are odd.
        candidate_k_values = [k - 1 if k %
                              2 == 0 else k for k in candidate_k_values]

        # Ensure k values are greater than zero.
        candidate_k_values = [max(1, k) for k in candidate_k_values]

        # Just return unique candidate values, sorted in ascending order.
        return sorted(list(set(candidate_k_values)))

    def _preprocess_dataset(
        self,
        dataset: pd.DataFrame,
        training_targets: pd.Series,
        training_cols: pd.Index | None = None,
        scaler: StandardScaler | None = None,
    ) -> tuple[np.ndarray, np.ndarray, pd.Index, StandardScaler]:
        '''Preprocesses data and returns cols and scaler used for the dataset.

        Keyword arguments:
        dataset -- the dataset we wish to preprocess.
        training_targets -- the targets associated with the training dataset.
        training_cols -- default None; accounts for missing cols post split.
        scaler -- the StandardScaler to use, init if None is passed.
        '''

        # Columns to one-hot encode.
        cols_to_encode: list[str] = []

        # If marked with '(cat)' it is a categorical column.
        # Else, try convert columns to numeric. If can't, it's a cat col.
        for col in dataset.columns:
            if '(cat)' in col.lower():
                cols_to_encode.append(col)
                continue
            try:
                dataset[col] = pd.to_numeric(dataset[col])
            except ValueError:
                cols_to_encode.append(col)

        # One-hot encode categorical columns.
        dataset = pd.get_dummies(dataset, columns=cols_to_encode, dtype=float)

        # Add missing cols if this is the test or validation dataset.
        if training_cols is not None:
            dataset = self._align_test_cols(training_cols, dataset)

        # If this is a dev or training set, set the training_cols.
        else:
            training_cols = dataset.columns

        dataset_scaled: np.ndarray

        # If we didn't pass a scaler this must be a dev or training set.
        if scaler is None:
            # Initialise a scaler.
            scaler = StandardScaler()
            dataset_scaled = scaler.fit_transform(dataset)

        # This is either test or val so we should use existing scaler.
        else:
            dataset_scaled = scaler.transform(dataset)

        # Return scaled data, train targets as np array, train cols & scaler.
        return (
            dataset_scaled,
            training_targets.to_numpy(),
            training_cols,
            scaler
        )

    def _align_test_cols(
        self,
        training_cols: pd.Index,
        testing_dataset: pd.DataFrame
    ) -> pd.DataFrame:
        '''Aligns testing_dataset (or val) cols with those of dev or train set.

        Keyword arguments:
        training_cols -- the columns found in the training dataset.
        testing_dataset -- the dataset we wish to align to training_dataset.
        '''

        # Get the cols in training that are not in testing.
        missing_cols: set[str] = set(
            training_cols) - set(testing_dataset.columns)

        # Create a DataFrame for the missing columns filled with zeroes.
        missing_data: pd.DataFrame = pd.DataFrame(
            {col: np.zeros(len(testing_dataset)) for col in missing_cols})

        # Reset indices of both DataFrames.
        testing_dataset.reset_index(inplace=True)
        missing_data.reset_index(inplace=True)

        # Concatenate the original testing_dataset with the missing columns.
        testing_dataset = pd.concat([testing_dataset, missing_data], axis=1)

        # Make sure the ordering of the cols in both datasets matches.
        testing_dataset = testing_dataset[training_cols]

        # Return the aligned dataset.
        return testing_dataset

    def _knn_regressor(
            self,
            example_to_predict: np.ndarray,
            dataset: np.ndarray,
            target_column: np.ndarray,
            k: int = 3
    ) -> float:
        '''Predicts the target value of an example using kNN.

        Keyword arguments:
        example_to_predict -- the example we are running the regression on.
        dataset -- the dataset to get the nearest neighbors from.
        target_column -- column w/ target vals of the examples in the dataset.
        k -- the number of closest neighbors to use in the mean calculation.
        '''

        # Get the indices of the nearest neighbors.
        indices: np.ndarray = self._get_k_nearest_neighbors(
            example_to_predict,
            dataset,
            k
        )
        # Return mean of corresponding target values.
        return float(target_column[indices].mean())

    def _get_k_nearest_neighbors(
            self,
            example_to_get_neighbors_of: np.ndarray,
            dataset: np.ndarray,
            k: int = 3,
            removed_examples = None,
            example_to_exclude: int = 0
    ) -> np.ndarray:
        '''Returns k nearest neighbors of an example (as an array of indices).

        Keyword arguments:
        example_to_get_neighbors_of -- the example we are interested in.
        dataset -- the example to get the nearest neighbors from.
        k -- the number of nearest neighbors to fetch.
        '''

        # Compute euclidean distances using vectorized operations.
        distances: np.ndarray = np.sqrt(
            np.sum((dataset - example_to_get_neighbors_of)**2, axis=1))
        # Get indices of the k smallest distances
        indices: np.ndarray = np.argsort(distances)[:k]

        # Return the indices of the closest neighbors.
        return indices

    def _knn_classifier(
            self,
            example_to_predict: np.ndarray,
            dataset: np.ndarray,
            target_column: np.ndarray,
            k: int = 3
    ) -> float:
        '''Predicts the class label of an example using kNN.

        Keyword arguments:
        example_to_predict -- the example we are running the classification on.
        dataset -- the dataset to get the nearest neighbors from.
        target_column -- col w/ class labels of the examples in the dataset.
        k -- the number of closest neighbors to use in the mode calculation.
        '''

        # Get the indices of the nearest neighbors.
        indices: np.ndarray = self._get_k_nearest_neighbors(
            example_to_predict,
            dataset,
            k
        )

        # Return the most common class of the nearest neighbors.
        return self._get_most_common_class(target_column, indices)

    def _get_most_common_class(self, target_column, indices) -> float:
        '''Returns most common class in target_column among examples @ indices.

        Keyword arguments:
        target_column -- column containing the target values / class labels.
        indices -- indices of the examples we wish to get the mode class of.
        '''

        values: np.ndarray
        counts: np.ndarray

        # Find the mode of the target values.
        values, counts = np.unique(target_column[indices], return_counts=True)

        # Get indices of examples with the mode class.
        max_indices: np.ndarray = np.argwhere(
            counts == np.amax(counts)).flatten()

        # Pick one at random (handles cases where there's a tie).
        random_index = np.random.choice(max_indices)

        # Give the final answer of the vote.
        most_frequent: float = values[random_index]

        # Return most common class of corresponding target values.
        return most_frequent

    def _get_mae_of_knn_regressor(
            self,
            k: int,
            training_dataset: np.ndarray,
            testing_dataset: np.ndarray,
            training_targets: np.ndarray,
            testing_targets: pd.Series
    ):
        '''Returns the MAE of a KNN regressor.

        Keyword arguments:
        k -- the value of k for the KNN regression.
        training_dataset -- the dataset the neighbors are taken from.
        testing_dataset -- the validation or test dataset to get the MAE for.s
        training_targets -- the target values of training_dataset.
        testing_targets -- the target values of testing_dataset.
        '''

        # The list of our predicted values.
        predictions: list[float] = []

        # Predict the target value for each example in the testing dataset.
        for example in testing_dataset:
            predicted_value = self._knn_regressor(
                example, training_dataset, training_targets, k)
            predictions.append(predicted_value)

        # Convert predictions to a pandas DataFrame.
        predictions_series: pd.DataFrame = pd.DataFrame(predictions)

        # Calculate mean absolute error between predictions and true values.
        mae = mean_absolute_error(testing_targets, predictions_series)

        # Return the MAE.
        return mae

    def _get_accuracy_of_knn_classifier(
            self,
            k: int,
            training_dataset: np.ndarray,
            testing_dataset: np.ndarray,
            training_targets: np.ndarray,
            testing_targets: pd.Series
    ) -> float:
        '''Returns the accuracy of a KNN classifier.

        Keyword arguments:
        k -- the value of k for the KNN classification.
        training_dataset -- the dataset the neighbors are taken from.
        testing_dataset -- the validation or test dataset to get the MAE for.
        training_targets -- the target values of training_dataset.
        testing_targets -- the target values of testing_dataset.
        '''

        # Create list to store predictions for each example in testing data.
        predictions: list[float] = []

        # For each example in the testing data.
        for example in testing_dataset:
            # Predict the class for this example using the kNN classifier.
            predicted_class: float = self._knn_classifier(
                example, training_dataset, training_targets, k)
            predictions.append(predicted_class)

        # Calculate the accuracy.
        accuracy: float = accuracy_score(testing_targets, predictions)

        # Return the accuracy.
        return accuracy

    def _expand_k_search_space(
            self,
            candidate_k_values: list[int],
            curr_best_k: int,
            tried_k: set[int]
    ) -> list[int]:
        '''
        Returns new list of candidate k_values by expanding grid search space.

        Keyword arguments:
        candidate_k_values -- the initial search space for best k.
        curr_best_k -- the current best value for k found.
        tried_k -- the k values we have tried so far.
        '''

        # The list of new candidates.
        new_candidates: list[int] = []

        step: int

        # If this isn't an edge of a list of candidates.
        if (
            curr_best_k != candidate_k_values[0] and
            curr_best_k != candidate_k_values[-1]
        ):

            # Reduce step size by 75%.
            step = int(self.step_size_k * 0.25)

            # If we've gone too small, make the step size 2.
            if step <= 1:
                step = 2

            self.step_size_k = step

            # Take two steps back and two steps forward.
            new_candidates = [curr_best_k + i*step for i in range(-2, 3)]

        # If this is an edge case i.e. start of prev list or end of prev list.
        else:

            # Get the step size from the class.
            step = self.step_size_k

            # If we are at the left edge, should be a negative step.
            if curr_best_k == candidate_k_values[0]:
                step *= - 1

            # Expand search space according to step.
            curr_new_candidate: int = curr_best_k + step

            # While we haven't exausted positive values and > 4 new k's.
            while curr_new_candidate > 0 and len(new_candidates) < 4:

                # Add the new candidate.
                new_candidates.append(curr_new_candidate)

                # Expand search space according to step.
                curr_new_candidate += step

            # Add the curr_best_k back to the list of candidates.
            new_candidates.append(curr_best_k)

        # Only take odd k's.
        new_candidates = list(set([k - 1 if k %
                                   2 == 0 else k for k in new_candidates]))

        # Only take positive ks that we haven't tried before (or == curr_best).
        new_candidates = sorted(
            list(
                set(
                    [k for k in new_candidates if k ==
                        curr_best_k or k not in tried_k and k > 0]
                )
            )
        )

        return new_candidates

    def _get_best_k(
            self,
            dev_data: pd.DataFrame,
            dev_targets: pd.Series,
            candidate_k_values: list[int],
            best_avg_score: float | None = None,
            curr_best_k: int = 3,
            tried_k: set[int] = set()

    ) -> int:
        '''Returns best k found using 5-fold cross-validation.

        Keyword arguments:
        dev_data -- the unison of the training and validation datasets.
        dev_targets -- the target values associated with each row of dev_data.
        candidate_k_values -- the candidates for k currently being considered.
        best_avg_score -- the best average accuracy / MAE recorded so far.
        curr_best_k -- the best value for k recorded so far.
        tried_k -- the k values we have tried so far.
        '''

        # If this is the first call on the function.
        if best_avg_score is None:
            # If classifier set the init best_avg_score to negative infinity.
            if self.regressor_or_classifier == 'classifier':
                best_avg_score = float('-inf')
            # If regressor set the init best_avg_score to positive infinity.
            else:
                best_avg_score = float('inf')

        if self.regressor_or_classifier == "classifier":
            kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            folds = list(kfold.split(dev_data, dev_targets))
        else:
            kfold = KFold(n_splits=5, shuffle=True, random_state=42)
            folds = list(kfold.split(dev_data))

        candidate_k: int

        # For each candidate k value
        for candidate_k in candidate_k_values:

            # total_score is either total MAE or total accuracy.
            total_score: float = 0.0

            train_idx: np.ndarray
            val_idx: np.ndarray
            train_targets: pd.Series
            val_targets: pd.Series
            train_targets_np: np.ndarray

            # For each fold, train the model with 4 folds and val w/ remaining.
            for train_idx, val_idx in folds:

                train_data: pd.DataFrame
                val_data: pd.DataFrame

                # Split data
                train_data, val_data = (
                    dev_data.iloc[train_idx].copy(),
                    dev_data.iloc[val_idx].copy()
                )

                train_data.reset_index(drop=True, inplace=True)
                val_data.reset_index(drop=True, inplace=True)

                train_targets, val_targets = (
                    dev_targets.iloc[train_idx].copy(),
                    dev_targets.iloc[val_idx].copy()
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
                    scaler
                ) = self._preprocess_dataset(train_data, train_targets)

                val_data_scaled, _, _, _ = self._preprocess_dataset(
                    val_data,
                    train_targets,
                    training_cols=training_cols,
                    scaler=scaler
                )

                score: float
                

                # Get accuracy for this fold if its a classifier.
                if self.regressor_or_classifier == 'classifier':
                    score = self._get_accuracy_of_knn_classifier(
                        candidate_k, train_data_scaled,
                        val_data_scaled, train_targets_np, val_targets
                    )

                # Get MAE for this fold if its a regressor.
                else:
                    score = self._get_mae_of_knn_regressor(
                        candidate_k, train_data_scaled,
                        val_data_scaled, train_targets_np, val_targets
                    )

                # Add the score for this fold to the total_score.
                total_score += score

            # Get the avg_score by dividing by the number of folds.
            avg_score: float = total_score / kfold.get_n_splits()

            # If better than best, update best_score and best_k.
            if self.regressor_or_classifier == 'classifier':

                # If classification avg_score is better if higher.
                if avg_score > best_avg_score:
                    best_avg_score = avg_score
                    curr_best_k = candidate_k

            else:

                # If regression avg_score is better if lower.
                if avg_score < best_avg_score:
                    best_avg_score = avg_score
                    curr_best_k = candidate_k

        # Update tried_k with the candidate_k_values we just tried.
        tried_k.update(candidate_k_values)

        new_candidates: list[int]

        # Get a new list of candidate k values.
        new_candidates = self._expand_k_search_space(
            candidate_k_values, curr_best_k, tried_k)

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
            tried_k
        )

        return curr_best_k

    def evaluate(self) -> float:
        '''Returns MAE or accuracy of kNN on the dataset.'''

        # Total MAE / accuracy (depending on task).
        total_score: float = 0

        # The indices of the dev data i.e. training + validation.
        dev_idx: np.ndarray
        # The indices of the testing data.
        test_idx: np.ndarray

        if self.regressor_or_classifier == "classifier":
            kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            folds = list(kfold.split(self.dataset, self.dataset_targets))
        else:
            kfold = KFold(n_splits=5, shuffle=True, random_state=42)
            folds = list(kfold.split(self.dataset))

        # 5-fold cross validation.
        # tqdm provides progress bar.
        for dev_idx, test_idx in tqdm(folds):

            dev_data: pd.DataFrame
            test_data: pd.DataFrame
            dev_targets: pd.Series
            test_targets: pd.Series

            # Split data into dev and test.

            dev_data, test_data = (
                self.dataset.iloc[dev_idx].copy(),
                self.dataset.iloc[test_idx].copy()
            )

            dev_targets, test_targets = (
                self.dataset_targets.iloc[dev_idx].copy(),
                self.dataset_targets.iloc[test_idx].copy()
            )

            # Get initial list of candidate k values.
            candidate_k_values: list[int] = \
                self._get_initial_candidate_k_values(len(self.dataset))

            best_k: int

            # Get best k by getting predictions for validation from training.
            best_k = self._get_best_k(
                dev_data, dev_targets, candidate_k_values)

            dev_data_scaled: np.ndarray
            testing_data_scaled: np.ndarray
            training_cols: pd.Index
            scaler: StandardScaler
            dev_targets_np: np.ndarray

            self.curr_k = best_k
            
            # Preprocess split datasets.
            (
                dev_data_scaled,
                dev_targets_np,
                training_cols,
                scaler
            ) = self._preprocess_dataset(dev_data, dev_targets)

            testing_data_scaled, _, _, _ = self._preprocess_dataset(
                test_data, dev_targets, training_cols, scaler)

            # Get MAE/accuracy of test data when neighbors are gotten from dev.

            if self.regressor_or_classifier == 'regressor':

                total_score += self._get_mae_of_knn_regressor(
                    best_k, dev_data_scaled,
                    testing_data_scaled,
                    dev_targets_np,
                    test_targets
                )

            else:

                total_score += self._get_accuracy_of_knn_classifier(
                    best_k, dev_data_scaled,
                    testing_data_scaled,
                    dev_targets_np,
                    test_targets
                )
        # Divide total_score by number of folds to get average.
        return total_score / kfold.get_n_splits()


# test = KNNHarness('classifier', 'datasets/wine.data', 'class')
# test = KNNHarness('classifier', 'datasets/heart.data', 'num')
# print(test.evaluate())
# test = KNNHarness('regressor', 'datasets/abalone.data', 'Rings')
# print(test.evaluate())
