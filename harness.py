import pandas as pd
import numpy as np
from sklearn.model_selection import KFold  # type: ignore
from sklearn.metrics import mean_absolute_error, accuracy_score  # type: ignore
from sklearn.preprocessing import StandardScaler, LabelEncoder  # type: ignore
from tqdm import tqdm

class KNNHarness:

    def __init__(
            self,
            regressor_or_classifier: str,
            dataset_file_path: str,
            target_column_name: str,
            test_size: float = 0.2,
            missing_values: list[str] = ['?']
    ):
        '''Initialises a kNN Harness.

        Keyword arguments:
        regressor_or_classifier -- what kNN it runs 'regressor' | 'classifier'.
        dataset_file_path -- file path to the dataset to run the kNN on.
        target_column_name -- name of the column we are predicting.
        test_size -- what percentage of the dataset to reserve for testing.
        missing_values -- strings denoting missing values in the dataset.
        '''

        np.random.seed(42)

        # Raise Exception if the user passed incorrent regressor_or_classifier value.
        if regressor_or_classifier.lower() not in ['classifier', 'regressor']:
            raise ValueError(
                'regressor_or_classifier must be set to "regressor" or "classifier" ' +
                f'not "{regressor_or_classifier}"'
            )

        self._regressor_or_classifier: str = regressor_or_classifier.lower()
        self._dataset_file_path: str = dataset_file_path
        self._target_column_name: str = target_column_name
        self._test_size: float = test_size
        self._missing_values: list[str] = missing_values
        self._dev_data: pd.DataFrame = pd.DataFrame()
        self._testing_data: pd.DataFrame = pd.DataFrame()
        self._dev_targets: pd.Series = pd.Series()
        self._testing_targets: pd.Series = pd.Series()
        # Value of the best_k (to be validated later).
        self._best_k: int = 3
        # Curr value of k (used for preprocessing).
        self._curr_k: int = 3
        self._candidate_k_values: list[int] = [3]
        self._dataset: pd.DataFrame = pd.DataFrame()
        self._dataset_targets: pd.Series = pd.Series()
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
    def test_size(self) -> float:
        '''Getter for the test_size property.'''

        return self._test_size

    @property
    def missing_values(self) -> list[str]:
        '''Getter for the missing_values property.'''

        return self._missing_values

    @property
    def dev_data(self) -> pd.DataFrame:
        '''Getter for the dev_data property.'''

        return self._dev_data

    @dev_data.setter
    def dev_data(self, data: pd.DataFrame) -> None:
        '''Setter for the dev_data property.'''

        self._dev_data = data

    @property
    def testing_data(self) -> pd.DataFrame:
        '''Getter for the testing_data property.'''

        return self._testing_data

    @testing_data.setter
    def testing_data(self, data: pd.DataFrame) -> None:
        '''Setter for the testing_data property.'''

        self._testing_data = data

    @property
    def dev_targets(self) -> pd.Series:
        '''Getter for the dev_targets property.'''

        return self._dev_targets

    @dev_targets.setter
    def dev_targets(self, targets: pd.Series) -> None:
        '''Setter for the dev_targets property.'''

        self._dev_targets = targets

    @property
    def testing_targets(self) -> pd.Series:
        '''Getter for the testing targets property.'''

        return self._testing_targets

    @testing_targets.setter
    def testing_targets(self, targets: pd.Series) -> None:
        '''Setter for the testing_targets property.'''

        self._testing_targets = targets

    @property
    def best_k(self) -> int:
        '''Getter for the best_k property.'''

        return self._best_k

    @best_k.setter
    def best_k(self, k: int) -> None:
        '''Setter for the best_k property.'''

        self._best_k = k

    @property
    def curr_k(self) -> int:
        '''Getter for the curr_k property.'''

        return self._curr_k

    @curr_k.setter
    def curr_k(self, k: int) -> None:
        '''Setter for the curr_k property.'''

        self._curr_k = k

    @property
    def candidate_k_values(self) -> list[int]:
        '''Getter for the candidate_k_values property.'''
        return self._candidate_k_values

    @candidate_k_values.setter
    def candidate_k_values(self, values: list[int]) -> None:
        '''Setter for the candidate_k_values property'''

        self._candidate_k_values = values

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
        '''Loads dataset into a DataFrame and extracts target Series.'''

        # First the load the dataset.
        dataset: pd.DataFrame = pd.read_csv(self.dataset_file_path)

        # Get a range of candidate k values to validate.
        self.candidate_k_values: list[int] = self._get_candidate_k_values(
            len(dataset))

        # Replace specified missing values with NaN.
        dataset.replace(self.missing_values, np.nan, inplace=True)

        # Drop rows containing NaN.
        dataset.dropna(inplace=True)

        self.dataset = dataset.drop(columns=[self.target_column_name])
        self.dataset_targets = dataset[self.target_column_name]

        # If it's a classification task, label encode the dataset targets.
        if self.regressor_or_classifier == 'classifier':
            label_encoder: LabelEncoder = LabelEncoder()
            label_encoder.fit(self.dataset_targets)
            self.dataset_targets = pd.Series(
                label_encoder.transform(self.dataset_targets))

    def _get_candidate_k_values(self, num_examples: int) -> list[int]:
        '''Returns a list of 5 candidate values for k for KNN based on num_examples.

        Keyword arguments:
        num_examples -- the number of examples in the dataset the KNN will be run on.
        '''

        # Get a good initial value for k by getting the square root of the num_examples.
        initial_k: int = int(np.sqrt(num_examples))

        # Ensure the initial k value is odd.
        if initial_k % 2 == 0:
            initial_k -= 1

        # Get a good step size based on magnitude of initial_k.

        step: int

        if initial_k >= 1000:
            step = 1000
        elif initial_k >= 100:
            step = 100
        elif initial_k >= 10:
            step = 10
        else:
            step = 1

        candidate_k_values: list[int] = [
            # 2 steps back.
            initial_k - (2 * step),
            # 1 step back.
            initial_k - step,
            # Initial value based on sqrt(n).
            initial_k,
            # 1 step forward.
            initial_k + step,
            # 2 steps forward.
            initial_k + (2 * step)
        ]

        # Ensure all k values are odd.
        candidate_k_values = [k - 1 if k %
                              2 == 0 else k for k in candidate_k_values]

        # Ensure k values are greater than zero.
        candidate_k_values = [max(1, k) for k in candidate_k_values]

        # Just return unique candidate values.
        return sorted(list(set(candidate_k_values)))

    def _preprocess_dataset(
        self,
        dataset: pd.DataFrame,
        training_cols: pd.Index | None = None,
        scaler: StandardScaler | None = None,
        training_targets: pd.Series = pd.Series(),
    ) -> tuple[np.ndarray, np.ndarray, pd.Index, StandardScaler]:
        '''Preprocesses data and returns the cols and scaler used for the dataset.

        Keyword arguments:
        dataset -- the dataset we wish to preprocess.
        training_cols -- defaults to None; used to account for missing cols post split.
        scaler -- the StandardScaler used to scale the data, init if None is passed.
        training_targets -- target series for training data (empty if val or test).
        '''

        # If empty, just set training targets to the class attribute.
        if len(training_targets) == 0:
            training_targets = self.dev_targets

        # Columns to one-hot encode.
        cols_to_encode: list[str] = []

        # If marked with '(cat)' it is a categorical column.
        # Else, try convert columns to numeric. If unsuccessful, it's a categorical col.
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
        else:
            training_cols = dataset.columns

        dataset_scaled: np.ndarray

        if scaler is None:
            scaler = StandardScaler()
            dataset_scaled = scaler.fit_transform(dataset)
        else:
            dataset_scaled = scaler.transform(dataset)

        return (dataset_scaled, training_targets.to_numpy(), training_cols, scaler)

    def _align_test_cols(
        self,
        training_cols: pd.Index,
        testing_dataset: pd.DataFrame
    ) -> pd.DataFrame:
        '''Aligns testing_dataset (or val) cols with those of training_dataset.

        Keyword arguments:
        training_dataset -- the columns found in the training dataset.
        testing_dataset -- the dataset we wish to align to training_dataset.
        '''

        # Get the cols in training that are not in testing.
        missing_cols: set[str] = set(
            training_cols) - set(testing_dataset.columns)

        # Create a DataFrame for the missing columns filled with zeroes.
        missing_data: pd.DataFrame = pd.DataFrame(
            {col: np.zeros(len(testing_dataset)) for col in missing_cols})
        
        testing_dataset.reset_index(inplace=True)
        missing_data.reset_index(inplace=True)

        # Concatenate the original testing_dataset with the missing columns.
        testing_dataset = pd.concat([testing_dataset, missing_data], axis=1)

        # Make sure the ordering of the cols in both datasets matches.
        testing_dataset = testing_dataset[training_cols]

        # Return the aligned dataset.
        return testing_dataset

    def knn_regressor(
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
        target_column -- column w/ target values of the examples in the dataset.
        k -- the number of closest neighbors to use in the mean calculation.
        '''

        indices: np.ndarray = self.get_k_nearest_neighbors(
            example_to_predict,
            dataset,
            k
        )
        # Return mean of corresponding target values.
        return float(target_column[indices].mean())

    def get_k_nearest_neighbors(
            self,
            example_to_get_neighbors_of: np.ndarray,
            dataset: np.ndarray,
            k: int = 3
    ) -> np.ndarray:
        '''Returns the k nearest neighbors of an example.

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

        return indices

    def knn_classifier(
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
        target_column -- column w/ the class labels of the examples in the dataset.
        k -- the number of closest neighbors to use in the mode calculation.
        '''

        indices: np.ndarray = self.get_k_nearest_neighbors(
            example_to_predict,
            dataset,
            k
        )

        return self.get_most_common_class(target_column, indices)

    def get_most_common_class(self, target_column, indices) -> float:
        '''Returns the most common class in target_column among examples at indices.

        Keyword arguments:
        target_column -- column containing the target values / class labels.
        indices -- the indices of the examples we wish to get the mode class of.
        '''

        values: np.ndarray
        counts: np.ndarray

        # Find the mode of the target values
        values, counts = np.unique(target_column[indices], return_counts=True)

        # Get indices of max counts.
        max_indices: np.ndarray = np.argwhere(
            counts == np.amax(counts)).flatten()

        # Pick one at random (handles cases where there's a tie).
        random_index = np.random.choice(max_indices)

        most_frequent: float = values[random_index]

        # Return most common class of corresponding target values.
        return most_frequent

    def get_mae_of_knn_regressor(
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

        predictions: list[float] = []

        # Predict the target value for each example in the testing dataset.
        for example in testing_dataset:
            predicted_value = self.knn_regressor(
                example, training_dataset, training_targets, k)
            predictions.append(predicted_value)

        # Convert predictions to a pandas DataFrame.
        predictions_series: pd.DataFrame = pd.DataFrame(predictions)

        # Calculate mean absolute error between predictions and true values.
        mae = mean_absolute_error(testing_targets, predictions_series)

        return mae

    def get_accuracy_of_knn_classifier(
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

        # Create a list to store predictions for each example in the testing data.
        predictions: list[float] = []

        # For each example in the testing data.
        for example in testing_dataset:
            # Predict the class for this example using the kNN classifier.
            predicted_class: float = self.knn_classifier(
                example, training_dataset, training_targets, k)
            predictions.append(predicted_class)

        # Calculate the accuracy.
        accuracy: float = accuracy_score(testing_targets, predictions)

        # Return the accuracy.
        return accuracy

    def _get_best_k_for_regressor(
            self,
            candidates: list[int] | None = None,
            forward: bool = False
    ) -> int:
        '''Returns best k found for regression using 5-fold cross-validation.

        Keyword arguments:
        candidates -- k values are passed if recursively expanding search space.
        forward -- if True, recursively expanding search space with a positive step.
        '''

        best_avg_mae: float = float('inf')
        kfold: KFold = KFold(n_splits=5, shuffle=True, random_state=42)
        candidate_k: int
        candidate_k_values: list[int]

        # If this is a recursive call, assign candidates accordingly.
        if candidates:
            candidate_k_values = candidates
        else:
            candidate_k_values = self.candidate_k_values

        # For each candidate k value
        for candidate_k in candidate_k_values:

            total_mae: float = 0.0
            train_idx: np.ndarray
            val_idx: np.ndarray
            train_targets: pd.Series
            val_targets: pd.Series
            train_targets_np: np.ndarray

            # For each fold, train the model with 4 folds and validate with remaining.
            for train_idx, val_idx in kfold.split(self.dev_data):

                train_data: pd.DataFrame
                val_data: pd.DataFrame

                # Split data
                train_data, val_data = (
                    self.dev_data.iloc[train_idx].copy(),
                    self.dev_data.iloc[val_idx].copy()
                )

                train_data.reset_index(drop=True, inplace=True)
                val_data.reset_index(drop=True, inplace=True)

                train_targets, val_targets = self.dev_targets.iloc[
                    train_idx].copy(), self.dev_targets.iloc[val_idx].copy()

                train_targets.reset_index(drop=True, inplace=True)
                val_targets.reset_index(drop=True, inplace=True)

                train_data_scaled: np.ndarray
                val_data_scaled: np.ndarray
                training_cols: pd.Index
                scaler: StandardScaler

                # Update curr_k.
                self.curr_k = candidate_k

                # Preprocess the training data and apply transformations to val data.
                (
                    train_data_scaled,
                    train_targets_np,
                    training_cols,
                    scaler
                ) = self._preprocess_dataset(train_data, training_targets=train_targets)

                val_data_scaled, _, _, _ = self._preprocess_dataset(
                    val_data, training_cols=training_cols, scaler=scaler)

                mae: float = self.get_mae_of_knn_regressor(
                    candidate_k, train_data_scaled,
                    val_data_scaled, train_targets_np, val_targets
                )

                total_mae += mae

            avg_mae: float = total_mae / kfold.get_n_splits()

            # If better than best, update best_avg_mae and best_k.
            if avg_mae < best_avg_mae:
                best_avg_mae = avg_mae
                self.best_k = candidate_k

        # Default value for best_k if left undefined.
        if not self.best_k:
            self.best_k = 3

        # If we reach one of the edges of candidate_k_values, expand search space.
        # Only expand 'backward' if we weren't expanding 'forward'.
        if (
            (self.best_k == candidate_k_values[0] and not forward) or
            self.best_k == candidate_k_values[-1]
        ):
            self._expand_k_search_space(candidate_k_values, forward)

        return self.best_k

    def _evaluate_regressor(self) -> float:
        '''Returns error of KNN regressor on the test set.'''

        total_mae: float = 0

        dev_idx: np.ndarray
        test_idx: np.ndarray

        kfold: KFold = KFold(n_splits=5, shuffle=True, random_state=42)

        # Nested k-fold cross validation.
        # tqdm provides progress bar.
        for dev_idx, test_idx in tqdm(kfold.split(self.dataset), total=5):

            dev_data: pd.DataFrame
            test_data: pd.DataFrame
            dev_targets: pd.Series
            test_targets: pd.Series

            # Split data
            dev_data, test_data = (
                self.dataset.iloc[dev_idx].copy(),
                self.dataset.iloc[test_idx].copy()
            )

            dev_targets, test_targets = (
                self.dataset_targets.iloc[dev_idx].copy(),
                self.dataset_targets.iloc[test_idx].copy()
            )

            self.dev_data = dev_data
            self.testing_data = test_data
            self.dev_targets = dev_targets
            self.testing_targets = test_targets
            
            self.best_k = self._get_best_k_for_classifier()

            dev_data_scaled: np.ndarray
            testing_data_scaled: np.ndarray
            training_cols: pd.Index
            dev_targets: np.ndarray
            scaler: StandardScaler

            self.curr_k = self.best_k

            # Preprocess split datasets.
            (
                dev_data_scaled,
                dev_targets,
                training_cols,
                scaler
            ) = self._preprocess_dataset(self.dev_data)

            testing_data_scaled, _, _, _ = self._preprocess_dataset(
                self.testing_data, training_cols, scaler)
            # Get MAE of test data when neighbors are gotten from train+val.
            
            total_mae += self.get_mae_of_knn_regressor(
                self.best_k, dev_data_scaled,
                testing_data_scaled,
                dev_targets,
                self.testing_targets
            )

        return total_mae / 5

    def _expand_k_search_space(
            self,
            candidate_k_values: list[int],
            forward: bool = False
    ) -> None:
        '''
        Recursively calls _get_best_k... while expanding candidate_k_values.

        Keyword arguments:
        candidate_k_values -- the initial search space for best k.
        forward -- whether we are expanding forwards i.e. increasing k values.
        '''

        step: int

        # Get a good step size based on magnitude of best_k.
        if self.best_k >= 1000:
            step = 1000
        elif self.best_k >= 100:
            step = 100
        elif self.best_k >= 10:
            step = 10
        else:
            step = 1

        # If we are at the left edge, should be a negative step.
        if self.best_k == candidate_k_values[0]:
            step *= -1
        else:
            forward = True

        new_candidates: list[int] = []
        curr_new_candidate: int = self.best_k + step

        # While we haven't exausted positive values and haven't more than 2 new k's.
        while curr_new_candidate > 0 and len(new_candidates) < 2:
            new_candidates.append(curr_new_candidate)
            # Expand search space according to step.
            curr_new_candidate += step

        # Only take odd k's and sort (otherwise will be backwards w/ negative step).
        new_candidates = sorted([k - 1 if k %
                                 2 == 0 else k for k in new_candidates])

        # If we only had negative k's we'll have an empty list, so return.
        if not new_candidates:
            return
        
        # Call _get_best_k... with new search space.
        if self.regressor_or_classifier == 'classifier':
            self.best_k = self._get_best_k_for_classifier(
                new_candidates, forward)
        else:
            self.best_k = self._get_best_k_for_regressor(
                new_candidates, forward)

    def _get_best_k_for_classifier(
            self,
            candidates: list[int] | None = None,
            forward: bool = False
    ) -> int:
        '''Returns the best k found for classification using 5-fold cross-validation.

        Keyword arguments:
        candidates -- k values are passed if recursively expanding search space.
        forward -- if True, recursively expanding search space with a positive step.
        '''

        best_avg_accuracy: float = float('-inf')
        kfold: KFold = KFold(n_splits=5, shuffle=True, random_state=42)
        candidate_k: int
        candidate_k_values: list[int]

        # If this is a recursive call, assign candidates accordingly.
        if candidates:
            candidate_k_values = candidates
        else:
            candidate_k_values = self.candidate_k_values

        # For each candidate k value
        for candidate_k in candidate_k_values:

            total_accuracy: float = 0.0
            train_idx: np.ndarray
            val_idx: np.ndarray
            train_targets: pd.Series
            val_targets: pd.Series
            train_targets_np: np.ndarray

            # For each fold, train the model with 4 folds and validate with remaining.
            for train_idx, val_idx in kfold.split(self.dev_data):

                train_data: pd.DataFrame
                val_data: pd.DataFrame

                # Split data
                train_data, val_data = (
                    self.dev_data.iloc[train_idx].copy(),
                    self.dev_data.iloc[val_idx].copy()
                )

                train_data.reset_index(drop=True, inplace=True)
                val_data.reset_index(drop=True, inplace=True)

                train_targets, val_targets = (
                    self.dev_targets.iloc[train_idx].copy(),
                    self.dev_targets.iloc[val_idx].copy()
                )

                train_targets.reset_index(drop=True, inplace=True)
                val_targets.reset_index(drop=True, inplace=True)

                train_data_scaled: np.ndarray
                val_data_scaled: np.ndarray
                training_cols: pd.Index
                scaler: StandardScaler

                self.curr_k = candidate_k

                # Preprocess the training data and apply transformations to val data.
                (
                    train_data_scaled,
                    train_targets_np,
                    training_cols,
                    scaler
                ) = self._preprocess_dataset(train_data, training_targets=train_targets)

                val_data_scaled, _, _, _ = self._preprocess_dataset(
                    val_data, training_cols=training_cols, scaler=scaler)

                accuracy: float = self.get_accuracy_of_knn_classifier(
                    candidate_k, train_data_scaled,
                    val_data_scaled, train_targets_np, val_targets
                )

                total_accuracy += accuracy

            avg_accuracy: float = total_accuracy / kfold.get_n_splits()

            # If better than best, update best_avg_accuracy and best_k.
            if avg_accuracy > best_avg_accuracy:
                best_avg_accuracy = avg_accuracy
                self.best_k = candidate_k

        # Default value for best_k if left undefined.
        if not self.best_k:
            self.best_k = 3

        # If we reach one of the edges of candidate_k_values, expand search space.
        # Only expand 'backward' if we weren't expanding 'forward'.
        if (
            (self.best_k == candidate_k_values[0] and not forward) or
            self.best_k == candidate_k_values[-1]
        ):
            self._expand_k_search_space(candidate_k_values, forward)

        return self.best_k

    def _evaluate_classifier(self) -> float:
        '''Returns accuracy of KNN classifier on the test set.'''

        total_accuracy: float = 0

        dev_idx: np.ndarray
        test_idx: np.ndarray

        kfold: KFold = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # Nested k-fold cross validation.
        # tqdm provides progress bar.
        for dev_idx, test_idx in tqdm(kfold.split(self.dataset), total=5):

            dev_data: pd.DataFrame
            test_data: pd.DataFrame
            dev_targets: pd.Series
            test_targets: pd.Series

            # Split data
            dev_data, test_data = (
                self.dataset.iloc[dev_idx].copy(),
                self.dataset.iloc[test_idx].copy()
            )

            dev_targets, test_targets = (
                self.dataset_targets.iloc[dev_idx].copy(),
                self.dataset_targets.iloc[test_idx].copy()
            )

            self.dev_data = dev_data
            self.testing_data = test_data
            self.dev_targets = dev_targets
            self.testing_targets = test_targets
            
            self.best_k = self._get_best_k_for_classifier()

            dev_data_scaled: np.ndarray
            testing_data_scaled: np.ndarray
            training_cols: pd.Index
            scaler: StandardScaler
            dev_targets_np: np.ndarray

            # Preprocess split datasets.
            (
                dev_data_scaled,
                dev_targets_np,
                training_cols,
                scaler
            ) = self._preprocess_dataset(self.dev_data)

            testing_data_scaled, _, _, _ = self._preprocess_dataset(
                self.testing_data, training_cols, scaler)
            
            # Get accuracy of test data when neighbors are gotten from train+val.
            total_accuracy += self.get_accuracy_of_knn_classifier(
                self.best_k, dev_data_scaled,
                testing_data_scaled,
                dev_targets_np,
                self.testing_targets
            )

        # Did 5 repeated holdouts, return average accuracy.
        return total_accuracy / 5

    def evaluate(self) -> float:
        '''Returns error or accuracy rate (depending on task) of kNN on the dataset.'''

        if self.regressor_or_classifier == 'regressor':
            return self._evaluate_regressor()
        else:
            return self._evaluate_classifier()

# test = KNNHarness('classifier', 'datasets/zoo.data', 'type')
## test = KNNHarness('classifier', 'datasets/custom_cleveland.data', 'num')
# test = KNNHarness('regressor', 'datasets/abalone.data', 'Rings')
# print(test.evaluate())
