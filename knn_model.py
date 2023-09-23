import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.metrics import mean_absolute_error, accuracy_score  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore


class KNNHarness:

    def __init__(
            self,
            regressor_or_classifier: str,
            dataset_file_path: str,
            target_column_name: str,
            test_size: float = 0.2,
            missing_values: list[str] = ['?']
    ):
        '''Initialises a KNN Harness.

        Keyword arguments:
        regressor_or_classifier -- what KNN it runs 'regressor' | 'classifier'.
        dataset_file_path -- file path to the dataset to run the KNN on.
        target_column_name -- name of the column we are predicting.
        test_size -- what percentage of the dataset to reserve for testing.
        missing_values -- strings denoting missing values in the dataset.
        '''
        
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
        self._training_data: np.ndarray = np.ndarray([])
        self._validation_data: np.ndarray = np.ndarray([])
        self._testing_data: np.ndarray = np.ndarray([])
        self._training_targets: np.ndarray = np.ndarray([])
        self._validation_targets: pd.Series = pd.Series()
        self._testing_targets: pd.Series = pd.Series()
        self._merged_val_training_data: np.ndarray = np.ndarray([])
        self._merged_val_training_targets: np.ndarray = np.ndarray([])
        # Value of the best_k (to be validated later).
        self._best_k: int  = 3
        self._candidate_k_values: list[int] = [3]
        self._load_split_and_preprocess_data()

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
    def training_data(self) -> np.ndarray:
        '''Getter for the training_data property.'''

        return self._training_data

    @training_data.setter
    def training_data(self, data: np.ndarray) -> None:
        '''Setter for the training_data property.'''

        self._training_data = data

    @property
    def validation_data(self) -> np.ndarray:
        '''Getter for the validation_data property.'''

        return self._validation_data

    @validation_data.setter
    def validation_data(self, data: np.ndarray) -> None:
        '''Setter for the validation_data property.'''

        self._validation_data = data

    @property
    def testing_data(self) -> np.ndarray:
        '''Getter for the testing_data property.'''

        return self._testing_data

    @testing_data.setter
    def testing_data(self, data: np.ndarray) -> None:
        '''Setter for the testing_data property.'''

        self._testing_data = data

    @property
    def training_targets(self) -> np.ndarray:
        '''Getter for the training_targets property.'''

        return self._training_targets

    @training_targets.setter
    def training_targets(self, targets: np.ndarray) -> None:
        '''Setter for the training_targets property.'''

        self._training_targets = targets

    @property
    def validation_targets(self) -> pd.Series:
        '''Getter for the validation targets property.'''

        return self._validation_targets

    @validation_targets.setter
    def validation_targets(self, targets: pd.Series) -> None:
        '''Setter for the validation_targets property.'''

        self._validation_targets = targets

    @property
    def testing_targets(self) -> pd.Series:
        '''Getter for the testing targets property.'''

        return self._testing_targets

    @testing_targets.setter
    def testing_targets(self, targets: pd.Series) -> None:
        '''Setter for the testing_targets property.'''

        self._testing_targets = targets

    @property
    def merged_val_training_data(self) -> np.ndarray:
        '''Getter for the merged_val_training_data property.'''

        return self._merged_val_training_data

    @merged_val_training_data.setter
    def merged_val_training_data(self, data: np.ndarray) -> None:
        '''Setter for the merged_val_training_data property.'''

        self._merged_val_training_data = data

    @property
    def merged_val_training_targets(self) -> np.ndarray:
        '''Getter for the merged_val_training_targets property.'''

        return self._merged_val_training_targets

    @merged_val_training_targets.setter
    def merged_val_training_targets(self, targets: np.ndarray) -> None:
        '''Setter for the merged_val_training_targets property'''

        self._merged_val_training_targets = targets

    @property
    def best_k(self) -> int:
        '''Getter for the best_k property.'''

        return self._best_k

    @best_k.setter
    def best_k(self, k: int) -> None:
        '''Setter for the best_k property.'''

        self._best_k = k

    @property
    def candidate_k_values(self) -> list[int]:
        '''Getter for the candidate_k_values property.'''
        return self._candidate_k_values

    @candidate_k_values.setter
    def candidate_k_values(self, values: list[int]) -> None:
        '''Setter for the candidate_k_values property'''

        self._candidate_k_values = values

    def _load_split_and_preprocess_data(self) -> None:
        '''Loads, splits, and preprocesses dataset.'''

        # First the load the dataset.
        dataset: pd.DataFrame = pd.read_csv(self.dataset_file_path)

        # Get a range of candidate k values to validate.
        self.candidate_k_values: list[int] = self._get_candidate_k_values(
            len(dataset))

        # Replace specified missing values with NaN.
        dataset.replace(self.missing_values, np.nan, inplace=True)

        # Drop rows containing NaN.
        dataset.dropna(inplace=True)

        # Split dataset into training, validation, and test datasets.
        training_data, validation_data, testing_data = self._split_dataset(
            dataset, self.test_size)

        # Reset the indices.
        training_data.reset_index(inplace=True, drop=True)
        validation_data.reset_index(inplace=True, drop=True)
        testing_data.reset_index(inplace=True, drop=True)

        # Get targets for each of the subsets of the data.
        training_targets = training_data[self.target_column_name]
        validation_targets = validation_data[self.target_column_name]
        testing_targets = testing_data[self.target_column_name]

        # Exclude targets from features.
        training_data = training_data.drop(columns=[self.target_column_name])
        validation_data = validation_data.drop(
            columns=[self.target_column_name])
        testing_data = testing_data.drop(columns=[self.target_column_name])

        # Preprocess split datasets.
        training_data_scaled, training_cols, scaler = self._preprocess_dataset(
            training_data)
        validation_data_scaled, _, _ = self._preprocess_dataset(
            validation_data, training_cols, scaler)
        testing_data_scaled, _, _ = self._preprocess_dataset(
            testing_data, training_cols, scaler)

        training_targets_np: np.ndarray = training_targets.to_numpy()

        # Set class instance properties.
        self.training_data = training_data_scaled
        self.validation_data = validation_data_scaled
        self.testing_data = testing_data_scaled
        self.training_targets = training_targets_np
        self.validation_targets = validation_targets
        self.testing_targets = testing_targets

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
        return list(set(candidate_k_values))

    def _split_dataset(
        self,
        dataset: pd.DataFrame,
        test_size: float = 0.2,
    ) -> list[pd.DataFrame]:
        '''Splits dataset into training, validation and test sets.

        Keyword arguments:
        dataset -- the dataset to split. 
        test_size -- fraction of data to use for testing (same fig used for validation).
        '''

        # Split dataset into training+validation and test sets.
        training_validation_data, testing_data = train_test_split(
            dataset, test_size=test_size, random_state=42
        )

        # Get validation size (should be same percentage as size as testing data).
        validation_size: float = len(
            testing_data) / len(training_validation_data)

        # Split training+validation data into training and validation sets.
        training_data, validation_data = train_test_split(
            training_validation_data, test_size=validation_size, random_state=42
        )

        return [training_data, validation_data, testing_data]

    def _preprocess_dataset(
        self,
        dataset: pd.DataFrame,
        training_cols: pd.Index | None = None,
        scaler: StandardScaler | None = None
    ) -> tuple[np.ndarray, pd.Index, StandardScaler]:
        '''Preprocesses data and returns the cols and scaler used for the dataset.

        Keyword arguments:
        dataset -- the dataset we wish to preprocess.
        training_cols -- defaults to None; used to account for missing cols post split.
        scaler -- the StandardScaler used to scale the data, init if None is passed.
        '''

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

        return (dataset_scaled, training_cols, scaler)

    def _align_test_cols(
        self,
        training_cols: pd.Index,
        testing_dataset: pd.DataFrame
    ) -> pd.DataFrame:
        '''Aligns testing_dataset cols with those of training_dataset.

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

        # Compute euclidean distances using vectorized operations.
        distances = np.sqrt(np.sum((dataset - example_to_predict)**2, axis=1))

        # Get indices of the k smallest distances.
        indices = np.argpartition(distances, k)[:k]

        # Return mean of corresponding target values.
        return float(target_column[indices].mean())

    def knn_classifier(
            self,
            example_to_predict: np.ndarray,
            dataset: np.ndarray,
            target_column: np.ndarray,
            k: int = 3
    ) -> str:
        '''Predicts the class label of an example using kNN.

        Keyword arguments:
        example_to_predict -- the example we are running the classification on.
        dataset -- the dataset to get the nearest neighbors from.
        target_column -- column w/ the class labels of the examples in the dataset.
        k -- the number of closest neighbors to use in the mode calculation.
        '''

        # Compute euclidean distances using vectorized operations.
        distances = np.sqrt(np.sum((dataset - example_to_predict)**2, axis=1))

        # Get indices of the k smallest distances
        indices = np.argpartition(distances, k)[:k]

        # Find the mode of the target values
        values, counts = np.unique(target_column[indices], return_counts=True)
        most_frequent = values[np.argmax(counts)]

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
    ):
        '''Returns the accuracy of a KNN classifier.

        Keyword arguments:
        k -- the value of k for the KNN classification.
        training_dataset -- the dataset the neighbors are taken from.
        testing_dataset -- the validation or test dataset to get the MAE for.
        training_targets -- the target values of training_dataset.
        testing_targets -- the target values of testing_dataset.
        '''

        # Create a list to store predictions for each example in the testing data.
        predictions: list[str] = []

        # For each example in the testing data.
        for example in testing_dataset:
            # Predict the class for this example using the kNN classifier.
            predicted_class: str = self.knn_classifier(
                example, training_dataset, training_targets, k)
            predictions.append(predicted_class)

        # Calculate the accuracy.
        accuracy: float = accuracy_score(testing_targets, predictions)

        # Return the accuracy.
        return accuracy


    def evaluate_knn(self) -> float:
        '''Returns error or accuracy rate (depending on task) of kNN on a dataset.'''
        # If it's a regression task.
        if self.regressor_or_classifier == 'regressor':

            # Set the best_mae to positive infinity.
            best_mae_for_validation: float = float('inf')

            # For each canidate k value, get its MAE.
            for candidate_k in self.candidate_k_values:

                mae: float = self.get_mae_of_knn_regressor(
                    candidate_k, self.training_data,
                    self.validation_data, self.training_targets,
                    self.validation_targets
                )

                # If better than best, update best_mae and best_k.
                if mae < best_mae_for_validation:
                    best_mae_for_validation = mae
                    self.best_k = candidate_k

            # Default value for best_k if left undefined.
            if not self.best_k:
                self.best_k = 3
                
            # Merge validation and training data so we have more data for testing.
            self.merged_val_training_data = np.vstack(
                [self._training_data, self.validation_data])
            self.merged_val_training_targets = np.concatenate(
                [self.training_targets, self.validation_targets])

            # Get MAE of test data when neighbors are gotten from training + validation.
            return self.get_mae_of_knn_regressor(
                self.best_k, self.merged_val_training_data,
                self.testing_data, self.merged_val_training_targets,
                self.testing_targets
            )

        # If it is a classification task.
        else:

            # Set the best_accuracy to positive infinity.
            best_accuracy_for_validation: float = float('-inf')

            # For each canidate k value, get its accuracy.
            for candidate_k in self.candidate_k_values:

                accuracy: float = self.get_accuracy_of_knn_classifier(
                    candidate_k, self.training_data,
                    self.validation_data, self.training_targets,
                    self.validation_targets)

                # If better than best, update best_accuracy and best_k.
                if accuracy > best_accuracy_for_validation:
                    best_accuracy_for_validation = accuracy
                    self.best_k = candidate_k

            # Default value for best_k if left undefined.
            if not self.best_k:
                self.best_k = 3

            # Merge validation and training data so we have more data for testing.
            self.merged_val_training_data = np.vstack(
                [self.training_data, self.validation_data])
            self.merged_val_training_targets = np.concatenate(
                [self.training_targets, self.validation_targets])

            # Get MAE of test data when neighbors are gotten from training + validation.
            return self.get_accuracy_of_knn_classifier(
                self.best_k, self.merged_val_training_data,
                self.testing_data, self.merged_val_training_targets,
                self.testing_targets
            )

#test = KNNHarness('regressor', 'datasets/abalone.data', 'Rings')
#test = KNNHarness('classifier', 'datasets/custom_cleveland.data', 'num')
#print(test.evaluate_knn())