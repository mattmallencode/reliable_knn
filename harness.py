import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.metrics import mean_absolute_error  # type: ignore


def preprocess_dataset(
        dataset: pd.DataFrame,
) -> pd.DataFrame:
    '''Preprocesses data, converts numeric cols and one-hot encodes categorical cols.

    Keyword arguments:
    dataset -- the dataset we wish to preprocess.
    '''

    # Columns to one-hot encode.
    cols_to_encode: list[str] = []

    # If marked with '(cat)' it is a categorical column.
    # Else, try to convert columns to numeric. If unsuccessful, it's a categorical col.
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

    # Reset indices.
    dataset.reset_index(inplace=True, drop=True)

    return dataset


def align_test_cols(
        training_dataset: pd.DataFrame,
        testing_dataset: pd.DataFrame
) -> pd.DataFrame:
    '''Aligns testing_dataset cols with those of training_dataset.

    Keyword arguments:
    training_dataset -- the training dataset.
    testing_dataset -- the dataset we wish to align to training_dataset.
    '''

    # Get the cols in training that are not in testing.
    missing_cols: set[str] = set(
        training_dataset.columns) - set(testing_dataset.columns)

    # Create a DataFrame for the missing columns filled with zeroes.
    missing_data: pd.DataFrame = pd.DataFrame(
        {col: np.zeros(len(testing_dataset)) for col in missing_cols})

    # Concatenate the original testing_dataset with the missing columns.
    testing_dataset = pd.concat([testing_dataset, missing_data], axis=1)

    # Make sure the ordering of the cols in both datasets matches.
    testing_dataset = testing_dataset[training_dataset.columns]

    # Return the aligned dataset.
    return testing_dataset


def split_dataset(
        dataset: pd.DataFrame,
        test_size: float = 0.2,
) -> list[pd.DataFrame]:
    '''Splits dataset into training, validation and test sets.

    Keyword arguments:
    dataset -- the dataset to split. 
    test_size -- fraction of dataset to use for testing (same fig used for validation).
    '''

    # Split dataset into training+validation and test sets.
    training_validation_data, testing_data = train_test_split(
        dataset, test_size=test_size, random_state=42
    )

    # Get validation size (should be same percentage as size as testing data).
    validation_size: float = len(testing_data) / len(training_validation_data)

    # Split training+validation data into training and validation sets.
    training_data, validation_data = train_test_split(
        training_validation_data, test_size=validation_size, random_state=42
    )

    return [training_data, validation_data, testing_data]


def knn_regressor(
        example_to_predict: np.ndarray,
        dataset: np.ndarray,
        target_column: np.ndarray,
        k: int = 3
) -> float:
    '''Predicts the target value of an example using kNN.

    Keyword arguments:
    example_to_predict -- the example we are running the regression on.
    dataset -- the dataset to get the nearest neighbors from.
    target_column -- column containing the target values of the examples in the dataset.
    k -- the number of closest neighbors to use in the mean calculation.
    '''

    # Compute euclidean distances using vectorized operations.
    distances = np.sqrt(np.sum((dataset - example_to_predict)**2, axis=1))

    # Get indices of the k smallest distances.
    indices = np.argpartition(distances, k)[:k]

    # Return mean of corresponding target values.
    return float(target_column[indices].mean())


def knn_classifier(
        example_to_predict: np.ndarray,
        dataset: np.ndarray,
        target_column: np.ndarray,
        k: int = 3
) -> str:
    '''Predicts the class label of an example using kNN.

    Keyword arguments:
    example_to_predict -- the example we are running the classification on.
    dataset -- the dataset to get the nearest neighbors from.
    target_column -- column containing the class labels of the examples in the dataset.
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


def get_candidate_k_values(num_examples: int) -> list[int]:
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


def get_mae_of_knn_regressor(
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
        predicted_value = knn_regressor(
            example, training_dataset, training_targets, k)
        predictions.append(predicted_value)

    # Convert predictions to a pandas DataFrame.
    predictions_series: pd.DataFrame = pd.DataFrame(predictions)

    # Calculate mean absolute error between predictions and true values.
    mae = mean_absolute_error(testing_targets, predictions_series)

    return mae


def get_accuracy_of_knn_classifier(
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
    testing_dataset -- the validation or test dataset to get the MAE for.s
    training_targets -- the target values of training_dataset.
    testing_targets -- the target values of testing_dataset.
    '''

    # Initialise some variables to keep track of accuracy.
    num_correct_class_predictions: int = 0
    num_total_class_predictions: int = testing_dataset.shape[0]

    # For each example in the testing data
    for index, example in enumerate(testing_dataset):

        # Predict the class for this example using the kNN classifier.
        predicted_class: str = knn_classifier(
            example, training_dataset, training_targets, k)

        # Get the actual class for this example.
        actual_class: str = str(testing_targets.iloc[index])

        # Increment correct prediction counter if predicted class is actual class.
        if str(predicted_class) == actual_class:
            num_correct_class_predictions += 1

    # Calculate the accuracy for this kNN classification.
    accuracy: float = num_correct_class_predictions / num_total_class_predictions

    # Return the accuracy.
    return accuracy


def evaluate_knn(
        regressor_or_classifier: str,
        dataset_file_path: str,
        target_column_name: str,
        test_size: float = 0.2,
        missing_values: list[str] = ['?']
) -> float:
    '''Returns the error or accuracy rate (depending on task) of kNN on a dataset.

    Keyword arguments:
    regressor_or_classifier -- pass regressor or classifier depending on the KNN task.
    dataset_file_path -- file path to the dataset to run the kNN on.
    target_column_name -- name of the target / class column i.e. the value to predict.
    test_size -- what percentage of the dataset to reserve for testing.
    missing_values -- strings denoting missing values in the dataset.
    '''

    # First the load the dataset.
    dataset: pd.DataFrame = pd.read_csv(dataset_file_path)

    # Get a range of candidate k values to validate.
    candidate_k_values: list[int] = get_candidate_k_values(len(dataset))

    training_data: pd.DataFrame
    testing_data: pd.DataFrame
    training_targets: pd.Series
    testing_targets: pd.Series
    validation_data: pd.DataFrame
    validation_targets: pd.Series
    merged_val_training_data: np.ndarray
    merged_val_training_targets: np.ndarray

    # Replace specified missing values with NaN.
    dataset.replace(missing_values, np.nan, inplace=True)

    # Drop rows containing NaN.
    dataset.dropna(inplace=True)

    # Split dataset into training, validation, and test datasets.
    training_data, validation_data, testing_data = split_dataset(
        dataset, test_size)

    # Get targets for each of the subsets of the data.
    training_targets = training_data[target_column_name]
    validation_targets = validation_data[target_column_name]
    testing_targets = testing_data[target_column_name]

    # Exclude targets from features.
    training_data = training_data.drop(columns=[target_column_name])
    validation_data = validation_data.drop(columns=[target_column_name])
    testing_data = testing_data.drop(columns=[target_column_name])

    # Preprocess split datasets.
    training_data = preprocess_dataset(training_data)
    validation_data = preprocess_dataset(validation_data)
    testing_data = preprocess_dataset(testing_data)

    # Add any missing columns to the non-training datasets.
    validation_data = align_test_cols(training_data, validation_data)
    testing_data = align_test_cols(training_data, testing_data)

    # Convert to NumPy arrays to leverage vectorisation.
    training_data_np: np.ndarray = training_data.to_numpy()
    testing_data_np: np.ndarray = testing_data.to_numpy()
    training_targets_np: np.ndarray = training_targets.to_numpy()
    validation_data_np: np.ndarray = validation_data.to_numpy()

    # Value of the best_k_for_validation (to be validated later).
    best_k_for_validation: int | None = None

    # If it's a regression task.
    if regressor_or_classifier.lower() == 'regressor':

        # Set the best_mae to positive infinity.
        best_mae_for_validation: float = float('inf')

        # For each canidate k value, get its MAE.
        for candidate_k in candidate_k_values:

            mae: float = get_mae_of_knn_regressor(
                candidate_k, training_data_np,
                validation_data_np, training_targets_np,
                validation_targets
            )

            # If better than best, update best_mae and best_k_for_validation.
            if mae < best_mae_for_validation:
                best_mae_for_validation = mae
                best_k_for_validation = candidate_k

        # Default value for best_k_for_validation if left undefined.
        if not best_k_for_validation:
            best_k_for_validation = 3

        # Merge validation and training data so we have more data for testing.
        merged_val_training_data = np.vstack(
            [training_data_np, validation_data_np])
        merged_val_training_targets = np.concatenate(
            [training_targets_np, validation_targets])

        # Get MAE of testing data when neighbors are gotten from training + validation.
        return get_mae_of_knn_regressor(
            best_k_for_validation, merged_val_training_data,
            testing_data_np, merged_val_training_targets,
            testing_targets
        )

    # If it is a classification task.
    elif regressor_or_classifier.lower() == 'classifier':

        # Set the best_accuracy to positive infinity.
        best_accuracy_for_validation: float = float('-inf')

        # For each canidate k value, get its accuracy.
        for candidate_k in candidate_k_values:

            accuracy: float = get_accuracy_of_knn_classifier(
                candidate_k, training_data_np,
                validation_data_np, training_targets_np,
                validation_targets)

            # If better than best, update best_accuracy and best_k_for_validation.
            if accuracy > best_accuracy_for_validation:
                best_accuracy_for_validation = accuracy
                best_k_for_validation = candidate_k

        # Default value for best_k_for_validation if left undefined.
        if not best_k_for_validation:
            best_k_for_validation = 3

        # Merge validation and training data so we have more data for testing.
        merged_val_training_data = np.vstack(
            [training_data_np, validation_data_np])
        merged_val_training_targets = np.concatenate(
            [training_targets_np, validation_targets])

        # Get MAE of testing data when neighbors are gotten from training + validation.
        return get_accuracy_of_knn_classifier(
            best_k_for_validation, merged_val_training_data,
            testing_data_np, merged_val_training_targets,
            testing_targets
        )

    # Raise Exception if the user passed incorrent regressor_or_classifier value.
    raise ValueError(
        'regressor_or_classifier must be set to "regressor" or "classifier" ' +
        f'not "{regressor_or_classifier}"'
    )


# print(evaluate_knn('regressor', 'datasets/abalone.data', 'Rings'))
# print(evaluate_knn('classifier', 'datasets/processed.cleveland.data', 'num'))
