import numpy as np
from src.harness import KNNHarness
import sys


class WeightedKNNHarness(KNNHarness):
    def __init__(
        self,
        regressor_or_classifier: str,
        dataset_file_path: str,
        target_column_name: str,
        test_size: float = 0.2,
        missing_values: list[str] = ["?"],
    ):
        """Initialises a Weighted kNN Harness.

        Keyword arguments:
        regressor_or_classifier -- what kNN it runs 'regressor' | 'classifier'.
        dataset_file_path -- file path to the dataset to run the kNN on.
        target_column_name -- name of the column we are predicting.
        test_size -- what percentage of the dataset to reserve for testing.
        missing_values -- strings denoting missing values in the dataset.
        """

        # Call the constructor of the super class.
        super().__init__(
            regressor_or_classifier,
            dataset_file_path,
            target_column_name,
            test_size,
            missing_values,
        )

    def _knn_classifier(
        self,
        example_to_predict: np.ndarray,
        dataset: np.ndarray,
        target_column: np.ndarray,
        k: int = 3,
    ) -> str:
        """Predicts the class label of an example using weighted kNN.

        Keyword arguments:
        example_to_predict -- the example we are running the classification on.
        dataset -- the dataset to get the nearest neighbors from.
        target_column -- column w/ the class labels of the examples in dataset.
        k -- the number of closest neighbors to use in the mode calculation.
        """

        # Compute euclidean distances using vectorized operations.
        distances: np.ndarray = np.sqrt(
            np.sum((dataset - example_to_predict) ** 2, axis=1)
        )

        # Add a small value to avoid division by zero.
        distances = distances + sys.float_info.epsilon

        # Get indices of the k smallest distances.
        indices: np.ndarray = np.argpartition(distances, k)[:k]

        # Compute weights as the inverse of distances.
        weights: np.ndarray = 1 / distances[indices]

        # Calculate weighted frequency of each class in k neighbors.
        class_weights: dict[str, float] = {}

        index: int
        weight: float
        label: str

        for index, weight in zip(indices, weights):
            label = target_column[index]
            class_weights[label] = class_weights.get(label, 0) + weight

        # Determine the class with the highest weighted frequency
        most_frequent: str = max(class_weights, key=lambda x: class_weights[x])

        return most_frequent

    def _knn_regressor(
        self,
        example_to_predict: np.ndarray,
        dataset: np.ndarray,
        target_column: np.ndarray,
        k: int = 3,
    ) -> float:
        """Predicts the target value of an example using weighted kNN.

        Keyword arguments:
        example_to_predict -- the example we are running the regression on.
        dataset -- the dataset to get the nearest neighbors from.
        target_column -- column w/ target values of examples in the dataset.
        k -- the number of closest neighbors to use in weighted mean calc.
        """

        # Compute euclidean distances using vectorized operations.
        distances: np.ndarray = np.sqrt(
            np.sum((dataset - example_to_predict) ** 2, axis=1)
        )

        # Add a small value to avoid division by zero.
        distances = distances + sys.float_info.epsilon

        # Get indices of the k smallest distances.
        indices: np.ndarray = np.argpartition(distances, k)[:k]

        # Compute the weights based on the inverse of the distance.
        weights: np.ndarray = 1.0 / distances[indices]

        # Compute the weighted mean of the corresponding target values.
        weighted_mean: float = np.sum(weights * target_column[indices]) / np.sum(
            weights
        )

        return float(weighted_mean)


# test = KNNHarness('regressor', 'datasets/abalone.data', 'Rings')
# print(test.evaluate())
# test = WeightedKNNHarness('regressor', 'datasets/abalone.data', 'Rings')
# print(test.evaluate())
