from src.harness import KNNHarness
from src.renn import RENNHarness
from src.bbnr import BBNRHarness
from src.noise_complaints import NoiseComplaintsHarness
from weighted_harness import WeightedKNNHarness
import pandas as pd
from scipy.stats import ttest_rel, friedmanchisquare
import scikit_posthocs as sp
import numpy as np

harness_types: list[str] = ["KNN", "WEIGHT", "RENN", "BBNR", "NCkNN"]
noise_levels: list[int] = [0, 0.1, 0.2, 0.3, 0.4, 0.5]


def find_optimal_noise_level(
    dataset: str, regressor_or_classifier: str, target: str
) -> int:
    """Returns min noise_level stats significant performance degradation.

    dataset -- the name of the dataset.
    regressor_or_classifier -- whether its a regressor / classifier.
    target -- name of the target value / class column.
    """
    fold_performances = []

    f = open(f"results/{dataset.split('/')[-1]}_lambda", "w")

    f.close()

    # For each possible noise level, spin up a kNN harness.
    for noise_level in noise_levels:
        np.random.seed(42)
        harness = KNNHarness(regressor_or_classifier, dataset, target, noise_level)
        # Evaluate the harness @ this noise level, record fold results.
        fold_results, _ = harness.evaluate()
        fold_performances.append(fold_results)

    # Baseline i.e. 0% noise.
    baseline_fold_results = fold_performances[0]
    baseline_performance = np.mean(fold_performances[0])

    for i in range(1, len(fold_performances)):
        # Comparing each noise level's fold results to baseline fold.
        _, p = ttest_rel(baseline_fold_results, fold_performances[i])
        curr_performance = np.mean(fold_performances[i])

        degraded_performance: bool = False

        # "degraded performance" is task dependent.
        if regressor_or_classifier == "classifier":
            degraded_performance = curr_performance < baseline_performance
        else:
            degraded_performance = curr_performance > baseline_performance

        # Check if stats significant AND caused worse performance.
        if p < 0.05 and degraded_performance:
            # Return min noise level.
            return noise_levels[i]

    # Use 50% as the upper limit.
    return 0.5


def run_tests(regressor_or_classifier: str, dataset: str, target: str):
    """Evaluate each algo on the dataset and log stats test to file.

    Keyword arguments:
    regressor_or_classifier -- whether its a regressor/classifier.
    dataset -- the name of the dataset.
    target -- name of the target value / class column.
    """
    fold_results = {harness_type: [] for harness_type in harness_types}

    file = open(f"results/{dataset}.txt", "w")

    if regressor_or_classifier == "classifier":
        dataset_full_name = f"datasets/classification/{dataset}.data"
    else:
        dataset_full_name = f"datasets/regression/{dataset}.data"

    # Get min noise level to use.
    optimal_noise_level = find_optimal_noise_level(
        dataset_full_name, regressor_or_classifier, target
    )

    # Run each algo on the dataset.
    for harness_type in harness_types:
        harness = create_harness(
            harness_type,
            dataset_full_name,
            regressor_or_classifier,
            target,
            optimal_noise_level,
        )

        results_for_harness: tuple[list[float], float] = harness.evaluate()

        # Collecting fold results.
        fold_results[harness_type] = results_for_harness[0]

        # Writing individual algo results to file.
        file.write(f"Results for {harness_type}\n\n")
        file.write(f"Fold Results: {results_for_harness[0]}\n")
        file.write(f"Aggregate Result: {results_for_harness[1]}\n\n")

    # Convert fold results to a DataFrame for Friedman test
    results_df = pd.DataFrame(fold_results)

    # Perform Friedman test.
    stat, p = friedmanchisquare(*results_df.values.T)

    file.write(f"Friedman test statistic: {stat}, P-value: {p}\n\n")

    file.write(f"Noise level: {optimal_noise_level}\n\n")

    file.close()

    # If significant, perform Nemenyi's test

    if p < 0.05:
        nemenyi_results = sp.posthoc_nemenyi_friedman(results_df)
        with open(f"results/{dataset}.txt", "a") as f_append:
            nemenyi_results.to_csv(f_append, float_format="%.4f")


def create_harness(
    harness_type: str,
    dataset: str,
    regressor_or_classifier: str,
    target: str,
    noise_level: float,
):
    """Initialises and returns the specified KNNHarness.

    Keyword arguments:
    harness_type -- the harness to spin up.
    dataset -- the name of the dataset.
    regressor_or_classifier -- whether it is a regressor/classifier.
    target -- the name of the target value / class label column.
    noise_level -- fraction of examples in training / val to make noisy.
    """
    if harness_type == "KNN":
        return KNNHarness(regressor_or_classifier, dataset, target, noise_level)
    elif harness_type == "WEIGHT":
        return WeightedKNNHarness(regressor_or_classifier, dataset, target, noise_level)
    elif harness_type == "RENN":
        return RENNHarness(regressor_or_classifier, dataset, target, noise_level)
    elif harness_type == "NCkNN":
        return NoiseComplaintsHarness(
            regressor_or_classifier, dataset, target, noise_level
        )
    else:
        return BBNRHarness(regressor_or_classifier, dataset, target, noise_level)


run_tests("classifier", "zoo", "type")
run_tests("classifier", "iris", "class")
run_tests("classifier", "wine_origin", "class")
run_tests("classifier", "heart", "num")
run_tests("classifier", "votes", "class")
run_tests("classifier", "car", "class")

"""
run_tests("regressor", "abalone", "Rings")
run_tests("regressor", "automobile", "symboling")
run_tests("regressor", "red_wine_quality", "quality")
run_tests("regressor", "student_math", "G3")
run_tests("regressor", "student_portugese", "G3")
run_tests("regressor", "white_wine_quality", "quality")
"""
