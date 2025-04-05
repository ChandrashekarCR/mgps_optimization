# Import libraries
import ray
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
import numpy as np
import time
from tqdm import tqdm
import os


def parallel_rfe_feature_selection(X: pd.DataFrame, y: pd.Series, n_jobs: int = 1, random_state: int = 123,
                                   cv: int = 10, subsets: list = None, remove_correlated: bool = True,
                                   correlation_threshold: float = 0.98, num_cpus: int = None):
    """
    Performs parallel Recursive Feature Elimination (RFE) with cross-validation to select the best feature subset.

    Args:
        X (pd.DataFrame): DataFrame of features.
        y (pd.Series): Series of the target variable.
        n_jobs (int): Number of jobs for the base estimator (RandomForestClassifier).
        random_state (int): Random state for reproducibility.
        cv (int): Number of cross-validation folds.
        subsets (list, optional): List of feature subset sizes to evaluate. If None, default subsets are used. Defaults to None.
        remove_correlated (bool, optional): Whether to remove highly correlated features before RFE. Defaults to True.
        correlation_threshold (float, optional): Threshold for identifying highly correlated features. Defaults to 0.98.
        num_cpus (int, optional): Number of CPUs to use for Ray. If None, Ray will auto-detect. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - best_params (dict): Dictionary with the best RFE parameters.
            - best_accuracy (float): The best mean cross-validation accuracy achieved.
            - results_df (pd.DataFrame): DataFrame containing the mean accuracy for each feature subset size.
            - elapsed_time (float): Total time taken for the feature selection process.
    """
    if ray.is_initialized():
        ray.shutdown()
    ray.init(ignore_reinit_error=True, num_cpus=num_cpus)

    model = RandomForestClassifier(n_jobs=n_jobs, random_state=random_state)

    if remove_correlated:
        # Compute correlation matrix
        print("Calculating correlation matrix...")
        corr_matrix = X.corr()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Identify correlated features (above threshold)
        correlated_features = [column for column in upper_tri.columns if any(upper_tri[column] > correlation_threshold)]

        # Drop correlated features
        X = X.drop(columns=correlated_features)
        print(f"Correlated features removed: {len(correlated_features)}")

    # Determine default subset sizes if not provided
    num_features = X.shape[1]
    if subsets is None:
        subsets = [num_features // 2, num_features // 4, num_features // 8, num_features // 16, num_features // 32, num_features // 64]
        subsets = [s for s in subsets if s > 0]  # Remove non-positive values

    n_features_options = sorted(list(set(subsets))) # Ensure unique and sorted subset sizes
    total_iterations = len(n_features_options) * cv

    print(f"\nStarting RFE with subsets of features: {n_features_options}")

    # Define remote function for parallel execution
    @ray.remote
    def evaluate_rfe_remote(n_features, fold, X_remote, y_remote):
        """Performs RFE feature selection and evaluates performance for a given fold."""
        pipe = make_pipeline(RFE(estimator=model, n_features_to_select=n_features, step=10))

        # We use the stratified K fold to split the data into training and validation sets
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=fold)
        train_index, test_index = list(skf.split(X_remote, y_remote))[fold]

        # train_index and test_index contain the index values for extracting training and testing data
        X_train = X_remote.iloc[train_index, :]
        X_test = X_remote.iloc[test_index, :]
        y_train = y_remote.iloc[train_index]
        y_test = y_remote.iloc[test_index]

        # Fit the model using the training data and then evaluate the score based on the testing data
        pipe.fit(X_train, y_train)
        score = pipe.score(X_test, y_test)

        return n_features, fold, score

    start_time = time.time()
    X_ray = ray.put(X)
    y_ray = ray.put(y)
    tasks = [evaluate_rfe_remote.remote(n_features, fold, X_ray, y_ray)
             for n_features in n_features_options for fold in range(cv)]

    results = []
    with tqdm(total=total_iterations, desc='Parallel RFE + Cross-validation') as pbar:
        while tasks:
            done, tasks = ray.wait(tasks, num_returns=1)
            result = ray.get(done[0])
            results.append((result[0], result[2]))  # (n_features, score)
            pbar.update(1)

    # Aggregate mean accuracy for each feature subset
    results_df = pd.DataFrame(results, columns=["n_features", "accuracy"])
    results_df = results_df.groupby("n_features").mean().reset_index()

    # Find best feature subset
    best_row = results_df.loc[results_df["accuracy"].idxmax()]
    best_n_features = int(best_row["n_features"])
    best_accuracy = best_row["accuracy"]

    end_time = time.time()
    elapsed_time = end_time - start_time

    best_params = {"rfe__n_features_to_select": best_n_features}

    ray.shutdown()

    return best_params, best_accuracy, results_df, elapsed_time


if __name__ == "__main__":
    

best_parameters, best_score, all_results, time_taken = parallel_rfe_feature_selection(
        X=X,
        y=y,
        n_jobs=-1,  # Use all available cores for RandomForest within each Ray task
        random_state=123,
        cv=5,
        subsets=[50, 100, 200, 300, 500],
        remove_correlated=True,
        correlation_threshold=0.95,
        num_cpus=50  # Limit Ray to 4 CPUs for this example
    )

print(f'\nBest params: {best_parameters}')
print(f'Best accuracy: {best_score:.6f}')
print(f'Mean accuracy for all tested feature subsets:\n{all_results}')
print(f'Total time taken: {time_taken:.2f} seconds')