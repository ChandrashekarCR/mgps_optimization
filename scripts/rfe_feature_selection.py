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
import argparse
import sys


# Global variables
DEFAULT_OUTPUT_FILE = os.getcwd()

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
            - all_supports (dict): Dictionary where keys are (n_features, fold) and values are boolean arrays indicating feature support.
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

        support = pipe[0].support_ # Get the boolean mask of selected features

        return n_features, fold, score, support

    start_time = time.time()
    X_ray = ray.put(X)
    y_ray = ray.put(y)
    tasks = [evaluate_rfe_remote.remote(n_features, fold, X_ray, y_ray)
             for n_features in n_features_options for fold in range(cv)]

    results = []
    all_supports = {}
    with tqdm(total=total_iterations, desc='Parallel RFE + Cross-validation') as pbar:
        while tasks:
            done, tasks = ray.wait(tasks, num_returns=1)
            result = ray.get(done[0])
            n_features_res, fold_res, score_res, support_res = result
            results.append((n_features_res, score_res))  # (n_features, score)
            all_supports[(n_features_res, fold_res)] = support_res
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

    return best_params, best_accuracy, results_df, elapsed_time, all_supports, X

def check_output_path(output_file_path: str) -> str:
    """
    Validates the output file path and handles the following scenarios:
    
    1. If the provided path is a directory, the output will be saved as 'default_output.txt' in that directory.
    2. If the path includes a valid directory and filename, use the specified filename.
    3. If the user provides only a filename without a directory, use the current working directory.
    4. If the directory is invalid, raise an error and exit.

    Args:
        output_file_path (str): The path where the output file should be saved.

    Returns:
        str: The final output file path.

    Raises:
        FileNotFoundError: If the provided directory does not exist.
    """

    # Normalize the path and handle relative paths
    output_file_path = os.path.normpath(output_file_path)
    print(f"Normalized path: {output_file_path}")

    # Get the current working directory
    wd = os.getcwd()

    # Case 1: If the provided path is a directory, use 'default_output.txt'
    if os.path.isdir(output_file_path):
        print(f"'{output_file_path}' is a directory. Using 'default_output.txt' as the filename.")
        return os.path.join(output_file_path, 'default_output.txt')

    # Extract the directory and filename from the provided path
    directory = os.path.dirname(output_file_path)
    filename = os.path.basename(output_file_path)

    # Case 2: If only a filename is provided without a directory, use the current working directory
    if not directory:
        print(f"No directory provided. Using the current directory with filename '{filename}'.")
        return os.path.join(wd, filename)

    # Case 3: If the directory exists, use the provided filename
    if os.path.isdir(directory):
        print(f"Valid directory found. Using '{filename}' as the output file.")
        return output_file_path

    # Case 4: If the directory is not valid, raise an error
    try:
        raise FileNotFoundError(f"The directory '{directory}' is invalid.")
    except FileNotFoundError as e:
        print(f"{e} Please try again...")
        sys.exit(1)



if __name__ == "__main__":

    # Parse command line arguements
    parser = argparse.ArgumentParser(
        prog='rfe_feature_selection.py',
        usage='python rfe_feature_selection.py -i <input processed metadata file> -o <output_file<optional>>',
        description='This program is used to calculate the best features that can be used to train the neural network model.'        
    )
    parser.add_argument('-i','--input_csv_file',dest='csv_file',help='Enter the processed metadata csv file.')
    parser.add_argument('-s','--start_taxa_column',dest='start_index',help='Enter the starting column number of the taxa. (python indexing)',type=int)
    parser.add_argument('-p','--prediction_column',dest='prediction_column',help='Enter the column to be predicted on', type=str)
    parser.add_argument('-o','--output_file',dest='output_file',help='Enter the path of the output file and the name.',default=DEFAULT_OUTPUT_FILE,nargs='?')
    parser.add_argument('--num_cpus', dest='num_cpus', help='Number of CPUs for Ray',type=int, default=None)

    # Parse arguements
    args= parser.parse_args()


    data = pd.read_csv(args.csv_file)
    print(f"\nData loaded successfully. Shape: {data.shape}")
    

    best_parameters, best_score, all_results, time_taken, all_supports, return_data = parallel_rfe_feature_selection(
            X=data.iloc[:,args.start_index:],
            y=data[args.prediction_column],
            n_jobs=-1,  # Use all available cores for RandomForest within each Ray task
            random_state=123,
            cv=5,
            subsets=[200],# [200,300,400,500,1000,1500]
            remove_correlated=True,
            correlation_threshold=0.95,
            num_cpus=args.num_cpus  # Limit Ray to 4 CPUs for this example
        )

    print(f'\nBest params: {best_parameters}')
    print(f'Best accuracy: {best_score:.6f}')
    print(f'Mean accuracy for all tested feature subsets:\n{all_results}')
    print(f'Total time taken: {time_taken:.2f} seconds')

    # Get the support_ for the best performing number of features (across all folds)
    best_n_features_from_params = best_parameters['rfe__n_features_to_select']
    best_supports = {k: v for k, v in all_supports.items() if k[0] == best_n_features_from_params}

    # Store the support arrays for the best models in this variable
    support_for_best_models = best_supports

    nn_data = pd.concat([data[return_data.columns[support_for_best_models[list(support_for_best_models.keys())[0]]]],data[[args.prediction_column,'continent','latitude','longitude']]],axis=1)
    
    # Check if the output file path is valid
    validated_output_file_path = check_output_path(args.output_file)
    print(validated_output_file_path)
    
    # Save the data into csv format
    nn_data.to_csv(path_or_buf=validated_output_file_path,index=False)


# python rfe_feature_selection.py -i ../results/processed_metasub.csv -o ../results/metasub_training_testing_400.csv -s 42 -p city
