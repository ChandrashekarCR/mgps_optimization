import pandas as pd
import numpy as np
import os
import argparse
import sys

# --- Data Loading Module ---
def load_marine_taxa(file_path):
    """Loads the marine taxa data from a CSV file."""
    try:
        marine_taxa = pd.read_csv(file_path)
        return marine_taxa
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

# --- Data Cleaning and Preprocessing Module ---
def preprocess_marine_taxa(marine_taxa):
    """Cleans and preprocesses the marine taxa data."""
    if marine_taxa is None:
        return None

    marine_taxa = marine_taxa.fillna(0)
    marine_taxa['Sea'] = marine_taxa['Sea'].str.replace('[^A-Za-z0-9_]+', '_', regex=True).apply(lambda x: str(x).lower()).astype('category')
    marine_taxa = marine_taxa.drop(columns='Unnamed: 0', axis=1, errors='ignore') # Handle case where 'Unnamed: 0' might not exist
    return marine_taxa

# --- Feature Selection Module ---
def select_taxa(marine_taxa):
    """Selects the taxa columns and identifies zero-sum columns."""
    if marine_taxa is None:
        return None

    taxa_cols = marine_taxa.iloc[:, 1:-6].columns.tolist() # Adjusting index based on the typical structure
    cols_to_drop = []
    for col in taxa_cols:
        if marine_taxa[col].sum() == 0:
            cols_to_drop.append(col)

    return taxa_cols, cols_to_drop

# --- Data Normalization Module ---
def normalize_taxa_data(marine_taxa, taxa_cols, cols_to_drop):
    """Normalizes the taxa abundance data."""
    if marine_taxa is None:
        return None

    marine_taxa_normalized = marine_taxa.copy()
    taxa_cols_after_drop = [col for col in taxa_cols if col not in cols_to_drop]
    if taxa_cols_after_drop:
        marine_taxa_normalized[taxa_cols_after_drop] = marine_taxa[taxa_cols_after_drop].div(marine_taxa[taxa_cols_after_drop].sum(axis=1), axis=0)
    marine_taxa_normalized = marine_taxa_normalized.drop(columns=cols_to_drop, errors='ignore') # Handle cases where these cols might already be dropped
    return marine_taxa_normalized

# --- Main Processing Function ---
def process_marine_data(file_path, output_file_path=None):
    """Main function to process the marine data."""
    print('Processing marine data...')

    # Load data
    marine_taxa = load_marine_taxa(file_path)
    if marine_taxa is None:
        return

    # Preprocess data
    marine_taxa = preprocess_marine_taxa(marine_taxa)
    if marine_taxa is None:
        return

    # Select taxa and identify columns to drop
    taxa_cols, cols_to_drop = select_taxa(marine_taxa)

    # Normalize taxa data
    marine_data_processed = normalize_taxa_data(marine_taxa, taxa_cols, cols_to_drop)

    if marine_data_processed is not None:
        print('Marine data processed successfully.')
        if output_file_path:
            # Basic output path handling (you can enhance this with the check_output_path function)
            output_dir = os.path.dirname(output_file_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            marine_data_processed.to_csv(output_file_path, index=False)
            print(f'Processed marine data saved to: {output_file_path}')
        else:
            print('Processed marine data:')
            print(marine_data_processed.head())

def check_output_path(output_file_path: str) -> str:
    """
    Validates the output file path (same as in the metasub script).
    """
    output_file_path = os.path.normpath(output_file_path)
    wd = os.getcwd()

    if os.path.isdir(output_file_path):
        print(f"'{output_file_path}' is a directory. Using 'marine_processed.csv' as the filename.")
        return os.path.join(output_file_path, 'marine_processed.csv')

    directory = os.path.dirname(output_file_path)
    filename = os.path.basename(output_file_path)

    if not directory:
        print(f"No directory provided. Using the current directory with filename '{filename}'.")
        return os.path.join(wd, filename)

    if os.path.isdir(directory):
        print(f"Valid directory found. Using '{filename}' as the output file.")
        return output_file_path

    try:
        raise FileNotFoundError(f"The directory '{directory}' is invalid.")
    except FileNotFoundError as e:
        print(f"{e} Please try again...")
        sys.exit(1)

if __name__ == "__main__":
    # Change working directory (as in your notebook)
    os.chdir(os.getcwd()) # Assuming you run the script from the correct directory

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        prog='preprocess_marine.py',
        usage="python preprocess_marine.py -i <input marine taxa data> -o <output file <optional>>",
        description="This program preprocesses marine taxa data."
    )

    parser.add_argument('-i', '--input_file', dest='input_file', help='Enter the marine taxa data file path.')
    parser.add_argument('-o', '--output_file', dest='output_file', help='Enter the path for the output file.', default=None, nargs='?')

    args = parser.parse_args()

    if args.input_file:
        output_path = None
        if args.output_file:
            output_path = check_output_path(args.output_file)
        process_marine_data(args.input_file, output_path)
    else:
        print("Error: Please provide the input marine taxa data file path using the -i or --input_file argument.")
        sys.exit(1)