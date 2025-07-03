# Import libraries
import pandas as pd
import numpy as np
from Bio import Entrez
import time
import argparse
import sys
import os

def get_ncbi_lineage(species_list, email="your.email@example.com", api_key=None, delay_seconds=0.35):
    """
    Retrieves the full taxonomic lineage from NCBI for a list of species names.

    Args:
        species_list (list): A list of species names (e.g., ['Escherichia coli', 'Bacillus subtilis']).
        email (str): Your email address, required by NCBI.
        api_key (str, optional): Your NCBI API key. Improves request limits. Defaults to None.
        delay_seconds (float): Delay between requests to adhere to NCBI's usage policy.
                               Default is 0.35 seconds (for 3 requests/sec without API key).
                               Set to 0.1 for API key.

    Returns:
        dict: A dictionary where keys are species names and values are their taxonomic lineages
              as lists of strings (e.g., ['Bacteria', 'Proteobacteria', ...]).
              Returns an empty list for lineage if a species is not found.
    """
    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key
        delay_seconds = 0.1 # Increase rate limit to 10 requests/sec

    lineages = {}
    found_species = set()
    not_found_species = []

    print(f"Starting lineage retrieval for {len(species_list)} species...")

    for i, species_name in enumerate(species_list):
        if i > 0: # Add delay between requests
            time.sleep(delay_seconds)

        print(f"Processing species {i+1}/{len(species_list)}: {species_name}")

        try:
            # 1. Search for the species name in the Taxonomy database to get its Tax ID
            handle = Entrez.esearch(db="taxonomy", term=species_name, retmode="xml")
            record = Entrez.read(handle)
            handle.close()

            tax_ids = record["IdList"]

            if not tax_ids:
                print(f"  Warning: No Tax ID found for '{species_name}'. Skipping.")
                lineages[species_name] = []
                not_found_species.append(species_name)
                continue

            # Take the first Tax ID if multiple are returned (usually the most relevant)
            tax_id = tax_ids[0]

            # 2. Fetch the taxonomy record using the Tax ID
            handle = Entrez.efetch(db="taxonomy", id=tax_id, retmode="xml")
            tax_record = Entrez.read(handle)
            handle.close()

            # The lineage information is typically in the 'Lineage' field
            if tax_record and tax_record[0] and 'Lineage' in tax_record[0]:
                lineage_str = tax_record[0]['Lineage']
                # Split the lineage string into a list of ranks
                # NCBI lineage is usually semicolon-separated
                lineage_list = [rank.strip() for rank in lineage_str.split(';') if rank.strip()]
                lineages[species_name] = lineage_list
                found_species.add(species_name)
                print(f"  Lineage found: {'; '.join(lineage_list)}")
            else:
                print(f"  Warning: Lineage not found for '{species_name}' (Tax ID: {tax_id}). Skipping.")
                lineages[species_name] = []
                not_found_species.append(species_name)

        except Exception as e:
            print(f"  Error retrieving lineage for '{species_name}': {e}")
            lineages[species_name] = []
            not_found_species.append(species_name)

    print("\n--- Summary ---")
    print(f"Successfully retrieved lineage for {len(found_species)} species.")
    if not_found_species:
        print(f"Could not find lineage for {len(not_found_species)} species:")
        for sp in not_found_species:
            print(f"  - {sp}")

    return lineages


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

def create_higher_rank_features(rsa_df, lineage_df, ranks_to_add=['Family', 'Order', 'Class']):
    """
    Adds new feature columns to an RSA DataFrame based on higher taxonomic ranks.

    Args:
        rsa_df (pd.DataFrame): DataFrame where index is sample IDs and columns are species names,
                               with values being their Relative Sequence Abundance (RSA).
                               Example:
                                       Species1  Species2  Species3 ...
                                Sample1     0.1       0.0       0.2
                                Sample2     0.0       0.5       0.0
        lineage_df (pd.DataFrame): DataFrame containing taxonomic lineage for each species.
                                   Expected to have 'Species' as an index or column,
                                   and columns like 'Family', 'Order', 'Class', etc.
                                   Example:
                                           Family       Order         Class
                                Species1  FamilyA     OrderX      ClassP
                                Species2  FamilyA     OrderX      ClassP
                                Species3  FamilyB     OrderY      ClassQ
        ranks_to_add (list): A list of strings specifying the taxonomic ranks to aggregate.
                             These must match column names in `lineage_df`.
                             Defaults to ['Family', 'Order', 'Class'].

    Returns:
        pd.DataFrame: The original RSA DataFrame augmented with new columns
                      representing the aggregated RSA for the specified higher ranks.
                      New column names will be prefixed (e.g., 'Family_FamilyA').
    """

    # Ensure species names are consistent for merging
    # If lineage_df has 'Species' as a column, set it as index for easy lookup
    if 'Species' in lineage_df.columns:
        lineage_df = lineage_df.set_index('Species')

    # Ensure that the columns in rsa_df (species names) are in the lineage_df index
    # and that the requested ranks exist in lineage_df
    missing_species = [s for s in rsa_df.columns if s not in lineage_df.index]
    if missing_species:
        print(f"Warning: {len(missing_species)} species in RSA data not found in lineage data. They will be excluded from higher-rank aggregation: {missing_species[:5]}...")

    missing_ranks = [r for r in ranks_to_add if r not in lineage_df.columns]
    if missing_ranks:
        raise ValueError(f"The following specified ranks are not found in the lineage DataFrame columns: {missing_ranks}")


    # Create a DataFrame to store the new higher-rank features
    higher_rank_features = pd.DataFrame(index=rsa_df.index)

    for rank in ranks_to_add:
        # Get unique values for the current rank (e.g., all unique Family names)
        unique_ranks = lineage_df[rank].dropna().unique()

        for rank_name in unique_ranks:
            # Find all species belonging to this specific rank (e.g., all species in FamilyA)
            species_in_rank = lineage_df[lineage_df[rank] == rank_name].index.tolist()

            # Filter species that are actually present in the RSA data columns
            valid_species_in_rank = [s for s in species_in_rank if s in rsa_df.columns]

            if valid_species_in_rank:
                # Sum the RSA for all species within this rank for each sample
                # Handle cases where there might be only one species in the rank
                if len(valid_species_in_rank) == 1:
                    aggregated_rsa = rsa_df[valid_species_in_rank[0]]
                else:
                    aggregated_rsa = rsa_df[valid_species_in_rank].sum(axis=1)

                # Add the aggregated RSA as a new column
                # Prefix the column name to avoid conflicts with original species names
                new_col_name = f"{rank}_{rank_name}"
                higher_rank_features[new_col_name] = aggregated_rsa
            else:
                print(f"No species in '{rank_name}' ({rank}) found in RSA data. Skipping aggregation for this rank.")

    # Concatenate the original RSA DataFrame with the new higher-rank features
    augmented_rsa_df = pd.concat([rsa_df, higher_rank_features], axis=1)

    return augmented_rsa_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script to get the taxonomic information of the species.")
    parser.add_argument('-d', '--data_path', type=str, required=True, help='Enter the complete metadata. This file should contain all the species.')
    parser.add_argument('-o', '--output_file', type=str, required=True, help='Enter the output path for the lineage CSV.')
    parser.add_argument('-ao', '--augmented_output_file', type=str, default="augmented_rsa_data.csv",
                        help='Enter the output path for the augmented RSA DataFrame. Defaults to "augmented_rsa_data.csv".')

    args = parser.parse_args()

    # Determine the lineage output file path
    lineage_output_file_path = check_output_path(args.output_file)

    # --- Configuration ---
    my_email = "1ms19bt011@gmail.com"
    my_api_key = None # Set your API key here if you have one

    lineage_df = None # Initialize lineage_df

    # Check if lineage file already exists
    if os.path.exists(lineage_output_file_path):
        print(f"Lineage file '{lineage_output_file_path}' found. Reading existing data.")
        lineage_df = pd.read_csv(lineage_output_file_path, index_col='Species')
    else:
        print(f"Lineage file '{lineage_output_file_path}' not found. Fetching from NCBI...")
        # Read the data to get species list for fetching
        df_for_species_query = pd.read_csv(args.data_path)
        species_to_query = df_for_species_query.iloc[:, 42:].columns.tolist() # Convert to list for get_ncbi_lineage

        # Get lineages
        species_lineages = get_ncbi_lineage(
            species_to_query,
            email=my_email,
            api_key=my_api_key
        )

        # Convert to DataFrame for easier manipulation and saving
        lineage_df = pd.DataFrame.from_dict(species_lineages, orient='index')
        lineage_df.index.name = 'Species'
        # Give generic column names for lineage ranks
        # Ensure that empty lineages (lists) don't cause issues with column creation
        max_rank = 0
        if not lineage_df.empty:
            max_rank = lineage_df.apply(lambda x: len(x.dropna().tolist()), axis=1).max()
        
        lineage_df.columns = [f'Rank_{i+1}' for i in range(max_rank)]

        # Save to CSV
        lineage_df.to_csv(lineage_output_file_path)
        print(f"\nLineage data saved to {lineage_output_file_path}")

    # Now proceed with RSA data and augmentation
    df = pd.read_csv(args.data_path)
    rsa_df = df.iloc[:, 42:]

    # It's good practice to align the columns of rsa_df with the index of lineage_df
    # by taking only species present in both.
    common_species = list(set(rsa_df.columns) & set(lineage_df.index))
    rsa_df = rsa_df[common_species] # Filter rsa_df to only include species with known lineages

    # Define the ranks to include based on your expected NCBI output
    # Since you observed 'Rank_5' and 'Rank_6', and NCBI often returns up to species/strain,
    # it's safer to include more ranks and let the function handle missing ones.
    # You can adjust this list based on what ranks you are most interested in.
    ranks_to_include = ['Rank_2','Rank_3','Rank_4','Rank_5','Rank_6','Rank_7','Rank_8','Rank_9','Rank_10','Rank_11','Rank_12']

    augmented_rsa_df = create_higher_rank_features(rsa_df, lineage_df, ranks_to_add=ranks_to_include)
    print(f"\nAugmented RSA DataFrame with higher-rank features:")
    # print(augmented_rsa_df.head()) # Suppress verbose output
    print("\nAugmented RSA DataFrame shape:", augmented_rsa_df.shape)

    # Concatenate original metadata columns with the augmented RSA data
    final_augmented_df = pd.concat([df.iloc[:, :42], augmented_rsa_df], axis=1)
    print("\nFinal Augmented DataFrame shape:", final_augmented_df.shape)

    # Save the augmented DataFrame
    # Check if the augmented output file path is valid
    augmented_output_file_path = check_output_path(args.augmented_output_file)
    final_augmented_df.to_csv(augmented_output_file_path, index=False) # index=False to avoid writing DataFrame index as a column
    print(f"\nAugmented RSA data saved to {augmented_output_file_path}")
