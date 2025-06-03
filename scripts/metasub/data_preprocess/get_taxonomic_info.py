# Import libraries
import pandas as pd
import numpy as np
from Bio import Entrez
import time
import argparse

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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A script to get the taxonomic information of the species.")
    parser.add_argument('-d','--data_path',type=str,required=True,help='Enter the complete metadata. This file should contain all the species.')
    
    args = parser.parse_args()


    # Read the data
    df = pd.read_csv(args.data_path)
    df = df.iloc[:,42:]

    # Species to query
    species_to_query = df.columns

    # --- Configuration ---
    # IMPORTANT: Replace with your actual email and (optionally) API key
    my_email = "1ms19bt011@gmail.com" # Replace with your email
    my_api_key = None  # Replace with your API key, or set to None

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
    lineage_df.columns = [f'Rank_{i+1}' for i in range(lineage_df.shape[1])]

    # Save to CSV
    output_filename = "species_lineages.csv"
    lineage_df.to_csv(output_filename)
    print(f"\nLineage data saved to {output_filename}")

    # Example: Display first few entries of the DataFrame
    print("\nFirst 5 entries of the lineage DataFrame:")
    print(lineage_df.head())