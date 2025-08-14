# Importing libraries
import pandas as pd
import numpy as np
import os
import argparse
import sys

# Global variables
DEFAULT_OUTPUT_FILE = os.getcwd()


def process_metasub(metadata, taxa_abundance):
    
    # Read the metadata and abundance for each taxa for the metasub data.
    print('Reading the datasets...')
    complete_meta = pd.read_csv(metadata)
    taxa_abund = pd.read_csv(taxa_abundance)
    taxa_abund = taxa_abund.drop_duplicates(subset=['uuid'])

    print('Appling filtering criterias to get processed data...')

    # Merge the bacterial and metadata
    metasub_data = pd.merge(complete_meta,taxa_abund,on='uuid')

    # Remove control samples
    control_cities = {'control','other_control','neg_control','other','pos_control'}
    control_types = {'ctrl cities','negative_control','positive_control'}

    mask = metasub_data['city'].isin(control_cities) | metasub_data['control_type'].isin(control_types)
    metasub_data = metasub_data[~mask].copy()

    #Re-label london boroughs
    metasub_data.loc[metasub_data['city'].isin(['kensington','islington']),'city'] = 'london'

    # Remove sparse sample locations and doubtful samples
    city_counts = metasub_data['city'].value_counts()
    small_cities = city_counts[city_counts<8].index.tolist()
    remove_samples = metasub_data['city'].isin(['antarctica']+small_cities)
    metasub_data = metasub_data[~remove_samples]

    # Correct the identified mislabeling of data
    kyiv_filter = metasub_data['city'] == 'kyiv'
    metasub_data.loc[kyiv_filter,'latitude'] = metasub_data.loc[kyiv_filter,'city_latitude'] # Set all the latitude to the city_latitude
    metasub_data.loc[kyiv_filter,'longitude'] = metasub_data.loc[kyiv_filter,'city_longitude'] # Set all the latitude to the city_longitutde


    # Fill missing latitude and longitude values with city-level data
    missing_lat = metasub_data["latitude"].isna()
    missing_lon = metasub_data["longitude"].isna()
    metasub_data.loc[missing_lat, "latitude"] = metasub_data.loc[missing_lat, "city_latitude"]
    metasub_data.loc[missing_lon, "longitude"] = metasub_data.loc[missing_lon, "city_longitude"]

    # Correction for incorrect London co-ordinates
    london_filter = metasub_data['city'] == 'london'
    metasub_data.loc[london_filter,'city_latitude'] = 51.50853
    metasub_data.loc[london_filter,'city_longitude'] = -0.12574

    return metasub_data



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
        prog= 'preprocess_metasub.py',
        usage="python preprocess_metasub.py -m <input metasub metadata> -t <input taxa abundance data> -o <output file <optional>>",
        description="This is a progam that can be used to process metasub data."
    )

    # Add arguements
    parser.add_argument('-m', '--metadata',dest='metadata_file',help='Enter the metasub metadata file.')
    parser.add_argument('-t','--taxa_abundance',dest='taxa_abundance_file', help='Enter the taxa abundance file.')
    parser.add_argument('-o','--output_file',dest='output_file',help='Enter the path of the output file and the name.',default=DEFAULT_OUTPUT_FILE,nargs='?')

    args = parser.parse_args()

    metasub_data = process_metasub(metadata=args.metadata_file, taxa_abundance=args.taxa_abundance_file)

    # Check if the output file path is valid
    validated_output_file_path = check_output_path(args.output_file)
    print(validated_output_file_path)

    # Save the file in the correct format
    metasub_data.to_csv(path_or_buf=validated_output_file_path,index=False)


# -m --metatdata is complete_metadata.csv
# -t --taxa_abundance is metasub_taxa_abundance.csv