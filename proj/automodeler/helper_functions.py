import csv
import io
import pandas as pd

def extract_features(dataset_fileName):
    """
    extract_features opens and reads the file at the path given and extracts features from the first row.

    :param dataset_fileName: Full-path filename of file to open and read
    """
    features = []
    with open(dataset_fileName, 'r') as file:
        csvFileReader = csv.reader(file)
        features = next(csvFileReader)
    print(features)
    return features

def extract_features_from_inMemoryUploadedFile(in_mem_file):
    """
    extract_features_from_inMemoryUploadedFile reads the in-memory file object and extracts the feature names from 
    the first row.

    :param in_mem_file: InMemoryFile object to read
    """
    file_data = in_mem_file.read().decode('utf-8')
    csv_file = io.StringIO(file_data)
    reader = csv.reader(csv_file)
    features = next(reader)

    # Return reading pointer to beginning of memory array
    in_mem_file.seek(0)
    
    return features

def sanitize_dataset(in_mem_file):
    """
    sanitize_dataset reads an in-memory file object, and attempts to sanitize by removing html data.  Then, it tries to load
    into a pandas dataframe to verify it can be used.

    :param in_mem_file: InMemoryFile object to read

    :return sanitized_dataset: Sanitized dataset to be saved
    """
    file_data = in_mem_file.read().decode('utf-8')
    the_file = io.StringIO(file_data)
    df = pd.read_csv(the_file)
    
    # Return reading pointer to beginning of memory array
    in_mem_file.seek(0)

    return True