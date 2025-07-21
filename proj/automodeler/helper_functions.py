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
    into a pandas dataframe to verify it can be used.  It returns the number of rows in the dataset as proof

    :param in_mem_file: InMemoryFile object to read

    :return number_of_rows: Returns the number of rows in the dataset
    """
    file_data = in_mem_file.read().decode('utf-8')
    the_file = io.StringIO(file_data)
    df = pd.read_csv(the_file)
    if not df.empty:
        # Return reading pointer to beginning of memory array
        in_mem_file.seek(0)
        number_of_rows = len(df)
        return number_of_rows

# Credit to the following:
# https://stackoverflow.com/questions/1094841/get-a-human-readable-version-of-a-file-size
def file_size_for_humans(filesize, suffix="B"):
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(filesize) < 1024.0:
            return f"{filesize:3.1f}{unit}{suffix}"
        filesize /= 1024.0
    return f"{filesize:.1f}Yi{suffix}"