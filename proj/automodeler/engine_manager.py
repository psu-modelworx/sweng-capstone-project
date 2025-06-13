from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import get_object_or_404
from django.conf import settings
from django.core.files.base import ContentFile

from .models import Dataset
from .models import PreprocessedDataSet
from engines.preprocessing_engine import PreprocessingEngine

import pandas as pd
import os

# self, df, target_column, categorical_columns=None, columns_to_remove=None, 
# test_size=DEFAULT_TEST_SIZE, random_state=DEFAULT_RANDOM_STATE
def start_preprocessing_request(request, dataset_id):
    dataset = Dataset.objects.get(id = dataset_id)
    
    #media_folder = settings.MEDIA_ROOT
    df = pd.read_csv(dataset.csv_file)
    target_column = dataset.target_feature
    all_features_dict = dataset.features
    categorical_columns = [f for f in dataset.features if all_features_dict[f] == 'C']
    ppe = PreprocessingEngine(df=df, target_column=target_column, categorical_columns=categorical_columns)
    ppe.run_preprocessing_engine()
    
    # Get the Dataframe and convert to an in-memory file
    new_df = ppe.final_df
    content = new_df.to_csv()
    temp_file = ContentFile(content.encode('UTF-8'))

    # Name the temp file
    pp_ds_name = ''.join([dataset.name, "_preprocessed", ".csv"])
    temp_file.name = pp_ds_name
    
    # First, see if there is a preprocessed dataset already
    try:
        pp_ds = PreprocessedDataSet.objects.get(original_dataset_id = dataset.id)
    except:
        print("No PP_DS related to original dataset, creating new one...")

    # If there is no previous preprocessed dataset, create a new one
    if not pp_ds:
        pp_ds = PreprocessedDataSet()
    else:
        # If there was, delete the old file
        pp_ds_filepath = pp_ds.csv_file.path
        if os.path.exists(pp_ds_filepath):
            os.remove(pp_ds_filepath)

    # Write/overwrite values
    pp_ds.name=pp_ds_name
    pp_ds.csv_file=temp_file
    pp_ds.original_dataset=dataset
    
    # Save the object
    pp_ds.save()
    return HttpResponse("Preprocessing completed...")

