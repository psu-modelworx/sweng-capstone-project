from django.http import HttpResponse, HttpResponseNotFound, HttpResponseNotAllowed, HttpResponseBadRequest
from django.contrib.auth.decorators import login_required
from django.conf import settings
from django.core.files.base import ContentFile

from .models import Dataset
from .models import PreprocessedDataSet
from .models import TrainTestDataFrame

from engines.preprocessing_engine import PreprocessingEngine

import pandas as pd
import os

@login_required
def start_preprocessing_request(request):
    print("Starting preprocessing")
    if request.method != 'POST':
        print("Error: Non-POST Request received!")
        return HttpResponseNotAllowed("Method not allowed")
    
    # Verify dataset exists
    dataset_id = request.POST.get('dataset_id')
    if not dataset_id:
        print("Error: Missing dataset_id in form!")
        return HttpResponseBadRequest('Missing value: dataset_id')
        
    print("Dataset ID: " + str(dataset_id))
    try:
        dataset = Dataset.objects.get(id = dataset_id)
    except:
        print("Original Dataset not found in database!")
        return HttpResponseNotFound("Dataset not found")
    
    df = pd.read_csv(dataset.csv_file)
    target_column = dataset.target_feature
    all_features_dict = dataset.features
    categorical_columns = [f for f in dataset.features if all_features_dict[f] == 'C'] # Create list of categorical columns
    ppe = PreprocessingEngine(df=df, target_column=target_column, categorical_columns=categorical_columns)
    x_train, x_test, y_train, y_test, ppe_task = ppe.run_preprocessing_engine()
    
    
    # Get the Dataframe and convert to an in-memory file
    new_df = ppe.final_df
    content = new_df.to_csv()
    temp_file = ContentFile(content.encode('UTF-8'))

    # Name the temp file
    pp_ds_name = ''.join([dataset.name, "_preprocessed", ".csv"])
    temp_file.name = pp_ds_name
    
    pp_ds = False
    
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
    
    create_or_update_tt_ds_obj(x_train, type='train', axis='x', pp_ds=pp_ds)
    create_or_update_tt_ds_obj(x_test, type='test', axis='x', pp_ds=pp_ds)
    create_or_update_tt_ds_obj(y_train, type='train', axis='y', pp_ds=pp_ds)
    create_or_update_tt_ds_obj(y_test, type='test', axis='y', pp_ds=pp_ds)
    
    return HttpResponse("Preprocessing completed...")


def create_or_update_tt_ds_obj(df, type, axis, pp_ds):
    tt_df_filepath = ''
    try:
        ttdf_obj = TrainTestDataFrame.objects.get(type=type, axis=axis, preprocessed_dataset = pp_ds)
        tt_df_filepath = ttdf_obj.tt_ds_file.path
    except:
        ttdf_obj = TrainTestDataFrame()
    
    content = df.to_csv()
    content_file = ContentFile(content.encode('UTF-8'))
    type_str = ''.join([axis, '_', type])
    tt_df_filename = ''.join([pp_ds.name, '_', type_str, '.csv'])
    content_file.name = tt_df_filename
    
    # Delete file if it exists

    if os.path.exists(tt_df_filepath):
        os.remove(tt_df_filepath)
    
    ttdf_obj.type = type
    ttdf_obj.axis = axis
    ttdf_obj.tt_ds_file = content_file
    ttdf_obj.preprocessed_dataset = pp_ds
    ttdf_obj.save()
