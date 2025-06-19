from django.http import HttpResponse, HttpResponseNotFound, HttpResponseNotAllowed, HttpResponseBadRequest, HttpResponseServerError
from django.contrib.auth.decorators import login_required
from django.conf import settings
from django.core.files.base import ContentFile

from .models import Dataset
from .models import PreprocessedDataSet
from .models import DatasetModel

from engines.preprocessing_engine import PreprocessingEngine
from engines.modeling_engine import ModelingEngine

import pandas as pd
import os
import pickle

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
    
    # First, see if there is a preprocessed dataset already; if not, create one
    try:
        pp_ds = PreprocessedDataSet.objects.get(original_dataset_id = dataset.id)
        
        # Delete the original if it exists and create a new object
        pp_ds.delete()
        pp_ds = PreprocessedDataSet()
    except:
        print("No PP_DS related to original dataset, creating new one...")
        pp_ds = PreprocessedDataSet()

    df = pd.read_csv(dataset.csv_file)
    target_column = dataset.target_feature
    all_features_dict = dataset.features
    categorical_columns = [f for f in dataset.features if all_features_dict[f] == 'C'] # Create list of categorical columns
    ppe = PreprocessingEngine(df=df, target_column=target_column, categorical_columns=categorical_columns)
    
    # Try to run the ppe; if there is an error, return internal server error 500
    try:
        #x_train, x_test, y_train, y_test, ppe_task = ppe.run_preprocessing_engine()
        ppe.run_preprocessing_engine()
    except:
        return HttpResponseServerError("Error running preprocessing engine")

    # If there is an old file, delete it
    if pp_ds.csv_file:
        pp_ds_filepath = pp_ds.csv_file.path
        if os.path.exists(pp_ds_filepath):
            os.remove(pp_ds_filepath)

    # Get the new Dataframe and convert to an in-memory file
    new_df = ppe.final_df

    content = new_df.to_csv()
    temp_file = ContentFile(content.encode('UTF-8'))

    # Name the temp file
    pp_ds_name = ''.join([dataset.name, "_preprocessed", ".csv"])
    temp_file.name = pp_ds_name

    # Write/overwrite values
    pp_ds.name=pp_ds_name
    pp_ds.csv_file=temp_file

    pp_ds.feature_encoder = ppe.feature_encoder
    pp_ds.scaler = ppe.scaler
    pp_ds.label_encoder = ppe.label_encoder

    # Get important objects from PPE, pickle them, and create ContentFiles for storage
    pp_ds.feature_encoder = obj_to_pkl_file(ppe.feature_encoder, ''.join([pp_ds_name, '_fe_enc.bin']))
    pp_ds.scaler = obj_to_pkl_file(ppe.scaler, ''.join([pp_ds_name, '_sca.bin']))
    pp_ds.label_encoder = obj_to_pkl_file(ppe.label_encoder, ''.join([pp_ds_name, '_la_enc.bin']))

    pp_ds.original_dataset=dataset
    
    pp_ds.meta_data = ppe.to_meta_dict()

    # Save the object
    pp_ds.save()
    
    
    return HttpResponse("Preprocessing completed...")


@login_required
def start_modeling_request(request):
    print("Starting modeling")
    if request.method != 'POST':
        print("Error: Non-POST Request received!")
        return HttpResponseNotAllowed("Method not allowed")
    
    # Verify dataset exists
    dataset_id = request.POST.get('dataset_id')
    if not dataset_id:
        print("Error: Missing dataset_id in form!")
        return HttpResponseBadRequest('Missing value: dataset_id')
        
    try:
        dataset = Dataset.objects.get(id = dataset_id)
    except:
        print("Original Dataset not found in database!")
        return HttpResponseNotFound("Dataset not found")
    
    # Verify dataset has been preprocessed
    try:
        pp_ds = PreprocessedDataSet.objects.get(original_dataset_id = dataset.id)
    except:
        print("Dataset has not yet been preprocessed...")
        return HttpResponse("Dataset must be preprocessed first.", status=412)
    
    # dataset & pp_ds are now available
    # Prior to modeling, we need x_train, x_test, y_train, y_test, and task type of the preprocessed set
    # To do this, we're reconstructing the PPE
    feature_encoder = pkl_file_to_obj(pp_ds.feature_encoder.path)
    scaler = pkl_file_to_obj(pp_ds.scaler.path)
    label_encoder = pkl_file_to_obj(pp_ds.label_encoder.path)

    #ppe = PreprocessingEngine(df=df, target_column=dataset.target_feature)
    ppe = PreprocessingEngine.load_from_files(meta=pp_ds.meta_data, feature_encoder=feature_encoder, scaler=scaler, label_encoder=label_encoder)
    

    # Load in original dataset and final dataset
    df = pd.read_csv(dataset.csv_file)
    ppe.df = df
    final_df = pd.read_csv(pp_ds.csv_file)
    ppe.final_df = final_df
    ppe.target_column = dataset.target_feature

    
    task_type = ppe.task_type

    x, y = ppe.split_features_and_target()
    x_train, x_test, y_train, y_test = ppe.train_test_split_data(x, y)

    moe = ModelingEngine(X_train=x_train, X_test=x_test, y_train=y_train, y_test=y_test, task_type=task_type)
    moe.evaluate_models()
    
    moe_models = moe.models
    
    for model_method, model_obj in moe_models.items():
        
        model_name = ''.join([dataset.name, '_', str(dataset.id), '_', str(model_method)])
        model_file_name = ''.join([model_name, '.bin'])
        model_file = obj_to_pkl_file(model_obj, model_file_name)
        
        ds_model = DatasetModel(name = dataset.name, model_file=model_file, model_method=model_method, model_type=task_type, user=request.user, original_dataset=dataset)
        ds_model.save()

    return HttpResponse("Completed modeling!")

def obj_to_pkl_file(data_obj, file_name):
    data_obj_pkl = pickle.dumps(data_obj)
    data_obj_file = ContentFile(data_obj_pkl, name=file_name)
    return data_obj_file


def pkl_file_to_obj(file_name):
    if os.path.exists(file_name):
        with open(file_name, 'rb') as pkl_file:
            data_obj = pickle.load(pkl_file)
    return data_obj