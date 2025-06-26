from django.http import HttpResponse, HttpResponseNotFound, HttpResponseNotAllowed, HttpResponseBadRequest, HttpResponseServerError
from django.contrib.auth.decorators import login_required
from django.core.files.base import ContentFile
from django.core.exceptions import ObjectDoesNotExist

from .models import Dataset
from .models import PreprocessedDataSet
from .models import DatasetModel
from .models import TunedDatasetModel

from engines.preprocessing_engine import PreprocessingEngine
from engines.modeling_engine import ModelingEngine

import pandas as pd
import os
import pickle
import json


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
    except ObjectDoesNotExist:
        print("Original Dataset not found in database!")
        return HttpResponseNotFound("Dataset not found")
    
    # First, see if there is a preprocessed dataset already; if not, create one
    try:
        pp_ds = PreprocessedDataSet.objects.get(original_dataset_id = dataset.id)
        
        # Delete the original if it exists and create a new object
        pp_ds.delete()
        pp_ds = PreprocessedDataSet()
    except ObjectDoesNotExist:
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
    except Exception as e:
        msg = "Error running preprocessing engine {0}".format(e)
        print(msg)
        return HttpResponseServerError("Error running preprocessing engine {0}".format(e))

    # If there is an old file, delete it
    if pp_ds.csv_file:
        pp_ds_filepath = pp_ds.csv_file.path
        if os.path.exists(pp_ds_filepath):
            os.remove(pp_ds_filepath)

    # Get the new Dataframe and convert to an in-memory file
    new_df = ppe.final_df

    content = new_df.to_csv(index=False)
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
    except ObjectDoesNotExist:
        print("Original Dataset not found in database!")
        return HttpResponseNotFound("Dataset not found")
    
    # Verify dataset has been preprocessed
    try:
        pp_ds = PreprocessedDataSet.objects.get(original_dataset_id = dataset.id)
    except ObjectDoesNotExist:
        print("Dataset has not yet been preprocessed...")
        return HttpResponse("Dataset must be preprocessed first.", status=412)
    
    # dataset & pp_ds are now available
    # Prior to modeling, we need x_train, x_test, y_train, y_test, and task type of the preprocessed set
    
    ppe = reconstruct_ppe(pp_ds)
    
    task_type = ppe.task_type

    x_train, x_test, y_train, y_test = ppe.split_data()

    moe = ModelingEngine(X_train=x_train, X_test=x_test, y_train=y_train, y_test=y_test, task_type=task_type)
    moe.run_modeling_engine()
    
    moe_results = moe.results
    untuned_models = moe_results['untuned']
    tuned_models = moe_results['tuned']

    for model_method, model_results in untuned_models.items():
        model_name = ''.join([dataset.name, '_', str(dataset.id), '_', str(model_method), '_untuned'])
        model_file_name = ''.join([model_name, '.bin'])
        model_file = obj_to_pkl_file(model_results['model'], model_file_name)
        ds_model = DatasetModel(name = model_name, model_file=model_file, model_method=model_method, model_type=task_type, user=request.user, original_dataset=dataset)
        ds_model.save()

        tuned_model_name = ''.join([dataset.name, '_', str(dataset.id), '_', str(model_method), '_tuned'])
        tuned_model_file_name = ''.join([tuned_model_name, '.bin'])
        tuned_model_file = obj_to_pkl_file(tuned_models[model_method]['optimized_model'], tuned_model_file_name)
        tuned_ds_model = TunedDatasetModel(name = tuned_model_name, model_file=tuned_model_file, model_method=model_method, model_type=task_type, untuned_model=ds_model, user=request.user, original_dataset=dataset)
        tuned_ds_model.save()
        

    return HttpResponse("Completed modeling!")

@login_required
def run_model(request):
    print("Starting model run")
    if request.method != 'POST':
        print("Error: Non-POST Request received!")
        return HttpResponseNotAllowed("Method not allowed")
    
    # Verify model exists
    model_id = request.POST.get('model_id')
    if not model_id:
        print("Error: Missing model_id in form!")
        return HttpResponseBadRequest('Missing value: model_id')

    # Get the model and verify it exists
    #try:
    #    ds_model = DatasetModel.objects.get(id=model_id)
    #except ObjectDoesNotExist:
    #    msg = "Error finding model with model ID: {0}".format(model_id)
    #    print(msg)
    #    return HttpResponseNotFound(msg)
    
    # Get the tuned model and verify it exists
    try:
        tuned_model = TunedDatasetModel.objects.get(id=model_id)
    except Exception as e:
        print("Exception: {0}".format(e))


    # Get preprocessed Dataset to recreate preprocessing engine
    try:
        dataset = Dataset.objects.get(id=tuned_model.original_dataset_id)
        pp_ds = PreprocessedDataSet.objects.get(original_dataset=dataset)
    except ObjectDoesNotExist:
        msg = "Error retrieving preprocessed dataset from model."
        print(msg)
        return HttpResponseNotFound(msg)

    # Get the data from the post request
    data = request.POST.get('data')
    if not data:
        msg = 'Error:  missing data field'
        print(msg)
        return HttpResponseBadRequest(msg)
    data = json.loads(data)
    try:
        data_values = data['values']
    except KeyError:
        msg = "Missing values field"
        print(msg)
        return HttpResponseBadRequest(msg)

    # data_values should be a list of dictionaries with 'key' 'value' key names
    #for value

    ds_features = list(dataset.features.keys())

    # Verify that the number of values sent is equal to the number of features
    if len(ds_features) != len(data_values):
        msg = "Invalid number of input features"
        print(msg)
        return HttpResponseBadRequest(msg)
    
    ppe = reconstruct_ppe(pp_ds)

    tuned_model_obj = pkl_file_to_obj(tuned_model.model_file)

    #x_train, x_test, y_train, y_test = ppe.split_data()
    #ds_model_obj.fit(x_train, y_train)

    df = pd.DataFrame([data_values], columns=ds_features)
    p_df = ppe.transform_single_row(df)
    results = tuned_model_obj.predict(p_df)
    
    if tuned_model.model_type == "classification":
        results = ppe.decode_target(results)

    # Still need to do stuff to convert categorical from integer to category name!
    print(results[0])
    return HttpResponse("Predicted results: {0}".format(results[0]), content_type="text/plain")

def reconstruct_ppe(pp_ds):
    test_df = pd.read_csv(pp_ds.csv_file)
    ppe = PreprocessingEngine.load_from_files(
        meta=pp_ds.meta_data,
        clean_df=test_df,
        feature_encoder=pkl_file_to_obj(pp_ds.feature_encoder), 
        scaler=pkl_file_to_obj(pp_ds.scaler), 
        label_encoder=pkl_file_to_obj(pp_ds.label_encoder)
    )

    return ppe
    

def obj_to_pkl_file(data_obj, file_name):
    data_obj_pkl = pickle.dumps(data_obj)
    data_obj_file = ContentFile(data_obj_pkl, name=file_name)
    return data_obj_file


def pkl_file_to_obj(file_obj):
    data_obj = pickle.load(file_obj)
    return data_obj