from celery import shared_task
from celery.utils.log import get_task_logger

from django.core.files.base import ContentFile
from django.core.exceptions import ObjectDoesNotExist
from django.contrib.auth import get_user_model

from .models import Dataset
from .models import PreprocessedDataSet
from .models import DatasetModel
from .models import UserTask

from engines.preprocessing_engine import PreprocessingEngine
from engines.modeling_engine import ModelingEngine
from engines.reporting_engine import ReportingEngine

import pandas as pd
import os
import pickle

import logging
logger = get_task_logger(__name__)

@shared_task(bind=True)
def start_preprocessing_task(self, dataset_id, user_id):
    print("in task preprocessing")

    ## Update Tasks
    #task_record = UserTask.objects.get(task_id=self.request.id)
    task_record = UserTask.objects.filter(task_id=self.request.id).first()
    if task_record:
        task_record.status = "STARTED"
        task_record.save()

    try:
        print("about to get dataset")
        dataset = Dataset.objects.get(id=dataset_id, user_id=user_id)
    except ObjectDoesNotExist:
        print("Original Dataset not found in database!")
        if task_record:
            task_record.status = "FAILURE"
            task_record.result_message = "Original Dataset not found in database!"
            task_record.save()
        return {"message": "Dataset not found", "status": 404}

    # Check that target feature has been selected
    try:
        target_feature = dataset.target_feature
        if target_feature is None or not target_feature:
            raise Exception("Target_feature is None or empty")
        if len(target_feature) == 0:
            raise Exception("Target_feature is empty")
    except Exception as e:
        print('Exception {0}'.format(e))
        if task_record:
            task_record.status = "FAILURE"
            task_record.result_message = "Target feature not selected"
            task_record.save()
        return {"message": "Target feature not selected", "status": 500}

    # Check that all features have been labeled
    try:
        target_feature = dataset.target_feature
        if target_feature is None or not target_feature:
            raise Exception("Target_feature is None or empty")
        if len(target_feature) == 0:
            raise Exception("Target_feature is empty")
    except Exception as e:
        print('Exception {0}'.format(e))
        if task_record:
            task_record.status = "FAILURE"
            task_record.result_message = "Target feature not selected"
            task_record.save()
        return {"message": "Target feature not selected", "status": 500}

    except Exception as e:
        print('Exception {0}'.format(e))
    # First, see if there is a preprocessed dataset already; if not, create one
    try:
        pp_ds = PreprocessedDataSet.objects.get(original_dataset_id=dataset.id)
        
        # Delete the original if it exists and create a new object
        pp_ds.delete()
        pp_ds = PreprocessedDataSet()
    except ObjectDoesNotExist:
        print("No PP_DS related to original dataset, creating new one...")
        pp_ds = PreprocessedDataSet()

    df = pd.read_csv(dataset.csv_file)
    target_column = dataset.target_feature
    all_features_dict = dataset.features

    try:
        categorical_columns = [f for f in dataset.features if all_features_dict[f] == 'C']
        ppe = PreprocessingEngine(df=df, target_column=target_column, categorical_columns=categorical_columns)
    except Exception as e:
        msg = "Potential TypeError: {0}".format(e)
        print(msg)
        if task_record:
            task_record.status = "FAILURE"
            task_record.result_message = msg
            task_record.save()
        return {"message": "Error running preprocessing engine. Did you select Target Variable?", "status": 500}

     # Try to run the ppe; if there is an error, return internal server error 500
    try:
        #x_train, x_test, y_train, y_test, ppe_task = ppe.run_preprocessing_engine()
        ppe.run_preprocessing_engine()
    except Exception as e:
        msg = "Error running preprocessing engine {0}".format(e)
        print(msg)
        if task_record:
            task_record.status = "FAILURE"
            task_record.result_message = msg
            task_record.save()
        return {"message": "Error running preprocessing engine {0}".format(e), "status": 500}

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
    pp_ds.name = pp_ds_name
    pp_ds.csv_file = temp_file

    pp_ds.file_size = temp_file.size
    pp_ds.number_of_removed_rows = len(ppe.dropped_columns)
    pp_ds.number_of_rows = dataset.number_of_rows - pp_ds.number_of_removed_rows
    pp_ds.removed_features = ppe.dropped_columns
    pp_ds.model_type = ppe.task_type
    

    # Get important objects from PPE, pickle them, and create ContentFiles for storage
    pp_ds.feature_encoder = obj_to_pkl_file(ppe.feature_encoder, f"{pp_ds_name}_fe_enc.bin")
    pp_ds.scaler = obj_to_pkl_file(ppe.scaler, f"{pp_ds_name}_sca.bin")
    pp_ds.label_encoder = obj_to_pkl_file(ppe.label_encoder, f"{pp_ds_name}_la_enc.bin")
    
    pp_ds.original_dataset = dataset
    pp_ds.meta_data = ppe.to_meta_dict()
    
    # Available models
    

    # Save the object
    pp_ds.save()
    msg = "Preprocessing completed..."
    if task_record:
        task_record.status = "SUCCESS"
        task_record.result_message = msg
        task_record.save()

    return {"filename": pp_ds_name, "status": 200}


@shared_task(bind=True)
def start_modeling_task(self, dataset_id, user_id):
    print("Starting modeling")

    # Get User object from user_id
    User = get_user_model()
    user = User.objects.get(id=user_id)

    ## Update Tasks
    #task_record = UserTask.objects.get(task_id=self.request.id)
    task_record = UserTask.objects.filter(task_id=self.request.id).first()
    if task_record:
        task_record.status = "STARTED"
        task_record.save()

    try:
        dataset = Dataset.objects.get(id=dataset_id, user_id=user_id)
    except ObjectDoesNotExist:
        print("Original Dataset not found in database!")
        if task_record:
            task_record.status = "FAILURE"
            task_record.result_message = "Original Dataset not found in database!"
            task_record.save()
        return {"message": "Dataset not found", "status": 404}
    
    # Verify dataset has been preprocessed
    try:
        pp_ds = PreprocessedDataSet.objects.get(original_dataset_id=dataset.id)
    except ObjectDoesNotExist:
        print("Dataset has not yet been preprocessed...")
        if task_record:
            task_record.status = "FAILURE"
            task_record.result_message = "Dataset has not yet been preprocessed..."
            task_record.save()
        return {"message": "Dataset must be preprocessed first.", "status": 412}

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
        ds_model = DatasetModel(
            name = model_name, 
            model_file=model_file, 
            model_method=model_method, 
            model_type=task_type, 
            user=user, 
            tuned=False,
            original_dataset=dataset)
        ds_model.save()

        tuned_model_name = ''.join([dataset.name, '_', str(dataset.id), '_', str(model_method), '_tuned'])
        tuned_model_file_name = ''.join([tuned_model_name, '.bin'])
        tuned_model_file = obj_to_pkl_file(tuned_models[model_method]['optimized_model'], tuned_model_file_name)
        tuned_ds_model = DatasetModel(
            name = tuned_model_name, 
            model_file=tuned_model_file, 
            model_method=model_method, 
            model_type=task_type, 
            user=user, 
            tuned=True,
            original_dataset=dataset)
        tuned_ds_model.save()
    
    if task_record:
        task_record.status = "SUCCESS"
        task_record.result_message = "Modeling completed!"
        task_record.save()

    return {"message": "Modeling completed!", "status": 200}

@shared_task(bind=True)
def run_model_task(self, model_id, user_id, data_dict):
    ## Update Tasks
    #task_record = UserTask.objects.get(task_id=self.request.id)
    task_record = UserTask.objects.filter(task_id=self.request.id).first()
    if task_record:
        task_record.status = "STARTED"
        task_record.save()

    # Get the model and verify it exists
    #try:
    #    ds_model = DatasetModel.objects.get(id=model_id)
    #except ObjectDoesNotExist:
    #    msg = "Error finding model with model ID: {0}".format(model_id)
    #    print(msg)
    #    return HttpResponseNotFound(msg)
    
    # Get the tuned model and verify it exists
    try:
        tuned_model = DatasetModel.objects.get(id=model_id, user_id=user_id, tuned=True)
    except DatasetModel.DoesNotExist as e:
        # return {"message": "Tuned model not found.", "status": 404}
        print(f"TunedDatasetModel not found: {e}")
        if task_record:
            task_record.status = "FAILURE"
            task_record.result_message = "Tuned model not found."
            task_record.save()
        return {"message": "Tuned model not found.", "status": 404}

    # Get preprocessed Dataset to recreate preprocessing engine
    try:
        dataset = Dataset.objects.get(id=tuned_model.original_dataset_id)
        pp_ds = PreprocessedDataSet.objects.get(original_dataset=dataset)
    except ObjectDoesNotExist:
        msg = "Error retrieving preprocessed dataset from model."
        print(msg)
        if task_record:
            task_record.status = "FAILURE"
            task_record.result_message = msg
            task_record.save()
        return {"message": msg, "status": 404}

    # Validate data_dict contains values
    try:
        data_values = data_dict['values']
    except KeyError:
        msg = "Missing values field"
        print(msg)
        if task_record:
            task_record.status = "FAILURE"
            task_record.result_message = msg
            task_record.save()
        return {"message": msg, "status": 400}

    # data_values should be a list of dictionaries with 'key' 'value' key names
    #for value

    ds_features = list(dataset.features.keys())
    
    # Verify that the number of values sent is equal to the number of features
    if len(ds_features) != len(data_values):
        msg = "Invalid number of input features"
        print(msg)
        if task_record:
            task_record.status = "FAILURE"
            task_record.result_message = msg
            task_record.save()
        return {"message": msg, "status": 400}
    
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
    # print(results[0])

    if task_record:
        task_record.status = "SUCCESS"
        task_record.result_message = "Predicted Result!"
        task_record.save()

    return {"message": f"Predicted result: {results[0]}", "status": 200}

@shared_task(bind=True)
def start_report_generation_task(self, dataset_id, pp_dataset_id, user_id):
    print("in task preprocessing")
    ## Update Tasks
    #task_record = UserTask.objects.get(task_id=self.request.id)
    task_record = UserTask.objects.filter(task_id=self.request.id).first()
    if task_record:
        task_record.status = "STARTED"
        task_record.save()
    
    try:
        dataset = Dataset.objects.get(id=dataset_id, user=user_id)
    except ObjectDoesNotExist:
        msg = "Error retrieving dataset for user."
        print(msg)
        if task_record:
            task_record.status = "FAILURE"
            task_record.result_message = msg
            task_record.save()
        return {"message": msg, "status": 404}
    
    try:
        pp_dataset = PreprocessedDataSet.objects.get(id=pp_dataset_id, dataset=dataset, user=user_id)
    except ObjectDoesNotExist:
        msg = "Error retrieving preprocessed dataset for user."
        print(msg)
        if task_record:
            task_record.status = "FAILURE"
            task_record.result_message = msg
            task_record.save()
        return {"message": msg, "status": 404}
        
    # reconstruct ppe
    ppe = reconstruct_ppe(pp_dataset)





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
