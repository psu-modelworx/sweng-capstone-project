from django.core.exceptions import ObjectDoesNotExist
from rest_framework import status
from rest_framework.authentication import TokenAuthentication
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .models import Dataset, PreprocessedDataSet, TunedDatasetModel
from .forms import DatasetForm
from . import helper_functions

import json

@api_view(["POST"])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
def api_upload(request):
    '''
    Making an API request to upload a dataset with a csv file and name.
    Verifying the file name and csv file in this function before anything is uploaded.
    Letting the user know if a uploading was successful or not.

    :param request: An API request using token authentication. It contains the csv file and name for a dataset.
    '''
    # Getting the form used in the API request.
    form = DatasetForm(request.POST, request.FILES)

    # Ensuring a post request is used and there is at least 1 file in the request.
    if form.is_valid():
        # Getting the file name, csv file, input features, target variable, and user ID from the request.
        try:
            file_name = request.POST.get('name')
            input_features = json.loads(request.POST.get("input_features"))
            file_target_variable = request.POST.get("target_variable")
            csv_file = request.FILES['csv_file']
            user_id = request.user.id
        except Exception as e:
            print("Exception: {0}".format(e))
            return Response(data="Could not retrieve the data in the request.", status=status.HTTP_400_BAD_REQUEST)

        # If the dataset is empty let the user know it cannot be uploaded.
        if csv_file.size > 0:
            # If the name of the dataset is too long, let the user know it cannot be used.
            if len(file_name) <= 50:
                try:
                    # Trying to sanitize the csv file by removing html data.
                    helper_functions.sanitize_dataset(csv_file)
                except Exception as e:
                    print('Exception: ' + e)
                    # Letting the user know they cannot use the CSV file because it failed sanitation.
                    return Response(data="CSV file failed sanitation.", status=status.HTTP_400_BAD_REQUEST)

                # Using a function to get the features from the dataset.
                features = helper_functions.extract_features_from_inMemoryUploadedFile(csv_file)

                # Ensuring the features used in the request are valid for the dataset.
                results = verify_features(features, input_features, file_target_variable)

                # Checking the results of verifying the features.
                if (results != "Valid Features"):
                    return Response(data=results, status=status.HTTP_400_BAD_REQUEST)

                # Making a datset object and saving it to the user.
                dataset_model = Dataset.objects.create(name=file_name, features=input_features, target_feature=file_target_variable, csv_file=csv_file, user_id=user_id)
                dataset_model.save()
            else:
                # Telling the user their file name is too long.
                return Response(data="The name has too many characters in it.", status=status.HTTP_400_BAD_REQUEST)
        else:
            # Telling the user their dataset file is empty.
            return Response(data="The csv file cannot be empty.", status=status.HTTP_400_BAD_REQUEST)

        # Informing the user that their uploaded was successful.
        return Response(data="Successfully uploaded dataset.", status=status.HTTP_200_OK)
    else:
        # Informing the user that their uploaded cannot be done.
        return Response(data="Could not upload dataset.", status=status.HTTP_400_BAD_REQUEST)

def verify_features(features, input_features, file_target_variable):
    # Looping through the list of features to compare with the input features in the request.
    for feature in features:
        # Checking the input features with the csv file.
        if feature not in input_features:
            return "The list of features in the input features do not match the csv file."
                    
        # Checking the feature types to ensure they are C or N.
        if input_features[feature] != "C" and input_features[feature] != "N":
            return "The feature types are not all C or N for categorical or numerical."
             
    # Ensuring the target variable is in the list of features.
    if file_target_variable not in features:
        return "The target variable selected was not found in the csv file."

    return "Valid Features"

@api_view(["POST"])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
def api_request_datasets(request):
    '''
    This function gets all the datasets for a users and checks if they have been preprocessed.
    Then make a response with all datasets in a list so they can be printed in an easy to read format.   

    :param request: The API request that contains the URL and authentication token for a user.
    '''
    # Collecting all datasets for the user making the request and setting up the dataset information list for the response.
    user_datasets = Dataset.objects.filter(user = request.user)
    dataset_information = ["Dataset Name, " + "Dataset File Name, " + "Dataset ID, " + "Preprocessed Dataset File Name"]
    
    # Going through the datasets and checking if they are preprocessed.
    for dataset in user_datasets:
        # Resetting the preprocessed string with each iteration.
        # pp_dataset = "" # Ruff mentioned to remove this?  If it breaks something, here's why

        try:
            # Trying to get the preprocessed file and appending it to the dataset information.
            pp_dataset_filename = PreprocessedDataSet.objects.get(original_dataset_id = dataset.id).filename()
            dataset_information.append(dataset.name + ", " + dataset.filename() + ", " + str(dataset.id) + ", " + pp_dataset_filename)
        except ObjectDoesNotExist:
            # If no preprocessed file exits, append that information to the dataset information list.
            dataset_information.append(dataset.name + ", " + dataset.filename() + ", " + str(dataset.id) + ", " + "No Preprocessed Dataset")

    # Returning a response with the dataset information which can be printed out by the user.
    return Response(data=dataset_information, status=status.HTTP_200_OK)

@api_view(["POST"])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
def api_request_models(request):
    '''
    The function gets the models from the user and puts them in a list for a response.

    :param request: The request parameter is used to get the user's models.
    '''
    # Getting the user's models.
    user_models = TunedDatasetModel.objects.filter(user=request.user)
    
    # Setting up a list to format the response. It contains the model ID which will be helpful in running the model.
    model_information = ["Model Name, " + "Model File Name, " + "Model Method, " + "Model Type" + "Model ID"]

    # Lopping through the models and adding their data to the model_information list.
    for model in user_models:
        model_information.append(model.name + ", " + str(model.model_file) + ", " + model.model_method + ", " + model.model_type + "," + str(model.id))

    # Returning a response with the model information and letting the user know that everything was successful.
    return Response(data=model_information, status=status.HTTP_200_OK)