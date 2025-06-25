from rest_framework.authentication import TokenAuthentication
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .models import Dataset
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
            return Response("Could not retrieve the data in the request.")

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
                    return Response("CSV file failed sanitation.")

                # Using a function to get the features from the dataset.
                features = helper_functions.extract_features_from_inMemoryUploadedFile(csv_file)

                # Ensuring the features used in the request are valid for the dataset.
                results = verify_features(features, input_features, file_target_variable)

                # Checking the results of verifying the features.
                if (results != "Valid Features"):
                    return Response(results)

                # Making a datset object and saving it to the user.
                dataset_model = Dataset.objects.create(name=file_name, features=input_features, target_feature=file_target_variable, csv_file=csv_file, user_id=user_id)
                dataset_model.save()
            else:
                # Telling the user their file name is too long.
                return Response("The name has too many characters in it.")
        else:
            # Telling the user their dataset file is empty.
            return Response("The csv file cannot be empty.")

        # Informing the user that their uploaded was successful.
        return Response("Successfully uploaded dataset.")
    else:
        # Informing the user that their uploaded cannot be done.
        return Response("Could not upload dataset.")

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