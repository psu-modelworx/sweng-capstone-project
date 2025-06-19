from rest_framework.authentication import TokenAuthentication
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .models import Dataset
from .forms import DatasetForm
from . import helper_functions

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
        # Getting the file name, csv file, and user ID from the request.
        file_name = request.POST.get('name')
        csv_file = request.FILES['csv_file']
        user_id = request.user.id

        # If the dataset is empty let the user know it cannot be uploaded.
        if csv_file.size > 0:
            # If the name of the dataset is too long, let the user know it cannot be used.
            if len(file_name) <= 50:
                try:
                    # Trying to sanitize the csv file by removing html data.
                    helper_functions.sanitize_dataset(csv_file)
                except Exception as e:
                    # Letting the user know they cannot use the CSV file because it failed sanitation.
                    return Response("CSV file failed sanitation for " + request.user.username + ".")
                
                # Using a function to get the features from the dataset.
                features = helper_functions.extract_features_from_inMemoryUploadedFile(csv_file)

                # Making a datset object and saving it to the user.
                dataset_model = Dataset.objects.create(name=file_name, features=features, csv_file=csv_file, user_id=user_id)
                dataset_model.save()
            else:
                # Telling the user their file name is too long.
                return Response("The name has too many characters in it. Could not upload dataset for " + request.user.username + ".")
        else:
            # Telling the user their dataset file is empty.
            return Response("The csv file cannot be empty. Could not upload dataset for " + request.user.username + ".")

        # Informing the user that their uploaded was successful.
        return Response("Successfully uploaded dataset for " + request.user.username + ".")
    else:
        # Informing the user that their uploaded cannot be done.
        return Response("Could not upload dataset for " + request.user.username + ".")