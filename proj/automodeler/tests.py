#from django.test import TestCase
import pytest
from django.urls import reverse
from django.contrib.auth.models import User
from rest_framework.test import APIRequestFactory
from .permissions import DetermineIfStaffPermissions
from rest_framework.authtoken.models import Token
from unittest.mock import MagicMock, patch
from automodeler.models import Dataset
from django.contrib.auth import get_user_model

# Commented out for lack of test
#from .api import verify_features

# Create your tests here.


@pytest.fixture
def user_factory(db):
    User = get_user_model()
    def create_user(**kwargs):
        data = {
            "username": "testuser",
            "email": "testuser@example.com",
            "password": "password123",
            **kwargs,
        }
        user = User.objects.create_user(**data)
        return user
    return create_user

@pytest.mark.django_db
def test_login_form(client):
    """
    test_login_form runs the test cases for modelworx/templates/registration/login.html

    :param client: An HTTP client object provided by pytest's Django plugin

    :Test Cases: TC-006
    """
    # Create test user
    user = User.objects.create_user(username='testuser', password='testpassword')
    assert not user.is_superuser

    # Test GET on login page
    url = reverse('login')
    response = client.get(url)
    assert response.status_code == 200

    # response = client.post(url, {'username': 'testuser', 'password': 'testpassword'})
    #
    #assert response.status_code == 302  # Redirect status code
    #assert response.url == reverse('') # Need to specify an expected login page 

@pytest.mark.django_db
def test_upload_api(client):
    '''
    Set up test cases to ensure the token authentication works and valid data needs to be sent.

    :param client: A client that is provided by pytest. It is used to make get and post requests.

    :Test Cases: TC-037 & TC-038
    '''
    # Defining the URL for the api request.
    url = reverse('api_upload')

    # Making a post request to the URL and asserting that this isn't authorized.
    response = client.post(url)
    assert response.status_code == 401
    
    # Creating a user with a test username and password and making a token for them.
    user = User.objects.create_user(username='testuser', password='testpassword')
    userToken, tokenExists = Token.objects.get_or_create(user=user)

    # Making a header to authorize the user to make a post request.
    headers = { "Authorization": "Token " + userToken.key}

    # Getting the response from the post request and asserting that the user didn't send data and files.
    response = client.post(url, headers=headers)
    assert response.status_code == 400

# Commenting this one out for now.  It does not have a pytest decorator and is not called anywhere.abs
# If it needs to be used, we will need to rename it

#def test_upload_api():
#    '''
#    Tests for verify_features in api.py which takes in features, input_features, and file_target_variable and ensures they are valid.
#
#    :Test Cases: TC-43, TC-44, TC-45, TC-46
#    '''
#    # Setting up variables to test the case where the list of features are not in the features from the request.
#    features = ["Feature_Different"]
#    input_features = {"Feature1": "F1"}
#    file_target_feature = "Feature_Different"
#
#    # Checking the features and asserting text saying features from the request are invalid.
#    results = verify_features(features, input_features, file_target_feature)
#    assert results == "The list of features in the input features do not match the csv file."
#
#    # Set up variables with an invalid feature type.
#    features = ["Feature1"]
#    input_features = {"Feature1": "F1"}
#    file_target_feature = "Feature_Different"
#
#    # Testing the results and asserting text saying the feature type is invalid.
#    results = verify_features(features, input_features, file_target_feature)
#    assert results == "The feature types are not all C or N for categorical or numerical."
#
#    # Set up variables with an invalid target variable.
#    features = ["Feature1"]
#    input_features = {"Feature1": "C"}
#    file_target_feature = "Feature_Different"
#
#    # Testing to results and asseting their is text saying the target variable isn't in the list of features.
#    results = verify_features(features, input_features, file_target_feature)
#    assert results == "The target variable selected was not found in the csv file."
#
#    # Setting up variables with valid features.
#    features = ["Feature1", "Feature2", "Feature3"]
#    input_features = {"Feature1": "C", "Feature2": "N", "Feature3": "C"}
#    file_target_feature = "Feature1"
#
#    # Asserting that the features are valid and can be saved with a dataset.
#    results = verify_features(features, input_features, file_target_feature)
#    assert results == "Valid Features"

@patch('automodeler.engine_manager.start_preprocessing_task.apply_async')
@pytest.mark.django_db
def test_engine_manager_preprocessing_api(mock_apply_async, client,user_factory):
    '''
    Testing the preprocessing API to verify a user needs an authentication token and a valid dataset ID.

    :param client: The client is used when making the post request to the URL.

    :Test Cases: TC-66 & TC-67
    '''
    user = user_factory()
    dataset = Dataset.objects.create(name="MyData", user=user, features={'f1': 'C'}, target_feature='f1')

    # Set up a fake return object for apply_async with a real `.id`
    mock_result = MagicMock()
    mock_result.id = "fake-task-id-123"
    mock_apply_async.return_value = mock_result

    # Set up the URL to the preprocessing API endpoint.
    url = reverse('ppe')

    # Making a request to the URL and asserting that the user doesn't have access.
    response = client.post(url)
    assert response.status_code == 403
    
    # Making a test user and assigning them an authentication token.
    userToken, tokenExists = Token.objects.get_or_create(user=user)

    # Putting the authentication token in the header and assigning an invalid dataset ID.
    headers = { "Authorization": "Token " + userToken.key}
    #data = {"dataset_id": dataset.id}

    # Getting the response from the URL request, logic to check if it is valid has been moved to the task
    response = client.post(url, headers=headers, data={"dataset_id":dataset.id})
    assert response.status_code == 200

@patch('automodeler.engine_manager.start_modeling_task.apply_async')
@pytest.mark.django_db
def test_engine_manager_modeling_api(mock_apply_async, client, user_factory):

    '''
    Testing the modeling API to verify a user needs a valid authentication token and dataset ID.

    :param client: The client comes with the post request to the URL.

    :Test Cases: TC-68 & TC-69
    '''
    user = user_factory()
    dataset = Dataset.objects.create(name="MyData", user=user, features={'f1': 'C'}, target_feature='f1')

    # Setting up a URL to the modeling API endpoint.
    # Set up a fake return object for apply_async with a real `.id`
    mock_result = MagicMock()
    mock_result.id = "fake-task-id-123"
    mock_apply_async.return_value = mock_result

    # Setting up a URL to the modeling API endpoint.
    url = reverse('ame')

    # Getting the response from the URL request and asserting that the user doesn't have access.
    response = client.post(url)
    assert response.status_code == 403
    
    userToken, tokenExists = Token.objects.get_or_create(user=user)

    # Assigning the authentication token to the header and making an invalid dataset ID.
    headers = { "Authorization": "Token " + userToken.key}

    # Asserting that an request from the client will be a 404 status code because the dataset ID is invalid.
    response = client.post(url, headers=headers, data={"dataset_id":dataset.id})
    assert response.status_code == 200


@pytest.mark.django_db
def test_request_datasets(client):
    '''
    Testing the request datasets API request to ensure a valid user is needed and the response.

    :param client: The client is used by pytest to make post requests.

    :Test Cases: TC-64 & TC-65
    '''
    # Setting the URL to API request for datasets.
    url = reverse('api_request_datasets')

    # Asserting that a user needs to be authenticated to make an API request.
    response = client.post(url)
    assert response.status_code == 401
    
    # Setting up a user and assigning them an authentication token.
    user = User.objects.create_user(username='testuser', password='testpassword')
    userToken, tokenExists = Token.objects.get_or_create(user=user)

    # Putting the token in a header so it will authenticate the user.
    headers = { "Authorization": "Token " + userToken.key}

    # Asserting that this is a valid request but the response wouldn't have any data.
    response = client.post(url, headers=headers)
    assert response.status_code == 200

@pytest.mark.django_db
def test_request_model(client):
    '''
    Testing an API request to get a user's models.
    There is a test case with and without a user's authentication token.

    parm client: The client is used to run the post request to the API URL endpoint.

    :Test Cases: TC-70 & TC-71
    '''
    # Set up the API enpoint to the request models URL and making a request without an authentication token.
    url = reverse('api_request_models')
    response = client.post(url)

    # Asserting that the user isn't given access becuase they aren't authenticated.
    assert response.status_code == 401
    
    # Set up a testr user with a test username and password. Assigning the user a token to authenticate them.
    user = User.objects.create_user(username='testuser', password='testpassword')
    userToken, tokenExists = Token.objects.get_or_create(user=user)

    # Assigning the header to the request and getting its response.
    headers = { "Authorization": "Token " + userToken.key}
    response = client.post(url, headers=headers)

    # Asserting that the user was given access and it would have all their models.
    assert response.status_code == 200

@pytest.mark.django_db
def test_run_model(client):
    '''
    This tests the run model API endpoint. There is a test to verify a user needs to be authenticated.
    There is another test ensuring an authenticated user has access but they need to send a model ID.

    parm client: The client is used to make API requests and sends back responses with status codes.

    :Test Cases: TC-106 & TC-107
    '''
    # Set the URL test the run model API endpoint and getting the response from a request.
    url = reverse('run_model')
    response = client.post(url)

    # Asserting that the response is a 403 status code because the user isn't authenticated.
    assert response.status_code == 403
    
    # Creating a test user and assigning them an authentication token.
    user = User.objects.create_user(username='testuser', password='testpassword')
    userToken, tokenExists = Token.objects.get_or_create(user=user)

    # Adding the authentication token to the header and getting the response from an API request.
    headers = { "Authorization": "Token " + userToken.key}
    response = client.post(url, headers=headers)

    # Asserting that there is a 400 status code response because the user did not send a model ID in their request.
    assert response.status_code == 400

@pytest.mark.django_db
def test_api_check_task_result(client):
    '''
    This is used to test the API enpoint which gets a tasks status. There is a test to verify a user needs to be authenticated.
    There is also a test to for an authenticated user to verify they have access.

    parm client: The client is used to make the API request and get responses.

    :Test Cases: TC-108 & TC-109
    '''
    # Set up a URL to the check task results API endpoint. Put a required argument for the task ID and getting the response from the request.
    url = reverse('check_task_result', args=["task_id"])
    response = client.post(url)

    # Asserting that there is a 403 status code response because the user is not authenticated.
    assert response.status_code == 403
    
    # Defining a test user and assigning them an authentication token.
    user = User.objects.create_user(username='testuser', password='testpassword')
    userToken, tokenExists = Token.objects.get_or_create(user=user)

    # Putting the authentication token in the header to authenticate the user.
    headers = { "Authorization": "Token " + userToken.key}
    response = client.post(url, headers=headers)

    # Asserting that this was a valid request, even though a valid task ID wasn't used.
    assert response.status_code == 200

@pytest.mark.django_db
def test_api_dataset_report_download(client):
    '''
    The test is for an API request to download a dataset report.
    There are tests to ensure a user needs to be authenticated and to verify the case when a dataset doesn't exist.

    parm client: The client is used to make an API request and send back a response with the status code.

    :Test Cases: TC-110 & TC-111
    '''
    # Setting up the URL to the API endpoint and including the dataset ID as an argument in the URL.
    url = reverse('report_download', args=[3])

    # Making an API request with the URL and asserting the response is 403 because the user isn't authenticated.
    response = client.post(url) 
    assert response.status_code == 403
    
    # Defining a test user and assigning them and authentication token.
    user = User.objects.create_user(username='testuser', password='testpassword')
    userToken, tokenExists = Token.objects.get_or_create(user=user)

    # Adding the authentication token to the header and makking an API request to download a dataset.
    headers = { "Authorization": "Token " + userToken.key}
    response = client.post(url, headers=headers)

    # Asserting that the status code response is 404 because the dataset ID wasn't found for the user.
    assert response.status_code == 404

@pytest.mark.django_db
def test_api_request_delete_dataset(client):
    '''
    There is a test to ensure the user must be authenticated to make the API request.
    A second test is used to ensure a dataset can be deleted successfully.

    parm client: Used to make an API request and get a response with the status code.

    :Test Cases: TC-118 & TC-119
    '''
    # Setting up the URL to the delete dataset URL enpoint with the dataset ID.
    url = reverse('dataset_delete', args=[4])

    # Getting the response after the API request and asserting that the user doesn't have access.
    response = client.post(url) 
    assert response.status_code == 403
    
    # Defining a test user and the authentication token to authenticate the user.
    user = User.objects.create_user(username='testuser', password='testpassword')
    userToken, tokenExists = Token.objects.get_or_create(user=user)

    # Putting the authentication token in the header and makeing a request with it.
    headers = { "Authorization": "Token " + userToken.key}
    response = client.post(url, headers=headers)

    # Asserting that the response has a status code of 302 because the user would be redirceted to the dataset collection.
    assert response.status_code == 302

@pytest.mark.django_db
def test_api_request_delete_model(client):
    '''
    Has a test to ensure a user must be authenticated.
    Theres another test to verify a model can be deleted.

    parm client: Used to make the API request and get a response.

    :Test Cases: TC-120 & TC-121
    '''
    # The URL for the API endpoint to delete a model.
    url = reverse('model_delete')

    # Making the API request and asserting that the user wasn't authenticated.
    response = client.post(url) 
    assert response.status_code == 403
    
    # Making a test user and assigning them an authentication token.
    user = User.objects.create_user(username='testuser', password='testpassword')
    userToken, tokenExists = Token.objects.get_or_create(user=user)

    # Created a model ID and assigned it to the data object.
    model_id = 5
    data = {"model_id": model_id}

    # Putting the authentication token in the header and making an API request to delete a model.
    headers = { "Authorization": "Token " + userToken.key}
    response = client.post(url, headers=headers, data=data)

    # Asserting that there was a 404 status code response because the model wasn't found.
    assert response.status_code == 404;

@pytest.mark.django_db
def test_account_page(client):
    # Defining the url that will be navigated to.
    url = reverse('account')

    # Getting the response when the client navigates to the URL.
    response = client.get(url)
    
    # The response should redirect the client to the login page.
    assert response.status_code == 302

    # Setting up a user and logging them into the client so they are authenticated.
    user = User.objects.create_user(username='testuser', password='testpassword')
    assert user
    client.login(username='testuser', password='testpassword')
    
    # Getting the response when the client navigates to the URL.
    response = client.get(url)

    # The response should open the account page successfully.
    assert response.status_code == 200

@pytest.mark.django_db
def test_user_staff_permissions_request():
    # Creating a user and setting their staff value to be false.
    user = User.objects.create_user(username='testuser', password='testpassword')
    user.is_staff = False

    # Defining a request to the modelworx path and assigning a user to it.
    request = APIRequestFactory().get('modelworx')
    request.user = user

    # Using the permissions.py class to check if permissions are met.
    determinePermissions = DetermineIfStaffPermissions()
    permissions = determinePermissions.has_permission(request, None)

    # The assertion is that permissions weren't given.
    assert not permissions

@pytest.mark.django_db
def test_user_no_staff_permissions_request():
    # Defining the user and making them a staff member.
    user = User.objects.create_user(username='testuser', password='testpassword')
    user.is_staff = True

    # Creating the request and assigning the user to it.
    request = APIRequestFactory().get('modelworx')
    request.user = user

    # Using the permissions.py class to check if the user has permissions to access the page.
    determinePermissions = DetermineIfStaffPermissions()
    permissions = determinePermissions.has_permission(request, None)

    # Asserting that the user does have permissions to view the page.
    assert permissions
    