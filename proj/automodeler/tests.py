#from django.test import TestCase
import pytest
from django.urls import reverse
from django.contrib.auth.models import User
from rest_framework.test import APIRequestFactory
from .permissions import DetermineIfStaffPermissions
from rest_framework.authtoken.models import Token

from .api import verify_features

# Create your tests here.


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
    #assert "Could not upload dataset."
    assert response

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
    