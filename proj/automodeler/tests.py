#from django.test import TestCase
import pytest
from django.urls import reverse
from django.contrib.auth.models import User
from rest_framework.test import APIRequestFactory
from .permissions import DetermineIfStaffPermissions


# Create your tests here.
def test_func():
    assert 0 == 0

@pytest.mark.django_db
def test_login_form(client):
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
def test_user_staff_permissions_request():
    # Creating a user and setting their staff value to be false.
    user = User.objects.create_user(username='testuser', password='testpassword')
    user.is_staff = False;

    # Defining a request to the modelworx path and assigning a user to it.
    request = APIRequestFactory().get('modelworx')
    request.user = user

    # Using the permissions.py class to check if permissions are met.
    determinePermissions = DetermineIfStaffPermissions()
    permissions = determinePermissions.has_permission(request, None)

    # The assertion is that permissions weren't given.
    assert permissions == False

@pytest.mark.django_db
def test_user_no_staff_permissions_request():
    # Defining the user and making them a staff member.
    user = User.objects.create_user(username='testuser', password='testpassword')
    user.is_staff = True;

    # Creating the request and assigning the user to it.
    request = APIRequestFactory().get('modelworx')
    request.user = user

    # Using the permissions.py class to check if the user has permissions to access the page.
    determinePermissions = DetermineIfStaffPermissions()
    permissions = determinePermissions.has_permission(request, None)

    # Asserting that the user does have permissions to view the page.
    assert permissions == True
