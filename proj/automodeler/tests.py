#from django.test import TestCase
import pytest
from django.urls import reverse
from django.contrib.auth.models import User

# Create your tests here.
def test_func():
    assert 0 == 0


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
