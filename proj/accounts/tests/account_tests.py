import pytest
from pytest_django.asserts import assertRedirects
from django.urls import reverse, reverse_lazy
from django.contrib import auth
from django.contrib.auth.models import User


from accounts import views

# Start unit tests here
@pytest.mark.django_db
def test_login_get(client):
    """
    Test Get on login page

    TC-25
    """
    user = User.objects.create_user(username='testuser', password='testpassword')
    assert not user.is_superuser
    url = reverse("login")
    response = client.get(url)
    assert response.status_code == 200


@pytest.mark.django_db
def test_login_post(client):
    """
    Test post on login page and checking that user is authenticated
    
    TC-26
    """
    user = User.objects.create_user(username='testuser', password='testpassword')
    assert not user.is_superuser
    #assert user.is_authenticated
    url = reverse("login")
    
    # Successful login
    response = client.post(url, {"username": user.username, "password": user.password})
    assert response.status_code == 200

    # Validate user is authenticated
    result = client.login(username="testuser", password="testpassword")
    response = client.get(reverse('index'))
    assert response.context['request'].user.is_authenticated

def test_logout(client):
    """
    Test Logout Functionality.  Verifies redirect from logout

    TC-27
    """
    url = reverse("logout")
    response = client.post(url)
    assert response.status_code == 302
    assert response.headers['Location'] == reverse('index')

@pytest.mark.django_db
def test_signup(client):
    """
    Test signup form

    TC-28
    """
    url = reverse("signup")
    username = "testuser"
    password1 = "testpassword"
    password2 = "testpassword"
    
    params = {"username": username, "password1": password1, "password2": password2}

    response = client.post(url, params)
    assert response.status_code == 302
    

#def test_dataset_configure():
#    return False


# Start system tests here

@pytest.mark.django_db
def test_signup_login_logout(client):
    """
    System test that will check if a user can signup, login,
    access the index, then logout.

    TC-29
    """
    # Create test user; 
    username = "testuser"
    password = "testpassword"
    url = reverse('signup')
    signup_params = {"username": username, "password1": password, "password2": password}

    # First, get the signup form
    response = client.get(url)
    assert response.status_code == 200

    # Next, post to signup to create the user; Response location will be the login form
    response = client.post(url, signup_params)
    assert response.status_code == 302

    # Now we need to login using the form
    url = reverse('login')
    login_params = {"username": username, "password": password}
    response = client.post(url, login_params)
    assert response.status_code == 302
    assert response.headers['Location'] == reverse('index')
    
    # Verify the user is authenticated
    url = reverse('index')

    # Validate user is authenticated
    result = client.login(username="testuser", password="testpassword")
    response = client.get(url)
    assert response.context['request'].user.is_authenticated

    # Finally, we test logging out
    url = reverse('logout')
    response = client.post(url)
    assert response.status_code == 302
    assert response.headers['Location'] == reverse('index')
    
    # Verify redirect to login since user is no longer authenticated
    response = client.get(reverse('dataset_collection'))
    assert response.status_code == 302
    #'accounts/login/?next=/automodeler/dataset_collection/''
    expected_redirecturi = "".join([reverse('login'), "?next=", reverse('dataset_collection')])
    assert response.headers['Location'] == expected_redirecturi


    


