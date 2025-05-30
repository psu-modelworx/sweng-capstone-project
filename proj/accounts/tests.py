import pytest
from django.urls import reverse, reverse_lazy
from django.contrib.auth.models import User

from . import views

# Create your tests here.
@pytest.mark.django_db
def test_login_logout(client):
    # Create test user; 
    user = User.objects.create_user(username='testuser', password='testpassword')
    assert not user.is_superuser

    # Test GET on login page
    url = reverse('login')
    response = client.get(url)
    assert response.status_code == 200

    response = client.post(url, {'username': 'testuser', 'password': 'testpassword'})
    assert response.status_code == 302
    
    # Need to fix this; currently it's not able to reverse match this url
    # assert response.url == reverse("/automodeler/") 
    
    # Asserts that the session id is equal to the user.id
    assert client.session['_auth_user_id'] == str(user.id)
    
    # Logout usser and test
    url = reverse('logout')
    response = client.post(url)
    assert '_auth_user_id' not in client.session
    
    # Assert redirect after logout
    assert response.status_code == 302
    
    # Need to fix this; currently it's not able to reverse match this url similar to the one above
    # assert response.url == reverse("/automodeler/") 
    
    # Delete the testuser
    user.delete()
    

@pytest.mark.django_db
def test_signup(client):
    username = 'testuser'
    password1 = 'testpassword'
    password2 = 'testpassword'
    page = reverse('signup')
    url = client.get(page)
    response = client.post(page, {'username': username, 'password1': password1, 'password2': password2})
    # Proper response will redirect to login page
    assert response.status_code == 302
    

