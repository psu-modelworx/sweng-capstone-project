import pytest
from django.urls import reverse
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile


# Start unit tests here

@pytest.mark.django_db
def test_upload_get(client):
    """
    Test GET on upload page
    """

    # Create user that can be used for test
    user = User.objects.create_user(username='testuser', password='testpassword')
    assert not user.is_superuser
    url = reverse("upload")
    
    # Login with test user
    response = client.login(username="testuser", password="testpassword")
    response = client.get(url)
    assert response.status_code == 200

@pytest.mark.django_db
def test_upload_post(client):
    """
    Test POST on upload page
    """
    user = User.objects.create_user(username='testuser', password='testpassword')
    assert not user.is_superuser
    url = reverse("upload")

    # Login with test user
    response = client.login(username="testuser", password="testpassword")
    
    # Create post parameters
    fileName = "testFileName"
    file_content = open('test_datasets/iris.data.csv', "rb")
    file_to_upload = SimpleUploadedFile(
        content = file_content.read(),
        name=file_content.name,
        content_type="multipart/form-data"
    )
    upload_params = {"name": fileName, "csv_file": file_to_upload}
    response = client.post(url, upload_params)
    assert response.status_code == 302


