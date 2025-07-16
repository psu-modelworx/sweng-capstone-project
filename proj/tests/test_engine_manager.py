import pytest
from django.urls import reverse
from automodeler.models import Dataset, DatasetModel, TunedDatasetModel
from django.contrib.auth import get_user_model
from unittest.mock import patch, MagicMock
import json

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
@patch('automodeler.engine_manager.start_preprocessing_task.apply_async')
def test_start_preprocessing_request_system(mock_apply_async, client, user_factory):
    """ TC-59 Verify start preprocessing.""" 
    user = user_factory()
    dataset = Dataset.objects.create(name="MyData", user=user, features={'f1': 'C'}, target_feature='f1')
    client.force_login(user)

    mock_async_result = MagicMock()
    mock_async_result.id = 'fake-task-id'
    mock_apply_async.return_value = mock_async_result

    response = client.post(reverse("ppe"), {"dataset_id": dataset.id})
   
    assert response.status_code == 200
    assert "task_id" in response.json()


@pytest.mark.django_db
@patch('automodeler.engine_manager.start_modeling_task.apply_async')
def test_start_modeling_request_system(mock_apply_async, client, user_factory):
    """ TC-60 Verify start modeling.""" 
    mock_task = MagicMock()
    mock_task.id = 'fake-modeling-task-id'
    mock_apply_async.return_value = mock_task

    user = user_factory()
    client.force_login(user)

    dataset = Dataset.objects.create(name="MyData", user=user, features={'f1': 'C'}, target_feature='f1')

    response = client.post(reverse("ame"), {"dataset_id": dataset.id})

    assert response.status_code == 200
    assert 'fake-modeling-task-id' in response.content.decode()

@pytest.mark.django_db
def test_run_model_system(client, user_factory):
    """ TC-61 Verify start run model.""" 
    user = user_factory()
    client.force_login(user)

    dataset = Dataset.objects.create(name="Test", user=user, features={"f1": "N"})

    untuned_model = DatasetModel.objects.create(
        user=user,
        original_dataset=dataset,
        model_type="classification",
        model_file="dummy_untuned_model.pkl"
    )

    tuned_model = TunedDatasetModel.objects.create(
        user=user,
        original_dataset=dataset,
        model_type="classification",
        model_file="dummy_model.pkl",
        untuned_model=untuned_model  
    )

    response = client.post(
        reverse("run_model"),
        {
            "model_id": tuned_model.id,
            "data": json.dumps({"values": [1.0]})
        }
    )

    assert response.status_code == 200
    assert "task_id" in response.json()