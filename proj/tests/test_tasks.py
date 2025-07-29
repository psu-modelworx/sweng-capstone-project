import pytest
from django.contrib.auth import get_user_model
from unittest.mock import patch, MagicMock
from django.core.exceptions import ObjectDoesNotExist
from automodeler.tasks import start_preprocessing_task, start_modeling_task, run_model_task,reconstruct_ppe, obj_to_pkl_file, pkl_file_to_obj

from automodeler.models import Dataset, DatasetModel
from django.core.files.base import ContentFile
import pandas as pd
import pickle


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

@patch('automodeler.tasks.Dataset.objects.get')
@patch('automodeler.tasks.UserTask.objects.create')
@patch('pandas.read_csv')
@patch('automodeler.tasks.PreprocessedDataSet.objects.get')
@patch('automodeler.tasks.PreprocessingEngine')
@patch('automodeler.tasks.obj_to_pkl_file', return_value=ContentFile(b'dummy-bytes', name='dummy_file.pkl'))
@pytest.mark.django_db
def test_start_preprocessing_success(mock_obj_to_pkl, mock_engine_cls, mock_pp_ds_get,mock_read_csv, mock_usertask_create, mock_dataset_get, user_factory):
    """TC-52 Test successful start of preprocessing task.."""

    user = user_factory()
    
    dataset = Dataset.objects.create(
        name='TestDataset',
        user=user,
        features={'f1': 'C'},
        target_feature='f1',
        csv_file='mock.csv',
        number_of_rows=150,
        file_size=100
    )
    mock_dataset_get.return_value = dataset

    mock_pp_ds_get.side_effect = ObjectDoesNotExist("Doesn't exist")

    mock_read_csv.return_value = pd.DataFrame({'f1': [1, 2, 3]})

    mock_ppe_instance = MagicMock()
    mock_ppe_instance.final_df = pd.DataFrame({'f1': [1, 2, 3]})
    mock_ppe_instance.feature_encoder = MagicMock()
    mock_ppe_instance.scaler = MagicMock()
    mock_ppe_instance.label_encoder = MagicMock()
    mock_ppe_instance.dropped_columns = {}
    mock_ppe_instance.to_meta_dict.return_value = {}
    mock_ppe_instance.task_type = 'classification'
    mock_engine_cls.return_value = mock_ppe_instance

    # make sure the `run_preprocessing_engine()` doesn't fail
    mock_ppe_instance.run_preprocessing_engine.return_value = None

    result = start_preprocessing_task(dataset_id=1, user_id=user.id)

    assert result["status"] == 200
    assert "TestDataset_preprocessed.csv" in result["filename"]

@pytest.mark.django_db
@patch('automodeler.tasks.Dataset.objects.get', side_effect=ObjectDoesNotExist)
@patch('automodeler.tasks.UserTask.objects.filter')
def test_start_preprocessing_dataset_not_found(mock_user_task_filter, mock_dataset_get, user_factory):
    """TC-53 Test start preprocessing when dataset not found."""
    user = user_factory()
    mock_user_task_filter.return_value.first.return_value = MagicMock()
    result = start_preprocessing_task(999, user.id)
    assert result['status'] == 404
    assert "Dataset not found" in result['message']

@pytest.mark.django_db
@patch('automodeler.tasks.Dataset.objects.get')
@patch('automodeler.tasks.PreprocessedDataSet.objects.get')
@patch('automodeler.tasks.UserTask.objects.filter')
@patch('automodeler.tasks.reconstruct_ppe')
@patch('automodeler.tasks.ModelingEngine')
@patch('automodeler.tasks.obj_to_pkl_file')
def test_start_modeling_success(mock_obj_to_pkl, mock_modeling_engine_cls, mock_reconstruct_ppe,
                               mock_user_task_filter, mock_pp_ds_get, mock_dataset_get, user_factory):
    """TC-54 Test successful start of modeling task."""
    user = user_factory()

    # Create a real Dataset instance in test DB
    dataset = Dataset.objects.create(
        name='TestDataset',
        user_id=user.id,
        features={'f1': 'C'}
    )
    mock_dataset_get.return_value = dataset

    mock_pp_ds = MagicMock()
    mock_pp_ds_get.return_value = mock_pp_ds
    mock_pp_ds.model_type = 'classification'

    mock_reconstruct_ppe.return_value = MagicMock(
        task_type='classification',
        split_data=lambda: ([], [], [], [])
    )

    mock_engine = MagicMock()
    mock_engine.results = {
        'untuned': {
            'modelA': {'model': MagicMock(name='UntunedModel')},
        },
        'tuned': {
            'modelA': {
                'optimized_model': MagicMock(name='TunedModel'),
                'final_scores': dict({'f1_score': 1})
                },
        }
    }
    mock_modeling_engine_cls.return_value = mock_engine
    mock_engine.task_type = 'classification'

    mock_task = MagicMock()
    mock_user_task_filter.return_value.first.return_value = mock_task

    mock_obj_to_pkl.side_effect = lambda obj, fname: f'/tmp/{fname}'

    result = start_modeling_task(dataset.id, user.id)
    assert result.get('status', '') != 'FAILURE'

@pytest.mark.django_db
@patch('automodeler.tasks.Dataset.objects.get', side_effect=ObjectDoesNotExist)
@patch('automodeler.tasks.UserTask.objects.filter')
def test_start_modeling_dataset_not_found(mock_user_task_filter, mock_dataset_get, user_factory):
    """TC-55 Test start modeling task when dataset not found."""
    user = user_factory()
    mock_task = MagicMock()
    mock_user_task_filter.return_value.first.return_value = mock_task
    result = start_modeling_task(999, user.id)
    assert result['status'] == 404
    assert mock_task.status == "FAILURE"

@pytest.mark.django_db
@patch('automodeler.tasks.DatasetModel.objects.get', side_effect=DatasetModel.DoesNotExist)
@patch('automodeler.tasks.UserTask.objects.filter')
def test_run_model_task_tuned_model_not_found(mock_user_task_filter, mock_tuned_model_get, user_factory):
   """TC-56 Test run_model_task returns 404 if tuned model not found."""
   user = user_factory()
   mock_user_task_filter.return_value.first.return_value = MagicMock()
   data_dict = {'values': [1.0]}
   result = run_model_task(999, user.id, data_dict)
   assert isinstance(result, dict)
   assert result.get("status") == 404
   assert "Tuned model not found" in result.get("message", "")

@pytest.mark.django_db
@patch('automodeler.tasks.DatasetModel.objects.get')
def test_run_model_task_invalid_feature_count(mock_tuned_model_get, user_factory):
    """TC-57 Test run_model_task with invalid feature count input."""
    user = user_factory()
    mock_tuned_model = MagicMock(id=1, original_dataset_id=1, user_id=user.id, model_type='classification', model_file='dummy')
    mock_tuned_model_get.return_value = mock_tuned_model

    with patch('automodeler.tasks.Dataset.objects.get') as mock_dataset_get, \
         patch('automodeler.tasks.PreprocessedDataSet.objects.get') as mock_pp_ds_get, \
         patch('automodeler.tasks.UserTask.objects.filter') as mock_user_task_filter:

        mock_dataset = MagicMock(id=1, features={'f1': 'C', 'f2': 'N'})
        mock_dataset_get.return_value = mock_dataset
        mock_pp_ds_get.return_value = MagicMock()
        mock_task = MagicMock()
        mock_user_task_filter.return_value.first.return_value = mock_task

        data_dict = {'values': [1.0]}  # Only 1 value, expect failure
        result = run_model_task(1, user.id, data_dict)

        assert result['status'] == 400
        assert "Invalid number of input features" in result['message']
        assert mock_task.status == "FAILURE"

@pytest.mark.django_db
@patch('automodeler.tasks.DatasetModel.objects.get')
@patch('automodeler.tasks.Dataset.objects.get')
@patch('automodeler.tasks.PreprocessedDataSet.objects.get')
@patch('automodeler.tasks.UserTask.objects.filter')
@patch('automodeler.tasks.reconstruct_ppe')
@patch('automodeler.tasks.pkl_file_to_obj')
@patch('pandas.DataFrame')
def test_run_model_task_success(mock_df, mock_pkl_file_to_obj, mock_reconstruct_ppe,
                               mock_user_task_filter, mock_pp_ds_get, mock_dataset_get,
                               mock_tuned_model_get, user_factory):
    """TC-58 Test successful run_model_task prediction workflow."""
    user = user_factory()
    mock_tuned_model = MagicMock(id=1, original_dataset_id=1, user_id=user.id, model_type='classification', model_file='dummy', tuned=True)
    mock_tuned_model_get.return_value = mock_tuned_model
    mock_dataset = MagicMock(id=1, name='TestDataset', features={'f1': 'float'})
    mock_dataset_get.return_value = mock_dataset
    mock_pp_ds = MagicMock()
    mock_pp_ds_get.return_value = mock_pp_ds

    mock_reconstruct_ppe.return_value = MagicMock(
        transform_single_row=lambda df: df,
        decode_target=lambda x: ['predicted_label']
    )
    mock_model_obj = MagicMock()
    mock_model_obj.predict.return_value = ['predicted_label']
    mock_pkl_file_to_obj.return_value = mock_model_obj

    mock_task = MagicMock()
    mock_user_task_filter.return_value.first.return_value = mock_task

    mock_df.return_value = ['mocked dataframe']

    data_dict = {'values': [1.0]}
    result = run_model_task(1, user.id, data_dict)

    mock_tuned_model_get.assert_called_once()
    mock_dataset_get.assert_called_once()
    mock_pp_ds_get.assert_called_once()
    assert mock_task.status == "SUCCESS"
    assert "Predicted result" in result['message']

@pytest.mark.django_db
@patch('automodeler.tasks.Dataset.objects.get')
@patch('automodeler.tasks.UserTask.objects.filter')
def test_start_preprocessing_missing_target_feature(mock_user_task_filter, mock_dataset_get, user_factory):
    """TC-99 """
    user = user_factory()
    dataset = Dataset.objects.create(
        name='TestDataset',
        user=user,
        features={'f1': 'C'},
        target_feature=''  # Empty target_feature triggers exception
    )
    mock_dataset_get.return_value = dataset
    mock_task = MagicMock()
    mock_user_task_filter.return_value.first.return_value = mock_task

    result = start_preprocessing_task(dataset_id=dataset.id, user_id=user.id)

    assert result['status'] == 500
    assert "Target feature not selected" in result['message']
    assert mock_task.status == "FAILURE"

@pytest.mark.django_db
@patch('automodeler.tasks.Dataset.objects.get')
@patch('automodeler.tasks.UserTask.objects.filter')
@patch('automodeler.tasks.PreprocessedDataSet.objects.get', side_effect=ObjectDoesNotExist)
@patch('automodeler.tasks.pd.read_csv')
@patch('automodeler.tasks.PreprocessingEngine')
def test_start_preprocessing_run_engine_exception(mock_engine_cls, mock_read_csv, mock_pp_ds_get, mock_user_task_filter, mock_dataset_get, user_factory):
    """ TC-100 """
    user = user_factory()
    dataset = Dataset.objects.create(
        name='TestDataset',
        user=user,
        features={'f1': 'C'},
        target_feature='f1',
        csv_file='mock.csv'
    )
    mock_dataset_get.return_value = dataset
    mock_task = MagicMock()
    mock_user_task_filter.return_value.first.return_value = mock_task

    mock_read_csv.return_value = pd.DataFrame({'f1': [1, 2, 3]})

    mock_ppe_instance = MagicMock()
    mock_ppe_instance.run_preprocessing_engine.side_effect = Exception("Preprocessing engine error")
    mock_engine_cls.return_value = mock_ppe_instance

    result = start_preprocessing_task(dataset_id=dataset.id, user_id=user.id)

    assert result['status'] == 500
    assert "Error running preprocessing engine" in result['message']
    assert mock_task.status == "FAILURE"


#@pytest.mark.django_db
#@patch('automodeler.tasks.Dataset.objects.get', side_effect=ObjectDoesNotExist)
#@patch('automodeler.tasks.UserTask.objects.filter')
#def test_start_modeling_dataset_not_found_two(mock_user_task_filter, mock_dataset_get, user_factory):
#    """TC-101 """
#    user = user_factory()
#    mock_task = MagicMock()
#    mock_user_task_filter.return_value.first.return_value = mock_task
#
#    result = start_modeling_task(999, user.id)

#@pytest.mark.django_db
#@patch('automodeler.tasks.Dataset.objects.get', side_effect=ObjectDoesNotExist)
#@patch('automodeler.tasks.UserTask.objects.filter')
#def test_start_modeling_dataset_not_found(mock_user_task_filter, mock_dataset_get, user_factory):
#    """TC-101 """
#    user = user_factory()
#    mock_task = MagicMock()
#    mock_user_task_filter.return_value.first.return_value = mock_task
#
#    result = start_modeling_task(999, user.id)
#
#    assert result['status'] == 404
#    assert mock_task.status == "FAILURE"

@pytest.mark.django_db
@patch('automodeler.tasks.Dataset.objects.get')
@patch('automodeler.tasks.PreprocessedDataSet.objects.get', side_effect=ObjectDoesNotExist)
@patch('automodeler.tasks.UserTask.objects.filter')
def test_start_modeling_pp_ds_not_found(mock_user_task_filter, mock_pp_ds_get, mock_dataset_get, user_factory):
    """TC-102 """
    user = user_factory()
    dataset = Dataset.objects.create(name='TestDataset', user=user, features={'f1': 'C'})
    mock_dataset_get.return_value = dataset
    mock_user_task_filter.return_value.first.return_value = MagicMock()

    result = start_modeling_task(dataset.id, user.id)

    assert result['status'] == 412
    assert "preprocessed" in result['message'].lower()

@pytest.mark.django_db
@patch('automodeler.tasks.DatasetModel.objects.get')
@patch('automodeler.tasks.Dataset.objects.get', side_effect=ObjectDoesNotExist)
@patch('automodeler.tasks.UserTask.objects.filter')
def test_run_model_task_preprocessed_dataset_not_found(mock_user_task_filter, mock_dataset_get, mock_tuned_model_get, user_factory):
    """ TC-103 """ 
    user = user_factory()
    # Mock the tuned model to have original_dataset_id
    mock_tuned_model = MagicMock(original_dataset_id=999, user_id=user.id, tuned=True)
    mock_tuned_model_get.return_value = mock_tuned_model

    # Dataset.objects.get will raise ObjectDoesNotExist (simulate failure here)
    mock_task = MagicMock()
    mock_user_task_filter.return_value.first.return_value = mock_task

    data_dict = {'values': [1.0]}  # dummy input to satisfy run_model_task input

    result = run_model_task(model_id=1, user_id=user.id, data_dict=data_dict)

    assert result['status'] == 404
    assert "Error retrieving preprocessed dataset" in result['message']
    assert mock_task.status == "FAILURE"


@pytest.mark.django_db
@patch('automodeler.tasks.pd.read_csv')
@patch('automodeler.tasks.pkl_file_to_obj')
@patch('automodeler.tasks.PreprocessingEngine')
def test_reconstruct_ppe(mock_ppe_cls, mock_pkl_file_to_obj, mock_read_csv):
    """ TC-104 """ 
    # Prepare mocks
    fake_df = MagicMock(name='DataFrame')
    mock_read_csv.return_value = fake_df

    mock_pkl_file_to_obj.side_effect = ['feature_encoder_obj', 'scaler_obj', 'label_encoder_obj']

    mock_ppe_instance = MagicMock(name='PreprocessingEngineInstance')
    mock_ppe_cls.load_from_files.return_value = mock_ppe_instance

    # Create a fake PreprocessedDataSet-like object with necessary attributes
    pp_ds_mock = MagicMock()
    pp_ds_mock.csv_file = 'fake_path.csv'  # Path not used due to patching pd.read_csv
    pp_ds_mock.meta_data = {'some': 'meta'}
    pp_ds_mock.feature_encoder = 'fe_file'
    pp_ds_mock.scaler = 'scaler_file'
    pp_ds_mock.label_encoder = 'label_enc_file'

    result = reconstruct_ppe(pp_ds_mock)

    # Assert pd.read_csv called with csv_file attribute
    mock_read_csv.assert_called_once_with(pp_ds_mock.csv_file)
    # Assert pkl_file_to_obj called for each file
    assert mock_pkl_file_to_obj.call_count == 3

    # Assert load_from_files called with expected arguments
    mock_ppe_cls.load_from_files.assert_called_once_with(
        meta=pp_ds_mock.meta_data,
        clean_df=fake_df,
        feature_encoder='feature_encoder_obj',
        scaler='scaler_obj',
        label_encoder='label_encoder_obj'
    )

    assert result == mock_ppe_instance


def test_obj_to_pkl_file_and_pkl_file_to_obj():
    """ TC-105 """
    # Prepare a simple Python object
    data_obj = {'key': 'value'}

    # Use obj_to_pkl_file to serialize to ContentFile
    content_file = obj_to_pkl_file(data_obj, 'test_file.pkl')

    assert isinstance(content_file, ContentFile)
    assert content_file.name == 'test_file.pkl'

    # The content should be pickled bytes, so try unpickling directly
    unpickled_obj = pickle.loads(content_file.read())

    assert unpickled_obj == data_obj

    # Now test pkl_file_to_obj on this ContentFile
    content_file.seek(0)  # Reset file pointer to start

    result_obj = pkl_file_to_obj(content_file)

    assert result_obj == data_obj