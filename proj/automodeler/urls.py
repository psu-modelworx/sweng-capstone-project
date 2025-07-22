from django.urls import path

from . import views
from . import api
from . import engine_manager


urlpatterns = [
    path("", views.index, name="index"),
    path("automodeler/", views.index, name="automodeler"),
    path("upload/", views.upload, name="upload"),
    path("api/upload", api.api_upload, name="api_upload"),
    path("api/request_datasets", api.api_request_datasets, name="api_request_datasets"),
    path("api/request_models", api.api_request_models, name="api_request_models"),
    path("dataset_delete/<int:dataset_id>", views.dataset_delete, name="dataset_delete"),
    path("dataset/<int:dataset_id>", views.dataset, name="dataset"),
    path("dataset_details/<int:dataset_id>", views.dataset_details, name="dataset_details"),
    path("dataset_collection/", views.dataset_collection, name="dataset_collection"),
    path("model_collection/", views.model_collection, name="model_collection"),
    path("model_details/<int:model_id>", views.model_details, name="model_details"),
    path("model_download/<int:model_id>", views.model_download, name="model_download"),
    path("model_delete/", views.model_delete, name="model_delete"),
    path("task_collection/", views.task_collection, name="task_collection"),
    path("account/", views.account, name="account"),
    path("account_delete/", views.account_delete, name='account_delete'),
    path("ppe/start_preprocessing_request/", engine_manager.start_preprocessing_request, name="ppe"),
    path("mod/start_modeling_request/", engine_manager.start_modeling_request, name="ame"), # ame = automated model engine
    path("mod/run_model/", engine_manager.run_model, name="run_model"),
    path('check_task/<str:task_id>/', engine_manager.check_task_result, name='check_task_result'),
    #path('check_ppe_task/<str:task_id>/', engine_manager.check_preprocessing_task_result, name='check_preprocessing_task_result'),
    #path('check_run_model_task/<str:task_id>/', engine_manager.check_run_model_task_result, name='check_run_model_task_result')
]