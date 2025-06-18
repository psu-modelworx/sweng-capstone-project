from django.urls import path

from . import views
from . import api

urlpatterns = [
    path("", views.index, name="index"),
    path("upload/", views.upload, name="upload"),
    path("api/upload", api.api_upload, name="api_upload"),
    path("dataset_delete/<int:dataset_id>", views.dataset_delete, name="dataset_delete"),
    path("dataset/<int:dataset_id>", views.dataset, name="dataset"),
    path("dataset_collection/", views.dataset_collection, name="dataset_collection"),
    path("model_collection/", views.model_collection, name="model_collection"),
    path("task_collection/", views.task_collection, name="task_collection"),
    path("account/", views.account, name="account"),

]