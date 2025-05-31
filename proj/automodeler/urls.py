from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("upload/", views.upload, name="upload"),
    path("dataset/<int:dataset_id>", views.dataset, name="dataset"),
    #path("upload_stage_two/", views.upload_stage_two, name="upload_stage_two"),
    path("save_dataset/", views.save_dataset, name="save_dataset"),
]