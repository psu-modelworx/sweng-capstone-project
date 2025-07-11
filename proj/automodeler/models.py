from django.db import models
from django.conf import settings
from django.db.models.signals import pre_delete
from django.dispatch import receiver

import os

# Create your models here.
class Dataset(models.Model):    
    name = models.CharField(max_length=50)
    features = models.JSONField(default=dict, null=True)
    target_feature = models.CharField(max_length=50, null=True)
    csv_file = models.FileField(upload_to='dataset_uploads/')
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)

    def filename(self):
        return os.path.basename(self.csv_file.name)
    
class PreprocessedDataSet(models.Model):
    name = models.CharField(max_length=100)
    csv_file = models.FileField(upload_to='preprocessed_datasets/')    
    feature_encoder = models.FileField(upload_to='pp_ds_bins/')
    scaler = models.FileField(upload_to='pp_ds_bins/')
    label_encoder = models.FileField(upload_to='pp_ds_bins/')
    meta_data = models.JSONField(default=dict)
    original_dataset = models.OneToOneField(Dataset, on_delete=models.CASCADE, blank=True, null=True)
    
    def filename(self):
        return os.path.basename(self.csv_file.name)


class DatasetModel(models.Model):
    MODEL_METHODS = {
        "LogisticRegression": "Logistic Regression",
        "RandomForestClassifier": "Random Forest Classifier",
        "GradientBoostingClassifier": "Gradient Boosting Classifier",
        "SVC": "SVC",
        "LinearRegression": "Linear Regression",
        "RandomForestRegressor": "Random Forest Regressor",
        "GradientBoostingRegressor": "Gradient Boosting Regressor",
        "SVR": "SVR"    
    }
    
    MODEL_TYPES = {
        "regression": "Regression",
        "classification": "Classification"
    }
    
    name = models.CharField(max_length=100)
    model_file = models.FileField(upload_to='models/')
    model_method = models.CharField(max_length=30, choices=MODEL_METHODS)
    model_type = models.CharField(max_length=15, choices=MODEL_TYPES)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    original_dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    
    
class TunedDatasetModel(models.Model):
    MODEL_METHODS = {
        "LogisticRegression": "Logistic Regression",
        "RandomForestClassifier": "Random Forest Classifier",
        "GradientBoostingClassifier": "Gradient Boosting Classifier",
        "SVC": "SVC",
        "LinearRegression": "Linear Regression",
        "RandomForestRegressor": "Random Forest Regressor",
        "GradientBoostingRegressor": "Gradient Boosting Regressor",
        "SVR": "SVR"    
    }
    
    MODEL_TYPES = {
        "regression": "Regression",
        "classification": "Classification"
    }
    
    name = models.CharField(max_length=100)
    model_file = models.FileField(upload_to='models/')
    model_method = models.CharField(max_length=30, choices=MODEL_METHODS)
    model_type = models.CharField(max_length=15, choices=MODEL_TYPES)
    untuned_model = models.OneToOneField(DatasetModel, on_delete=models.CASCADE)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    original_dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    
    
class UserTask(models.Model):
    TASK_TYPES = [
        ('preprocessing', 'Preprocessing'),
        ('modeling', 'Modeling'),
        ('prediction', 'Prediction'),
    ]

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    task_id = models.CharField(max_length=255, unique=True)
    task_type = models.CharField(max_length=50, choices=TASK_TYPES)
    status = models.CharField(max_length=50, default='PENDING')
    result_message = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, null=True, blank =True)

@receiver(pre_delete, sender=Dataset)
def delete_usertasks_with_dataset(sender, instance, **kwargs):
    # Delete all UserTasks related to this dataset
    UserTask.objects.filter(dataset=instance).delete()