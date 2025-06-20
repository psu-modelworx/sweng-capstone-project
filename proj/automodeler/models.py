from django.db import models
from django.conf import settings

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

    # Overriding default delete function to delete associated files when object is being deleted
    def delete(self, *args, **kwargs):
        if self.csv_file:
            if os.path.isfile(self.csv_file.path):
                os.remove(self.csv_file.path)
        if self.feature_encoder:
            if os.path.isfile(self.feature_encoder.path):
                os.remove(self.feature_encoder.path)
        if self.scaler:
            if os.path.isfile(self.scaler.path):
                os.remove(self.scaler.path)
        if self.label_encoder:
            if os.path.isfile(self.label_encoder.path):
                os.remove(self.label_encoder.path)
        super().delete(*args, **kwargs)


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
    
    

    