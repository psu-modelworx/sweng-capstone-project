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
    original_dataset = models.OneToOneField(Dataset, on_delete=models.CASCADE)
    
    def filename(self):
        return os.path.basename(self.csv_file.name)