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
    meta_data = JSONField(default=dict)
    original_dataset = models.OneToOneField(Dataset, on_delete=models.CASCADE)
    
    def filename(self):
        return os.path.basename(self.csv_file.name)


class TrainTestDataFrame(models.Model):
    POSSIBLE_TYPES = [
        ('train', 'Train'),
        ('test', 'Test')
    ]
    POSSIBLE_AXIS = [
        ('x', 'X'),
        ('y', 'Y')
    ]
    type = models.CharField(max_length=10, choices=POSSIBLE_TYPES) # Train/Test
    axis = models.CharField(max_length=1) # X or Y
    tt_ds_file = models.FileField(upload_to='traintest_dataframes/')
    preprocessed_dataset = models.ForeignKey(PreprocessedDataSet, on_delete=models.CASCADE)
    
    

    