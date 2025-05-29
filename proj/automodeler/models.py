from django.db import models

# Create your models here.
class Dataset(models.Model):
    name = models.CharField(max_length=50)
    input_fields = models.JSONField()
    output_fields = models.JSONField()
    csv_file = models.FileField(upload_to='dataset_uploads/')

