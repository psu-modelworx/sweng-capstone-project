from django.db import models
from django.conf import settings

# Create your models here.
class Dataset(models.Model):
    name = models.CharField(max_length=50)
    input_fields = models.JSONField(null=True)
    output_fields = models.JSONField(null=True)
    csv_file = models.FileField(upload_to='dataset_uploads/')
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)

