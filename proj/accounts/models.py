from django.db import models

# Create your models here.

class EmailTask(models.Model):
    TASK_TYPES = [
        ('emailing', 'Emailing'),
    ]

    task_id = models.CharField(max_length=255, unique=True)
    task_type = models.CharField(max_length=50, choices=TASK_TYPES)
    status = models.CharField(max_length=50, default='PENDING')
    result_message = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    