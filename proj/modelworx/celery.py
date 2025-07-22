import os
from celery import Celery
from celery.signals import setup_logging

import logging
from logging.config import dictConfig

from django.conf import settings

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'modelworx.settings')
app = Celery('modelworx')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()


@setup_logging.connect
def on_celery_setup_logging(**kwargs):
    logging.config.dictConfig(settings.LOGGING)

