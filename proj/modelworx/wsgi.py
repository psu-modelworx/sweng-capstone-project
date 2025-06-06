"""
WSGI config for modelworx project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/howto/deployment/wsgi/
"""

import os

from decouple import AutoConfig
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'modelworx.settings')

application = get_wsgi_application()


# Try to load .env.production if it exists, otherwise .env
base_dir = os.path.dirname(os.path.dirname(__file__))
prod_env = os.path.join(base_dir, ".env.production")

if os.path.exists(prod_env):
    config = AutoConfig(search_path=os.path.dirname(prod_env))
else:
    config = AutoConfig(search_path=base_dir)

os.environ.setdefault("DJANGO_DEBUG", config("DJANGO_DEBUG", default="True"))
os.environ.setdefault("DJANGO_ALLOWED_HOSTS", config("DJANGO_ALLOWED_HOSTS", default="localhost"))
