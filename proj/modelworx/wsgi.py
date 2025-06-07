"""
WSGI config for modelworx project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/howto/deployment/wsgi/
"""
import os


from decouple import Config, RepositoryEnv

# Set path to .env.production
base_dir = os.path.dirname(os.path.abspath(__file__))
prod_env = os.path.join(base_dir, ".env.production")

# Load .env.production explicitly
if os.path.exists(prod_env):
    config = Config(RepositoryEnv(prod_env))
else:
    config = Config(RepositoryEnv(os.path.join(base_dir, ".env")))  # fallback to .env

# Set environment variables BEFORE Django loads
os.environ["DJANGO_DEBUG"] = config("DJANGO_DEBUG", default="False")
os.environ["DJANGO_ALLOWED_HOSTS"] = config("DJANGO_ALLOWED_HOSTS", default="localhost")

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "modelworx.settings")

from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()
