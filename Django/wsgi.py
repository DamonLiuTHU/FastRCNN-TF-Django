"""
WSGI config for Django project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/1.11/howto/deployment/wsgi/
"""

import os

import sys

root = os.path.join(os.path.dirname(__file__), '..') # add parent path
sys.path.insert(0, root) # add to sys path


from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "Django.settings")

application = get_wsgi_application()