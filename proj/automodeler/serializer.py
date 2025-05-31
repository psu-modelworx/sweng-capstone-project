# This serializer defines fields from users with an account and stores them in a view set.
# The view set will be used in the modelworx\urls.py file to show user data at a modelworx path.

from django.contrib.auth.models import User
from rest_framework import serializers, viewsets

class ModelWorxSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = User # Defining a model using the Django User model.
        fields = ['date_joined', 'last_login', 'is_active', 'is_staff', 'username', 'url',] # Chose important fields to be collected and displayed at a modelworx path.

class ModelWorxViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all() # Creating a list of users that will be serialized by the ModelWorkxSerializer class to get chosen fields.
    serializer_class = ModelWorxSerializer # Setting the serializer to the ModelWorxSerializer class.