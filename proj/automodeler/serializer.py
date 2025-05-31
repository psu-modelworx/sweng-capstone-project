# This serializer defines fields from users with an account and stores them in a view set.
# The view set will be used in the modelworx\urls.py file to show user data in an API response.
from django.contrib.auth.models import User
from rest_framework import serializers, viewsets
from rest_framework.authentication import SessionAuthentication, BasicAuthentication
from .permissions import DetermineIfStaffPermissions

class ModelWorxSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = User # Defining a model using the Django User model.
        fields = ['date_joined', 'last_login', 'is_active', 'is_staff', 'username', 'url',] # Chose important fields to be collected and displayed at a modelworx path.

class ModelWorxViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all() # Creating a list of users that will be serialized by the ModelWorkxSerializer class to get chosen fields.
    serializer_class = ModelWorxSerializer # Setting the serializer to the ModelWorxSerializer class.
    authentication_classes = [SessionAuthentication, BasicAuthentication] # Ensuring someone is signed to show them the complete API response.
    permission_classes = [DetermineIfStaffPermissions] # Checking that the user has "is_staff" set to true before showing them the API response.