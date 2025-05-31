# This class takes the permission request and grants it if the user's "is_staff" field is true.
from rest_framework.permissions import BasePermission

class DetermineIfStaffPermissions(BasePermission):
    def has_permission(self, request, view):
        # If the user is staff, give them permission for the API response.
        if request.user.is_staff:
            return True
        # Defaulting to not giving permission to see the content in the API response.
        return False