from django.urls import path

from . import views
from .views import SignUpView

urlpatterns = [
    #path("signup/", views.signup, name="signup"),
    path("signup/", SignUpView.as_view(), name="signup")
]
