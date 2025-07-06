from django.urls import path

from .views import SignUpView
from .views import LoginView

urlpatterns = [
    #path("signup/", views.signup, name="signup"),
    path("signup/", SignUpView.as_view(), name="signup"),
    path("login/", LoginView.as_view(), name="login"),
]
