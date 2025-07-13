
from django.urls import reverse_lazy
from django.views.generic import CreateView
from django.contrib.auth.views import LoginView
from django.contrib.messages.views import SuccessMessageMixin

from .forms import UserRegisterForm
from .forms import UserLoginForm

class SignUpView(SuccessMessageMixin, CreateView):
  template_name = 'registration/signup.html'
  success_url = reverse_lazy('login')
  form_class = UserRegisterForm
  success_message = "Your profile was created successfully"


class LoginView(LoginView):
  template_name = 'registration/login.html'
  success_url = reverse_lazy('index')
  form_class = UserLoginForm
  