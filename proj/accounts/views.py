
from django.urls import reverse_lazy
from django.views.generic import CreateView
from django.contrib.auth.views import LoginView
from django.contrib.messages.views import SuccessMessageMixin

from .forms import UserRegisterForm
from .forms import UserLoginForm

from django.core.mail import EmailMessage
from django.http import HttpResponse
from django.conf import settings



class SignUpView(SuccessMessageMixin, CreateView):
  template_name = 'registration/signup.html'
  success_url = reverse_lazy('login')
  form_class = UserRegisterForm
  success_message = "Your profile was created successfully"

  def form_valid(self, form):
    # Save the form
    user = form.save()

    # Send email to admins
    email_enabled = settings.EMAIL_ENABLED
    if email_enabled == True:
      username = form.cleaned_data['username']
      email = form.cleaned_data['email']
      send_account_create_request_email(email, username)

    # Return the parent class' form_valid method
    return super().form_valid(form)


class LoginView(LoginView):
  template_name = 'registration/login.html'
  success_url = reverse_lazy('index')
  form_class = UserLoginForm




def send_account_create_request_email(useremail, username):
  msg_body = """
  The following user has requested Modelworx access:
  User Email Address:  {0}
  Username:  {1}
  """.format(useremail, username)

  email_admins = settings.EMAIL_ADMINS.split(',')
  email_sender = settings.EMAIL_SENDER

  message = EmailMessage(
    'Modelworx Account Request',
    msg_body,
    email_sender,
    email_admins
  )
  try:
    message.send()
  except Exception as e:
    print("Message send failed! Exception: {0}".format(e))
  
