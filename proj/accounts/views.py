
from django.urls import reverse_lazy
from django.views.generic import CreateView
from django.contrib.auth.views import LoginView
from django.contrib.messages.views import SuccessMessageMixin
from django.http import JsonResponse

from .forms import UserRegisterForm
from .forms import UserLoginForm

from django.conf import settings

from .models import EmailTask
from .tasks import send_account_creation_email_task

class SignUpView(SuccessMessageMixin, CreateView):
  template_name = 'registration/signup.html'
  success_url = reverse_lazy('login')
  form_class = UserRegisterForm
  success_message = "Your profile was created successfully"

  def form_valid(self, form):
    # Save the form
    form.save()

    # Send email to admins
    email_enabled = settings.EMAIL_ENABLED
    if email_enabled:
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
    # Launch celery task async
    async_results = send_account_creation_email_task.apply_async(args=[useremail, username])
    
    # Create UserTask
    email_task = EmailTask.objects.create( 
        task_id=async_results.id,
        task_type='emailing', 
        status='PENDING',
    )

    email_task.save() # Saving email to DB as ruff said it wasn't being used
    
    return JsonResponse({"task id": async_results.id})

  
