from django.contrib.auth.forms import UserCreationForm
#from django.urls import reverse_lazy
#from django.views.generic import CreateView

from django.shortcuts import render
from django.http import HttpResponse


#class SignUpView(CreateView):
#    form_class = UserCreationForm
#    success_url = reverse_lazy("login")
#    template_name = "registration/signup.html"
#
#
def signup(request):
    if request.method == 'GET':
        return render(request, "registration/signup.html", {})
    elif request.method == 'POST':
        print("Received POST request!")
        for key, value in request.POST.items():
            print(f'{key}: {value}')
        user_form = UserCreationForm(request.POST)
        if user_form.is_valid():
            #print("Valid form!")
            user_form.save()
            return render(request, "registration/login.html", {})
        else:
            return render(request, "registration/signup.html", {})
    else:
        return HttpResponseNotAllowed()