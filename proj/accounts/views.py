from django.contrib.auth.forms import UserCreationForm
#from django.urls import reverse_lazy
#from django.views.generic import CreateView

from django.shortcuts import render, redirect, reverse
from django.http import HttpResponse, HttpResponseNotAllowed


#class SignUpView(CreateView):
#    form_class = UserCreationForm
#    success_url = reverse_lazy("login")
#    template_name = "registration/signup.html"


def signup(request):
    if request.method == 'GET':
        return render(request, "registration/signup.html", {})
    elif request.method == 'POST':
        user_form = UserCreationForm(request.POST)
        if user_form.is_valid():
            user_form.save()
            url = reverse('login')
            return redirect(url)
            #return render(request, "registration/login.html", {})
            #return render(request, "automodeler/index.html", {})
        else:
            errors_dict = {}
            for field, errors in user_form.errors.items():
                errs = []
                for error in errors:
                    errs.append(error)
                    print(f"Error in field '{field}': {error}")
                errors_dict[field] = errs
            return HttpResponse(render(request, 'registration/signup.html', {'form_errors': errors_dict}).content, content_type='text/html', status=409)
            #return render(request, "registration/signup.html", { 'form_errors': errors_dict})
    else:
        return HttpResponseNotAllowed()