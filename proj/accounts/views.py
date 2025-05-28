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
        user_form = UserCreationForm(request.POST)
        username = user_form['username']
        if user_form.is_valid():
            #if not User.objects.filter(username=username).exists:
                #print("Valid form!")
            user_form.save()
            return render(request, "registration/login.html", {})
            #else:
            #    print("User exists!")
            #    return render(request, "registration/signup.html", {}) # Return errors here and display as error text
        else:
            #print(user_form)
            errors_dict = {}
            for field, errors in user_form.errors.items():
                errs = []
                for error in errors:
                    errs.append(error)
                    print(f"Error in field' {field}': {error}")
                errors_dict[field] = errs
            return render(request, "registration/signup.html", {'form_errors': errors_dict})
    else:
        return HttpResponseNotAllowed()