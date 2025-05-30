from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse
from django.shortcuts import redirect

from .models import Dataset
from .forms import DatasetForm

# Create your views here.

def index(request):
    if request.user.is_authenticated:
        auth_user = request.user
        #user_datasets = Dataset.objects.filter(auth_user == user)
        user_datasets = Dataset.objects.all()
        return render(request, "automodeler/index.html", {})
    else:
        url = reverse("login")
        return HttpResponseRedirect(url)


def upload(request):
    if request.user.is_authenticated:
        if request.method == 'POST':
            # Create form object with data from POST request
            form = DatasetForm(request.POST)
            if form.is_valid():
                print("valid form")
            return render(request, "automodeler/success.html", {})
        else:
            form = DatasetForm()
            return render(request, "automodeler/upload.html", {"form": form})
    else:
        url = reverse("login")
        return HttpResponseRedirect(url)

#def validate_dataset(request):


