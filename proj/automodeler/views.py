from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse
from django.shortcuts import redirect, get_object_or_404

from .models import Dataset
from .forms import DatasetForm

import csv

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
            form = DatasetForm(request.POST, request.FILES)
            if form.is_valid():
                print("valid form")
                file_name = request.POST.get('name')
                csv_file = request.FILES['csv_file']
                csv_file_name = csv_file
                user_id = request.user.id
                dataset_model = Dataset.objects.create(name=file_name, csv_file=csv_file, user_id=user_id)
                dataset_model.save()
                dataset_model_id = dataset_model.id
                url = reverse('dataset', kwargs={'dataset_id': dataset_model_id})
                return HttpResponseRedirect(url)
            else:
                print("form invalid!")
                print(form.errors)
                form = DatasetForm()
                return render(request, "automodeler/upload1.html", {"form": form}) # create context for error messages and send back here
        else:
            form = DatasetForm()
            return render(request, "automodeler/upload1.html", {"form": form})
    else:
        url = reverse("login")
        return HttpResponseRedirect(url)

def dataset(request, dataset_id):
    if request.user.is_authenticated:
        dataset = get_object_or_404(Dataset, pk=dataset_id)
        if request.user.id != dataset.user_id:
            return redirect(reverse("index"))
        if request.method == 'POST':
            return render(request, "automodeler/dataset.html", {"dataset": dataset})
        else:
            dataset_fileName = dataset.csv_file.file.name
            dataset_features  = get_column_headers(dataset_fileName)
            return render(request, "automodeler/dataset.html", {"dataset": dataset, "dataset_features": dataset_features})
    else:
        return redirect(reverse('login'))


def get_column_headers(dataset_fileName):
    features = []
    with open(dataset_fileName, 'r') as file:
        csvFileReader = csv.reader(file)
        features = next(csvFileReader)
    print(features)
    return features

#def validate_dataset(request):

