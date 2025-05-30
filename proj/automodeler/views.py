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
            form = DatasetForm(request.POST, request.FILES)
            #print(request.POST)
            #print(request.FILES)
            if form.is_valid():
                print("valid form")
                url = reverse("parameters")
                col_headers = parameter_prep(request.FILES['csv_file'])
                print(request.FILES['csv_file'])
                return render(request, "automodeler/parameters.html", {'col_headers': col_headers}) # Get parameter information and send back
            else:
                print("form invalid!")
                print(form.errors)
                form = DatasetForm()
                return render(request, "automodeler/upload.html", {"form": form}) # create context for error messages and send back here
        else:
            form = DatasetForm()
            return render(request, "automodeler/upload.html", {"form": form})
    else:
        url = reverse("login")
        return HttpResponseRedirect(url)


def parameter_selection(request):
    #print(request.META)
    return render(request, "automodeler/parameters.html", {})

def parameter_prep(dataset_file):
    #print(dataset_file)
    #print(dataset_file['csv_file'])
    # Converts from InMemoryFile object to a list
    file_data = dataset_file.read().decode('utf-8').split()
    col_headers = file_data[0].split(',')
    print(col_headers)
    #print(file_data[0])

    # Works if it's a file
    #with open(dataset_file, newline='') as f:
    #    reader = csv.reader(f)
    #    row1 = next(reader)
    #    print(row1)
    return col_headers

#def validate_dataset(request):


