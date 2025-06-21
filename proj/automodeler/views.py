from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse
from django.shortcuts import redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from rest_framework.authtoken.models import Token

from .models import Dataset, PreprocessedDataSet, DatasetModel
from .forms import DatasetForm
from . import helper_functions

import os

# Create your views here.

def index(request):
    """
    index is the default index.html page for the automodeler application.  If a user is not authenticated, they are
    redirected to the login page. It displays datasets that users have uploaded.  Only Datasets for current user.

    :param request: This is the HTTP request object containing the HTTP request information
    """
    return render(request, "automodeler/index.html")

def upload(request):
    """
    upload is the page used to upload CSV Files.  This page requires authentication to access  The files are given a name, and the 
    Features of the datasets are extracted from the first row of the CSV file.  From here, a Dataset object is created and saved 
    to the database.  The file is stored under proj/media/dataset_uploads.  If the form is invalid users are returned to the upload 
    form.  If it is valid, they are redirected to the dataset's appropriate details page.  

    :param request: Django HTTP Request object with HTTP request information
    """
    if request.user.is_authenticated:
        if request.method == 'POST':
            # Create form object with data from POST request
            form = DatasetForm(request.POST, request.FILES)
            if form.is_valid():
                print("valid form")
                file_name = request.POST.get('name')
                csv_file = request.FILES['csv_file']
                user_id = request.user.id
                try:
                    helper_functions.sanitize_dataset(csv_file)
                except Exception as e:
                    url = reverse('upload')
                    return render(request, url, {"form": form, "err_msg": "CSV File failed sanitation!"})
                    #return HttpResponse("Error sanitizing file!")
                features = helper_functions.extract_features_from_inMemoryUploadedFile(csv_file)
                dataset_model = Dataset.objects.create(name=file_name, features=features, csv_file=csv_file, user_id=user_id)
                dataset_model.save()
                
                dataset_model_id = dataset_model.id
                url = reverse('dataset', kwargs={'dataset_id': dataset_model_id})
                return HttpResponseRedirect(url)
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

def dataset(request, dataset_id):
    """
    dataset is the details page of the requested dataset. Requires authentication.  POSTing to this page allows modification of
    Target Feature for the dataset as well as Type for each Feature.

    :param request: Django HTTP Request object containing request data
    :param dataset_id: Integer ID of the dataset to be retrieved
    """
    if request.user.is_authenticated:
        dataset = get_object_or_404(Dataset, pk=dataset_id)
        if request.user.id != dataset.user_id:
            return redirect(reverse("index"))
        if request.method == 'POST':
            inputFeatures = {}
            targetFeature = request.POST.get('target_radio')
            for f in dataset.features:
                f_val = 'nc_radio_{feature}'.format(feature=f)
                inputFeatures[f] = request.POST.get(f_val)
                
            dataset.features = inputFeatures
            dataset.target_feature = targetFeature
            dataset.save()
            url = reverse("dataset_collection")
            return redirect(url)
            #return render(request, "automodeler/index.html", {})
        else:
            return render(request, "automodeler/dataset.html", {"dataset": dataset})
    else:
        return redirect(reverse('login'))

def account(request):
    """
    Ensuring a user is logged in before giving them access to the account page.
    When the "Receive Token" button is pressed, give or assign a token to a user.
    Displaying a user's token on their account page so they know it was successfully created.
    """
    if request.user.is_authenticated:
        # Tokens are stored in the session so they can be displayed after the redirect.
        token = request.session.get("token", "Token: ")
        
        # Called when the "Receive Token" button is pressed.
        if request.method == 'POST':
            # Getting the path of this page to refresh it after getting the token.
            url = reverse("account")
            
            # Getting or creating a token and storing it in the session.
            userToken, tokenExists = Token.objects.get_or_create(user=request.user)
            request.session["token"] = "Token: " + userToken.key
            
            # Redirecting the url to refresh it and show the token.
            return redirect(url)
        else:
            # When navigating to the page, render the account html file.
            return render(request, "automodeler/account.html", {"token": token})
    else:
        # If a user isn't authenticated, navigate to the login page.
        url = reverse("login")
        return HttpResponseRedirect(url)

@login_required
def dataset_collection(request):
    auth_user = request.user
    user_datasets = Dataset.objects.filter(user = auth_user)
    pp_datasets = {}
    for uds in user_datasets:
        try:
            pp_datasets[uds.filename] = PreprocessedDataSet.objects.get(original_dataset_id = uds.id)
        except:
            print("No preprocessed datasets for " + str(uds.filename))
    
    combined_datasets = []
    for ds in user_datasets:
        tmp_list = []
        tmp_list.append(ds)
        for pp_ds in pp_datasets:
            if pp_datasets[pp_ds].original_dataset == ds:
                tmp_list.append(pp_datasets[pp_ds])
                break
        combined_datasets.append(tmp_list)
                        
    print("Found Datasets: " + str(pp_datasets))
    return render(request, "automodeler/dataset_collection.html", {'combined_datasets': combined_datasets})

@login_required
def dataset_delete(request, dataset_id):
    if request.method != 'POST':
        return HttpResponse("Invalid request method")
    else:
        dataset = Dataset.objects.get(id = dataset_id)
        dataset.delete()
        url = reverse("dataset_collection")
        return redirect(url)

@login_required
def model_collection(request):
    auth_user = request.user
    # user_models = ...
    user_models = DatasetModel.objects.filter(user=auth_user)
    return render(request, "automodeler/model_collection.html", {"models": user_models})

@login_required
def model_delete(request):
    if request.method != 'POST':
        return HttpResponse("Invalid request method")
    model_id = request.POST.get('model_id')
    if not model_id:
        return HttpResponse("Empty model id!")

    ds_model = DatasetModel.objects.get(id=model_id)
    ds_model.delete()
    url = reverse("model_collection")
    return redirect(url)

@login_required
def task_collection(request):
    auth_user = request.user
    # user_models = ...
    return render(request, "automodeler/task_collection.html")
