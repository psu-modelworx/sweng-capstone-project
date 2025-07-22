from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseNotFound, HttpResponseServerError
from django.urls import reverse
from django.shortcuts import redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib.auth import logout
from django.contrib import messages
from django.core.exceptions import ObjectDoesNotExist
from rest_framework.authtoken.models import Token


from .models import Dataset, PreprocessedDataSet, DatasetModel, TunedDatasetModel, UserTask
from .forms import DatasetForm
from . import helper_functions

import logging
logger = logging.getLogger('django_file')


# Create your views here.

def index(request):
    """
    index is the default index.html page for the automodeler application.  If a user is not authenticated, they are
    redirected to the login page. It displays datasets that users have uploaded.  Only Datasets for current user.

    :param request: This is the HTTP request object containing the HTTP request information
    """
    logger.info('Test log message!')
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
                #file_size = csv_file.size / 1073741824 # This will convert to Gigabytes
                file_size = csv_file.size
                user_id = request.user.id

                number_of_rows = 0
                try:
                    number_of_rows = helper_functions.sanitize_dataset(csv_file)
                except Exception as e:
                    url = reverse('upload')
                    print("Exception: {0}".format(e))
                    return render(request, url, {"form": form, "err_msg": "CSV File failed sanitation!"})
                features = helper_functions.extract_features_from_inMemoryUploadedFile(csv_file)

                dataset_model = Dataset.objects.create(
                  name=file_name, 
                  features=features, 
                  csv_file=csv_file,
                  file_size=file_size,
                  number_of_rows=number_of_rows,
                  user_id=user_id
                  )
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
            url = reverse('dataset_details', kwargs={'dataset_id': dataset_id})
            return redirect(url)
            #return render(request, "automodeler/index.html", {})
        else:
            return render(request, "automodeler/dataset.html", {"dataset": dataset})
    else:
        return redirect(reverse('login'))

@login_required
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
def account_delete(request):
    if request.method == 'POST':
        user = request.user
        logout (request)
        user.delete()
        messages.success(request, "Your account has been deleted.")
        return redirect('index')

    return render(request, 'automodeler/account_delete_confirm.html')
    

@login_required
def dataset_collection(request):
    auth_user = request.user
    user_datasets = Dataset.objects.filter(user = auth_user)
    pp_datasets = {}
    for uds in user_datasets:
        try:
            pp_datasets[uds.filename] = PreprocessedDataSet.objects.get(original_dataset_id = uds.id)
        except ObjectDoesNotExist:
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
def dataset_details(request, dataset_id):
    ds_details = {}
    
    user = request.user
    dataset = get_object_or_404(Dataset, pk=dataset_id, user=user)
    ds = {}
    ds["ds_id"] = dataset.id
    ds["name"] = dataset.name
    ds["target_feature"] = dataset.target_feature
    ds["file_size"] = helper_functions.file_size_for_humans(dataset.file_size)
    ds["number_of_rows"] = dataset.number_of_rows
    ds["features"] = dataset.features
    ds_details["ds"] = ds

    try:
        pp_dataset = PreprocessedDataSet.objects.get(original_dataset=dataset)
        pp_ds = {}
        pp_ds['number_of_rows'] = pp_dataset.number_of_rows
        pp_ds['number_of_removed_rows'] = pp_dataset.number_of_removed_rows
        pp_ds['file_size'] = helper_functions.file_size_for_humans(pp_dataset.file_size)
        pp_ds['removed_features'] = pp_dataset.removed_features
        pp_ds['new_target_feature'] = pp_dataset.meta_data['target_column']
        pp_ds['task_type'] = pp_dataset.meta_data['task_type']
        ds_details["pp_ds"] = pp_ds
    except Exception as e:
        print("Exception e: {0}".format(e))
    
    try:
        ds_models = DatasetModel.objects.filter(original_dataset=dataset, tuned=True)
        md = {}
        ds_details["models"] = ds_models
    except Exception as e:
        print("Exception e: {0}".format(e))

    return render(request, "automodeler/dataset_details.html", { "ds_details": ds_details })


@login_required
def model_collection(request):
    auth_user = request.user
    user_models = DatasetModel.objects.filter(user=auth_user, tuned=True)
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
def model_details(request, model_id):
    try:
        dataset_model = DatasetModel.objects.get(id=model_id, tuned=True)
    except Exception as e:
        print("Exception {0}".format(e))
        msg = "Error retrieving model by:  id={0}".format(model_id)
        print(msg)
        return HttpResponseNotFound(msg)
    
    try:
        dataset = Dataset.objects.get(id=dataset_model.original_dataset.id)
        ds_features = list(dataset.features.keys())
    except Exception as e:
        msg = "Error getting features.  Exception: {0}".format(e)
        print(msg)
        return HttpResponseServerError(msg)
    
    model_details = {
        "id": dataset_model.id,
        "name": dataset_model.name,
        "method": dataset_model.model_method,
        "type": dataset_model.model_type,
        "features": ds_features
    }
    return render(request, "automodeler/model_details.html", { "model": model_details })

@login_required
def task_collection(request):
    #user_tasks = UserTask.objects.filter(user=request.user).order_by('-created_at')
    #return render(request, "automodeler/task_collection.html", {"user_tasks": user_tasks})
    user_tasks = UserTask.objects.filter(user=request.user).select_related('dataset').order_by('-created_at')
    return render(request, 'automodeler/task_collection.html', {'user_tasks': user_tasks})
