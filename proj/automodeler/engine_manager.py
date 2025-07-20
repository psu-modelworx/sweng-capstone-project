from django.contrib.auth.decorators import login_required

from rest_framework.authentication import TokenAuthentication, SessionAuthentication
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.permissions import IsAuthenticated

from django.http import JsonResponse
from celery.result import AsyncResult

from .models import TunedDatasetModel
from .models import UserTask

from .tasks import start_preprocessing_task, start_modeling_task, run_model_task

import json


@api_view(["POST"])
@authentication_classes([SessionAuthentication, TokenAuthentication])
@permission_classes([IsAuthenticated])
def start_preprocessing_request(request):
    print("Starting preprocessing")
    if request.method != 'POST':
        print("Error: Non-POST Request received!")
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    
    # Verify dataset exists
    dataset_id = request.POST.get('dataset_id')
    if not dataset_id:
        print("Error: Missing dataset_id in form!")
        return JsonResponse({'error': 'Missing value: dataset_id'}, status=400)
    
    #print("Dataset ID: " + str(dataset_id))

    # Launch celery task async
    async_results = start_preprocessing_task.apply_async(args=[dataset_id, request.user.id])
    # Create UserTask
    user_task = UserTask.objects.create(
        user=request.user,
        task_id=async_results.id,
        task_type='preprocessing', 
        status='PENDING',
        dataset_id=dataset_id
    )   

    user_task.save() # Saving user task to DB as rff is complaining about it not being used
  
    return JsonResponse({"task_id": async_results.id})

@api_view(["POST"])
@authentication_classes([SessionAuthentication, TokenAuthentication])
@permission_classes([IsAuthenticated])
def start_modeling_request(request):
    print("Starting modeling")
    if request.method != 'POST':
        print("Error: Non-POST Request received!")
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    
    # Verify dataset exists
    dataset_id = request.POST.get('dataset_id')
    if not dataset_id:
        print("Error: Missing dataset_id in form!")
        return JsonResponse({'error': 'Missing value: dataset_id'}, status=400)


    # Launch celery task async
    async_results = start_modeling_task.apply_async(args=[dataset_id, request.user.id])
    # Create UserTask
    user_task = UserTask.objects.create(
        user=request.user, 
        task_id=async_results.id,
        task_type='modeling', 
        status='PENDING',
        dataset_id=dataset_id
    )

    user_task.save() # Saving usertask to DB as ruff said it wasn't being used
    
    return JsonResponse({"task id": async_results.id})

@api_view(["POST"])
@authentication_classes([SessionAuthentication, TokenAuthentication])
@permission_classes([IsAuthenticated])
def run_model(request):
    print("Starting model run")
    if request.method != 'POST':
        print("Error: Non-POST Request received!")
        return JsonResponse({'error': 'Method not allowed'}, status=405)

    # Verify model exists
    model_id = request.POST.get('model_id')
    if not model_id:
        print("Error: Missing model_id in form!")
        return JsonResponse({'error': 'Missing value: model_id'}, status=400)

    # Get dataset_id from model FK before launching the task
    try:
        tuned_model = TunedDatasetModel.objects.get(id=model_id, user=request.user)
        dataset_id = tuned_model.original_dataset_id
    except TunedDatasetModel.DoesNotExist:
        return JsonResponse({"error": "Tuned model not found"}, status=404)


     # Get the data from the post request
    data = request.POST.get('data')
    if not data:
        msg = 'Error:  missing data field'
        print(msg)
        return JsonResponse({'error': 'Missing value: data'}, status=400)
    data_dict = json.loads(data)

    # enqueue the celery task first without passing task_id
    async_result =  run_model_task.apply_async(args=[model_id, request.user.id, data_dict])
  
    # Create UserTask
    user_task = UserTask.objects.create(
        user=request.user, 
        task_id=async_result.id,  # task_id celery assigned
        task_type='prediction', 
        status='PENDING',
        dataset_id=dataset_id)

    user_task.save() # Again, saving the task because of ruff
    
    # Return JSON with the task ID so frontend can poll
    return JsonResponse({'task_id': async_result.id})
    


@api_view(["GET", "POST"])
@authentication_classes([SessionAuthentication, TokenAuthentication])
@permission_classes([IsAuthenticated])
def check_task_result(request, task_id):
    result = AsyncResult(task_id)
    if result.ready():
        data = result.result
        return JsonResponse(data)
    return JsonResponse({'status': 'PENDING'})


