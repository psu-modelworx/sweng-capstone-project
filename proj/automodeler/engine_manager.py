from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import get_object_or_404
from django.conf import settings

from .models import Dataset
from engines.preprocessing_engine import PreprocessingEngine

import pandas as pd

# self, df, target_column, categorical_columns=None, columns_to_remove=None, 
# test_size=DEFAULT_TEST_SIZE, random_state=DEFAULT_RANDOM_STATE
def start(request, dataset_id):
    dataset = Dataset.objects.get(id = dataset_id)
    #media_folder = settings.MEDIA_ROOT
    df = pd.read_csv(dataset.csv_file)
    target_column = dataset.target_feature
    all_features_dict = dataset.features
    categorical_columns = [f for f in dataset.features if all_features_dict[f] == 'C']
    ppe = PreprocessingEngine(df=df, target_column=target_column, categorical_columns=categorical_columns)
    ppe.run_preprocessing_engine()
    return HttpResponse("Preprocessing started...")

