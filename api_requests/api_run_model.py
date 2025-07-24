import json
import requests

def api_run_model():
    '''
    An API request to run a model with input data.
    '''
    # Set up the URL and authentication token for the API request.
    url = "http://localhost:8000/automodeler/mod/run_model/"
    token = "f0c991d35bdcadea6ba41a3131fb8e3da77b292f"

    # The model ID can be found by running the API to request models.
    model_id = 15

    # The header contains the authorization token and authenticates a user's request.
    headers = { "Authorization": "Token " + token}

    # Set up the input values and added them to the request data with the key values, so they can be accessed.
    input_data = [0, 0, 0, 0, 0, 0, 0, 0]
    request_data = {"values": input_data}

    # The data contains the model ID and input data to run the model.
    data = {"model_id": model_id, "data": json.dumps(request_data)}

    # Making the request and getting the response which gives the user the task ID.
    response = requests.post(url=url, headers=headers, data=data)
    print(response.text)

# The main method to run the API request to run the model.
if __name__ == "__main__":    
    api_run_model()