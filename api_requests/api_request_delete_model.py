import requests

def api_request_delete_model():
    '''
    The API request to delete a model that are saved to a user.
    A user needs to send the model ID which is retreived from an API model request.
    '''
    # Setting up the API request delete model URL enpoint and the token to authenticate a user.
    url = "http://localhost:8000/automodeler/model_delete/"
    token = "bc35f1ed98772f18c17bb71484e285673d70c4da"

    # Adding the authentication token to the header to authenticate a user.
    headers = { "Authorization": "Token " + token}
    
    # Defining the model ID which will need to be updated with each request.
    model_id = 15

    # Adding the modle ID to the data so it can be sent in a request.
    data = {"model_id": model_id}

    # Getting the response from the request to ensure it was successful.
    response = requests.post(url=url, headers=headers, data=data)
    print(response.status_code)

# Is used to call the function to make an API request an delete a model.
if __name__ == "__main__":    
    api_request_delete_model()