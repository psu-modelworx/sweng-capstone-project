import requests

def api_request_delete_dataset():
    '''
    This is an API request to delete an uploaded dataset.
    The dataset ID must be included in the URL when making the request.
    A user can get the dataset ID by making an API request for all datasets.
    '''
    # Defining the URL to delete a dataset and the token to authenticate
    url = "http://localhost:8000/automodeler/dataset_delete/2"
    token = "bc35f1ed98772f18c17bb71484e285673d70c4da"

    # Adding the authentication token to the header to validate the user.
    headers = { "Authorization": "Token " + token}

    # Making the API request and printing the status code to ensure it was successful.
    response = requests.post(url=url, headers=headers)
    print(response.status_code)

# The main method is used to run the API request to delete a dataset.
if __name__ == "__main__":    
    api_request_delete_dataset()