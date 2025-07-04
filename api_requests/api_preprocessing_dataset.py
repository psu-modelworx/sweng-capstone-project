import requests

def api_preprocess_dataset():
    '''
    Making an API request to preprocess a dataset.
    The user can know the dataser_id by using the api_request_dataset.
    '''
    # Getting the URL and token to be authorized for the API request.
    url = "http://localhost:8000/automodeler/ppe/start_preprocessing_request/"
    token = "ad5c2121066a722d162ce9864ea99e7902371009"

    # Setting the dataset_id which was collected from using api_request_dataset.
    dataset_id = 4

    # Set up a header with the authorization token to authorize the user.
    headers = { "Authorization": "Token " + token}

    # Added the dataset_id to the data collection to send in the request.
    data = {"dataset_id": dataset_id}

    # Getting the response from the API request and printing its text to know if it was complete.
    response = requests.post(url=url, headers=headers, data=data)
    print(response.text)

# The main method calls the function to preprocess the dataset.
if __name__ == "__main__":    
    api_preprocess_dataset()