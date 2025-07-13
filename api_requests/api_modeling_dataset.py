import requests

def api_modeling_dataset():
    '''
    Make an API request to model a dataset.
    The dataset ID can be found by using api_request_dataset.
    '''
    # The URL points to the API request and token authenticates the user.
    url = "http://localhost:8000/automodeler/mod/start_modeling_request/"
    token = "ad5c2121066a722d162ce9864ea99e7902371009"

    # Setting the dataset_id which was collected from api_request_dataset.
    dataset_id = 4

    # The header contains the authorization token to authenticate the user.
    headers = { "Authorization": "Token " + token}

    # Adding the dataset_id to the data collection so it can be used.
    data = {"dataset_id": dataset_id}

    # Making a request to model the data and printing the response to know it was complete.
    response = requests.post(url=url, headers=headers, data=data)
    print(response.text)

# The main method calls the function to model the dataset.
if __name__ == "__main__":    
    api_modeling_dataset()