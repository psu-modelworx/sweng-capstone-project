import requests

def api_request_datasets():
    '''
    Showing an example of making an API request for dataset information.
    '''
    # Setting up the URL and token to make the request.
    url = "http://localhost:8000/automodeler/api/request_datasets"
    token = "ad5c2121066a722d162ce9864ea99e7902371009"

    # The token needs to be in a header so the user can be authenticated.
    headers = { "Authorization": "Token " + token}

    # The response collects the data and status code from the request.
    response = requests.post(url=url, headers=headers)

    # Setting up a string with the results.
    results = ""

    # Looping through the response and making a list of results.
    for item in response.json():
        results += item + "\n\n"

    # The results contain the dataset name, dataset file name, and preprocessed dataset file name.
    print(results)

# The main method runs the function to request all dataset information.
if __name__ == "__main__":    
    api_request_datasets()