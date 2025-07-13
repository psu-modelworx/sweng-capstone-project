import requests

def api_request_models():
    '''
    The API request for all models a user has.
    '''
    # Assigning the URL and token needed to make the API request.
    url = "http://localhost:8000/automodeler/api/request_models"
    token = "ad5c2121066a722d162ce9864ea99e7902371009"

    # Assigning the token to the header to authenticate the user.
    headers = { "Authorization": "Token " + token}

    # Making a request with the URL and header with the authentication token.
    response = requests.post(url=url, headers=headers)

    # Making an empty list of results which will contain the model information.
    results = ""

    # Looping through the response and printing the model data.
    for item in response.json():
        results += item + "\n\n"

    # Printing the results in an easy to read format.
    print(results)

# The main method is used to run the API request for model data.
if __name__ == "__main__":    
    api_request_models()