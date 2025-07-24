import requests

def api_check_task_result():
    '''
    The API request to get the task results after running a model.
    '''
    # Set up the URL and authentication token for the API request.
    # The URL must contain the task ID.
    url = "http://localhost:8000/automodeler/check_task/520e7549-e1ae-4854-9244-4d29c5c71d50/"
    token = "f0c991d35bdcadea6ba41a3131fb8e3da77b292f"

    # Added the authentication token to the header to authenticate the user.
    headers = { "Authorization": "Token " + token}

    # Getting the response after making the API request for task results and printing them.
    response = requests.post(url=url, headers=headers)
    print(response.text)

# The main method which is used to run the function for the API request.
if __name__ == "__main__":    
    api_check_task_result()