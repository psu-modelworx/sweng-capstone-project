import requests

def api_upload_request():
    '''
    This function makes an API request to upload a dataset if the user is authenticated.
    The csv file and name are expected to be changed with each upload.
    A response will be printed to let the user know if an upload was successful.
    '''
    # The url is where the API request is made and the token authenticates the user.
    url = "http://localhost:8000/automodeler/api/upload"
    token = "6aa53ad83773ba8a42834ff9317933759ed0e86b"

    # Defining the path for the csv file and name of the dataset.
    csv_file = "C:\\Users\\jpdun\\Desktop\\personality_dataset.csv"
    name = "API Upload Request"

    # Making variables for the csv file, name, and token so they can be passed in the response.
    files = {"csv_file": open(csv_file, "rb")}
    data = {"name": name}
    headers = { "Authorization": "Token " + token}
            
    # Getting the response after the API request is made and printing it to know if the upload was successful.
    response = requests.post(url=url, headers=headers, files=files, data=data)
    print(response.text)

# The main method which is used to run the application.
if __name__ == "__main__":    
    api_upload_request()