import requests

def api_dataset_report_download():
    '''
    This is an example API request to download a dataset report to a location set by the user.
    The user will need to update the URL with the dataset ID and set the location to download the file.
    A response will say if the file was successfully downloaded or now.
    '''
    # The URL contains the dataset ID which can be collected from the API request for all datasets.
    url = "http://localhost:8000/automodeler/report_download/3"

    # The token is from the user's Account page and added to the header to authenticate the user.
    token = "f0c991d35bdcadea6ba41a3131fb8e3da77b292f" 
    headers = { "Authorization": "Token " + token}

    # Making the API request and getting the response with the dataset report.
    response = requests.post(url=url, headers=headers)

    # Ensuring the dataset report was collected before trying to save it to a location set by the user.
    if (response.status_code == 200):
        # Downloaded the content in the dataset report.
        with open("C:\\Users\\jpdun\\Desktop\\dataset_report.pdf", "wb") as dataset_report:
            dataset_report.write(response.content)

        # Letting the user know that the download was successful
        print("The dataset report was downloaded successfully.")
    else:
        # Letting the user know that the report was not downloaded.
        print("The dataset report was not dwnloaded successfully.")

# The main method which is used to call the dataset report function.
if __name__ == "__main__":    
    api_dataset_report_download()