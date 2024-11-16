from google.cloud import vision
from google.oauth2 import service_account

def authenticate_vision_api():
    """
    Authenticate to Google Vision API using a service account JSON key file.

    Returns:
        google.oauth2.service_account.Credentials: The authenticated credentials for the API.
    """
    # Provide the path to the service account key file to authenticate.
    credentials = service_account.Credentials.from_service_account_file(
        r"C:\Users\mehri\Downloads\massive-petal-421620-b7e20c0e12f4.json"
    )
    return credentials

def get_image_labels(image_path):
    """
    Detects labels in a single image using Google Vision API.

    Args:
        image_path (str): The file path to the image for label detection.

    Returns:
        list of str: A list of label descriptions detected in the image.
    """
    # Authenticate and create a client for the Vision API.
    credentials = authenticate_vision_api()
    client = vision.ImageAnnotatorClient(credentials=credentials)
    # Read image data from a file path.
    with open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    # Perform label detection on the image.
    response = client.label_detection(image=image)
    labels = response.label_annotations
    # Extract and return the text description of each label.
    return [label.description for label in labels]

def label_images(image_paths):
    """
    Processes a list of image paths to detect labels for each image.

    Args:
        image_paths (list of str): A list of image file paths.

    Returns:
        dict: A dictionary mapping each image path to a list of detected labels.
    """
    labels_dict = {}
    # Loop through each image path, detect labels, and store the results in a dictionary.
    for image_path in image_paths:
        labels = get_image_labels(image_path)
        labels_dict[image_path] = labels
    return labels_dict

if __name__ == '__main__':
    # Define the paths to the images to be labeled.
    image_paths = ['C:/Users/mehri/Downloads/Indiv. Ass. - Resources/Data/zomato_809822_3059739671973416386_40704_396/2023-03-16_12-23-34_UTC_1.jpg']
    # Detect labels for each image and print the results.
    labels_dict = label_images(image_paths)
    for image_path, labels in labels_dict.items():
        print(f'Image: {image_path}, Labels: {labels}')
