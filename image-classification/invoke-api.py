
import requests

# invoke API Gateway HTTP endpoint

# grab environment variables
ENDPOINT_NAME = "huggingface-pytorch-inference-2023-12-11-22-58-20-578"
# os.environ['ENDPOINT_NAME']



image_path = "/Users/raphaelderbier/Downloads/image1.jpeg"
api_url ="https://okowp0f64g.execute-api.us-east-1.amazonaws.com/dev/classification-labels"
with open(image_path, "rb") as data_file:
    image_data = data_file.read()
    response = requests.post(url=api_url,
                    data=image_data,
                    headers={'Content-Type': 'image/x-image', 'Accept':'application/json'})
    

    # Parse the JSON response from the SageMaker endpoint
    labels = response.json()
    print(labels[0]['label'])
        
    