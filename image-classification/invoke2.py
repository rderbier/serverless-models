import os
import io
import boto3
import json
import base64
from sagemaker.huggingface import HuggingFacePredictor

# grab environment variables
ENDPOINT_NAME = "huggingface-pytorch-inference-2023-12-11-22-58-20-578"
# os.environ['ENDPOINT_NAME']
runtime= boto3.client('runtime.sagemaker')


image_path = "/Users/raphaelderbier/Downloads/image1.jpeg"

with open(image_path, "rb") as data_file:
    image_data = data_file.read()
        
    
predictor=HuggingFacePredictor(endpoint_name="huggingface-pytorch-inference-2023-10-05-22-50-22-808")
data = {
  "inputs": "the mesmerizing performances of the leads keep the film grounded and keep the audience riveted .",
}

import os
output = predictor.predict(data=data)
print(output)
    #print(response)

    # Parse the JSON response from the SageMaker endpoint
    result = json.loads(response['Body'].read().decode())
    print(result)
        
    