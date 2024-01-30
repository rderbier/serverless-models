import os
import io
import boto3
import json
import base64
# Invoke the SageMaker endpoint

ENDPOINT_NAME = "huggingface-pytorch-inference-2023-12-11-22-58-20-578"
# os.environ['ENDPOINT_NAME']
runtime= boto3.client('runtime.sagemaker')


image_path = "/Users/raphaelderbier/Downloads/image1.jpeg"

with open(image_path, "rb") as data_file:
    image_data = data_file.read()
    # image in binary format
    
    # Invoke the SageMaker endpoint
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        Body=image_data,
        ContentType='image/x-image')

    # Parse the JSON response from the SageMaker endpoint
    result = json.loads(response['Body'].read().decode())
    print(result)
        
    