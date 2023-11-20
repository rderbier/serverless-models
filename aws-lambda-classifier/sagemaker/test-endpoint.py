import boto3
import json
runtime = boto3.client("sagemaker-runtime")

endpoint_name = "huggingface-pytorch-training-2023-11-20-18-04-30-221"
content_type = "application/json"
data = {
  "inputs": [
    "XXXMobileMovieClub: To use your credit, click the WAP link in the next txt message or click here>> http://wap. xxxmobilemovieclub.com?n=QJKGIGHJJGCBL",
    "This is a normal text and should not be seen as spam!"
    ]
}
payload = json.dumps(data)
response = runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType=content_type,
    Body=payload
)
print(json.loads(response['Body'].read().decode("utf-8")))