
# classifier  exposed as a serverless AWS Lambda function




## Design
We deploy a Lambda function using a custom docker image build from public.ecr.aws/lambda/python:3.9.

The docker image contains python library, sentence transformer fine-tuned model and the lambda function handler.

**Service signature**
The service accepts a map of IDs and sentences
```json
{
    "0x01":"some text",
    "0x02":"other sentence
}
```
and returns a map of IDs and vectors.
```json
{ 
    "id1": {
        "label": "NOT-SPAM", 
        "score": 0.9827121496200562, 
        "probabilities": {
            "NOT-SPAM": 0.9827121496200562, 
            "SPAM": 0.017287809401750565}
        }, 
    "id2": ...
}
>
```


Client should use the IDs to associate the result with the right objects UIDs


## Usage
Some steps are still manual in this version but can be automated.


Following https://www.philschmid.de/serverless-bert-with-huggingface-aws-lambda-docker
### Set your AWS CLI 
have aws cli installed and a profile [dgraph] in ~/.aws/credential with aws_access_key_id and aws_secret_access_key.

### Create ECR repository
export aws_region=us-east-1
export aws_account_id=<account_id>

> aws ecr create-repository --repository-name classifier-lambda-arm64 --region $aws_region --profile sandbox
### build and tag the docker image

> docker build  --platform linux/arm64  --no-cache -f arm64-dockerfile -t aws-classifier-arm64 .
use ECR repositoryUri to tag the image

> docker tag aws-classifier-arm64 $aws_account_id.dkr.ecr.us-east-1.amazonaws.com/classifier-lambda-arm64
### push the image to ECR
> aws ecr get-login-password --region $aws_region \
| docker login \
    --username AWS \
    --password-stdin $aws_account_id.dkr.ecr.us-east-1.amazonaws.com/classifier-lambda-arm64
Login Succeeded

> docker push $aws_account_id.dkr.ecr.us-east-1.amazonaws.com/classifier-lambda-arm64
### deploy the serverless function
set the image reference in serverless.yml config.

Deploy to AWS
```
$ serverless deploy --aws-profile dgraph
```

If we build an arm64 docker image (on Mac without forcing the platform), then the lambda must be configured with
- architecture: arm64
else, if we build the docker image for platform linux/amd64, then set
- architecture: x86_64

_Note_: In current form, after deployment, your API is protected by an API key. 

For production deployments, you might want to configure an authorizer.

### Invocation
```
curl --request POST \

--url https://<>.execute-api.us-east-1.amazonaws.com/dev/embedding \
--header 'Content-Type: application/json' --header 'x-api-key: <apikey>' \
--data '{"id":"some sample text"}'
```


### Local development





docker run -d --name classifier -p 8180:8080  -v ./:/var/task/   aws-classifier-arm64

Runing locally a specific handler:

> docker run -d --name classifier -p 8180:8080  -v ./handler.py:/var/task/handler.py   classifier-lambda-arm64 "handler.embedding"

```
curl --request POST \
--url http://localhost:8180/2015-03-31/functions/function/invocations \
--header 'Content-Type: application/json' \
--data '{"body":"{\"id1\":\"some sample text\",\"id2\":\"some other text\"\n}"}'
```

```
curl --request POST \\n--url http://localhost:8180/2015-03-31/functions/function/invocations \\n--header 'Content-Type: application/json' \\n--data '{"body":"{\"type\":\"Company\",\"grpc_endpoint\":\"52.44.246.66:9080\",\"query\":\"about:Company.about\",\"prompt\":\"{{about}}\",\"predicate\":\"Company.embedding\"}"}'
```

Note that the payload has a “body” element as a string.


Docker tests
1 - on mac create but with amd64 target - working but slow.
> cd amd64
> docker build -f amd64-dockerfile --platform linux/amd64  -t python-aws-amd64 .

requirements.txt
    urllib3==1.26.*
    transformers==4.33.1
    sentence-transformers==2.2.2
    pydgraph==23.0.1
    pybars3
lead to an image of 8.8Gb
if requirements.txt contains
https://download.pytorch.org/whl/cpu/torch-2.1.0%2Bcpu-cp311-cp311-linux_x86_64.whl

the image is 2.37GB

> docker run -d --name transformer-amd64 -p 8180:8080  -v ./:/var/task/   python-aws-amd64

2- build arm64 image
> docker build  --platform linux/arm64  -t python-aws-arm64 .
with requirements
    https://download.pytorch.org/whl/cpu/torch-2.1.0-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl#sha256=8132efb782cd181cc2dcca5e58effbe4217cdb2581206ac71466d535bf778867
    transformers==4.33.1
    sentence-transformers==2.2.2
    pydgraph==23.0.1
    pybars3
-> not working

    
What is WORKING deployed?

    dev-python-embedding
    lambda  named :python-embedding-dev-sentence-encoding
    Created using the ECR image.
