from sagemaker.huggingface import HuggingFace
from sagemaker.serverless import ServerlessInferenceConfig
import os
import sagemaker
import boto3
iam = boto3.client('iam')
#sess = sagemaker.Session()
#role = sagemaker.get_execution_role()
#print(f"sagemaker role arn: {role}")


role = iam.get_role(RoleName='AmazonSageMaker-ExecutionRole')['Role']['Arn']

print(f"role: {role}")

from sagemaker.huggingface import get_huggingface_llm_image_uri

# retrieve the llm image uri
llm_image = get_huggingface_llm_image_uri(
  backend="huggingface",
  region="us-west-2"
)

# print ecr image uri
print(f"llm image uri: {llm_image}")


import json
from sagemaker.huggingface import HuggingFaceModel

# sagemaker config
instance_type = "ml.g4dn.2xlarge"
number_of_gpu = 1
health_check_timeout = 300

# TGI config
config = {
  'HF_MODEL_ID': "tiiuae/falcon-40b-instruct", # model_id from hf.co/models
  'SM_NUM_GPUS': json.dumps(number_of_gpu), # Number of GPU used per replica
  'MAX_INPUT_LENGTH': json.dumps(1024),  # Max length of input text
  'MAX_TOTAL_TOKENS': json.dumps(2048),  # Max length of the generation (including input text)
  # 'HF_MODEL_QUANTIZE': "bitsandbytes", # comment in to quantize
}

# create HuggingFaceModel
llm_model = HuggingFaceModel(
  role=role,
  image_uri=llm_image,
  env=config
)