import sagemaker
from sagemaker.serializers import DataSerializer
from sagemaker.huggingface.model import HuggingFaceModel

import boto3
sess = sagemaker.Session()
# sagemaker session bucket -> used for uploading data, models and logs
# sagemaker will automatically create this bucket if it not exists
sagemaker_session_bucket=None
if sagemaker_session_bucket is None and sess is not None:
    # set to default bucket if a bucket name is not given
    sagemaker_session_bucket = sess.default_bucket()
iam = boto3.client('iam')
role = iam.get_role(RoleName='AmazonSageMaker-ExecutionRole')['Role']['Arn']


sess = sagemaker.Session(default_bucket=sagemaker_session_bucket)

print(f"sagemaker role arn: {role}")
print(f"sagemaker bucket: {sess.default_bucket()}")
print(f"sagemaker session region: {sess.boto_region_name}")

from sagemaker.huggingface import HuggingFaceModel

# Hub Model configuration. https://huggingface.co/models
hub = {
  'HF_MODEL_ID':'google/vit-base-patch16-224', # model_id from hf.co/models
  'HF_TASK':'image-classification' # NLP task you want to use for predictions
}

# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
   env=hub,
   role=role, # iam role with permissions to create an Endpoint
   transformers_version="4.26", # transformers version used
   pytorch_version="1.13", # pytorch version used
   py_version="py39", # python version of the DLC
)

# create a serializer for the data
image_serializer = DataSerializer(content_type='image/x-image') # using x-image to support multiple image formats

# deploy model to SageMaker Inference
predictor = huggingface_model.deploy(
	initial_instance_count=1, # number of instances
	instance_type='ml.g4dn.xlarge', # ec2 instance type
  serializer=image_serializer, # serializer for our audio data.
)