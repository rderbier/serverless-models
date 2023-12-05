import sagemaker

# NOT WORKING  - 

role = "arn:aws:iam::262269367073:role/AmazonSageMaker-ExecutionRole"

print(f"sagemaker role arn: {role}")

from sagemaker.huggingface import HuggingFace

# create Hugging Face Model Class
huggingface_model = HuggingFace(
   model_data="s3://sagemaker-us-east-1-262269367073/huggingface-pytorch-training-2023-11-20-06-22-53-382/output/model.tar.gz",  # path to your trained sagemaker model
   entry_point='train.py',
   source_dir='./scripts',
   role=role, # iam role with permissions to create an Endpoint
   transformers_version="4.26", # transformers version used
   pytorch_version="1.13", # pytorch version used
   py_version="py39", # python version of the DLC
   instance_count=1,
   instance_type="local_gpu"
)
huggingface_model.fit(
  {'train': 's3://sagemaker-marketing-classifier-001/train',
   'test': 's3://sagemaker-marketing-classifier-001/eval'}
)
# deploy model to SageMaker Inference
predictor = huggingface_model.deploy(

)
# example request, you always need to define "inputs"
data = {
   "inputs": "The new Hugging Face SageMaker DLC makes it super easy to deploy models in production. I love it!"
}

# request
prediction = predictor.predict(data)
print(prediction)
