
from sagemaker.huggingface import HuggingFaceModel
from sagemaker.serverless import ServerlessInferenceConfig

huggingface_model = HuggingFaceModel(
    model_data="s3://hm-sentence-transformers/model.tar.gz",
    transformers_version='4.6',
    pytorch_version='1.7',
    py_version="py36",
    role="arn:aws:iam::262269367073:role/AmazonSageMaker-ExecutionRole")

serverless_config = ServerlessInferenceConfig(memory_size_in_mb=3072, max_concurrency=5)
huggingface_model.deploy(serverless_inference_config=serverless_config)
print("endpoint_name: {}".format(huggingface_model.endpoint_name))