from sagemaker.huggingface import HuggingFace
from sagemaker.serverless import ServerlessInferenceConfig


role = "arn:aws:iam::262269367073:role/AmazonSageMaker-ExecutionRole"



# hyperparameters which are passed to the training job
hyperparameters={
                 'model_name': 'distilbert-base-uncased',
                 'epochs': 1,
                 'train_batch_size': 32,
                 }

# create the Estimator
huggingface_estimator = HuggingFace(
        entry_point='train.py',
        source_dir='./scripts',
        instance_type= 'ml.p3.2xlarge', # 'local-gpu''ml.p3.2xlarge'
        instance_count=1,
        role=role,
        transformers_version='4.26', #4.6 pytorch1.6 py36
        pytorch_version='1.13',
        py_version='py39',
        hyperparameters = hyperparameters
)
huggingface_estimator.fit(
  {'train': 's3://sagemaker-marketing-classifier-001/train',
   'test': 's3://sagemaker-marketing-classifier-001/eval'}
)


serverless_config = ServerlessInferenceConfig(memory_size_in_mb=3072, max_concurrency=1)
predictor = huggingface_estimator.deploy(serverless_inference_config=serverless_config)
print("endpoint_name: {}".format(predictor.endpoint_name))