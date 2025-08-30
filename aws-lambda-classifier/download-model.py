import boto3
import os
import tarfile

s3 = boto3.client('s3')
BUCKET_NAME = "sagemaker-us-east-1-436061841671"
s3_resource = boto3.resource('s3')
bucket = s3_resource.Bucket(BUCKET_NAME) 

for obj in bucket.objects.filter(Prefix = "huggingface-pytorch-training-2023-11-28-23-46-43-395/output/model.tar.gz"):
    # if not os.path.exists(os.path.dirname(obj.key)):
    #    os.makedirs(os.path.dirname(obj.key))
    print(os.path.basename(obj.key))
# bucket.download_file("huggingface-pytorch-training-2023-11-28-23-46-43-395/output/model.tar.gz", "./model.tar.gz") # save to same path
tar = tarfile.open("./model.tar.gz", "r:gz")
tar.extractall("model")
tar.close()
