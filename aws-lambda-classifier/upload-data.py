import pandas as pd
import os
import s3fs
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import load_metric, Dataset

file_path="./data/spam.csv"
df = pd.read_csv(file_path,encoding = "ISO-8859-1")
df['labels'] = df.Category.map({'ham':0, 'spam':1})

# train_test_split: Split arrays or matrices into random train and test subsets.
# inputs are lists, numpy arrays, scipy-sparse matrices or pandas dataframes
# output type is the same as the input type.
# train_size represents the proportion of the dataset to include in the train split.
# random_state: Pass an int for reproducible output across multiple function calls
train, eval = train_test_split(df,train_size=.75,shuffle=True,random_state=69)
print(train.shape)
print(eval.shape)

AWS_S3_BUCKET = "sagemaker-marketing-classifier-001"


AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")

train.to_csv(
    f"s3://{AWS_S3_BUCKET}/data/train.csv",
    index=False
)
eval.to_csv(
    f"s3://{AWS_S3_BUCKET}/data/eval.csv",
    index=False
)


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
def tokenize_function(examples):
    return tokenizer(examples["Message"], padding="max_length", truncation=True,return_tensors="pt")
dataset_train = Dataset.from_pandas(train)
dataset_eval = Dataset.from_pandas(eval)
tokenized_dataset_train = dataset_train.map(tokenize_function, batched=True)
tokenized_dataset_eval = dataset_eval.map(tokenize_function, batched=True)

tokenized_dataset_train.save_to_disk(f"s3://{AWS_S3_BUCKET}/train")

tokenized_dataset_eval.save_to_disk(f"s3://{AWS_S3_BUCKET}/eval")