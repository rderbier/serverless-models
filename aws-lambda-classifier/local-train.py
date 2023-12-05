import pandas as pd
import os
import s3fs
from sklearn.model_selection import train_test_split
import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification, TrainingArguments,Trainer
import evaluate
from datasets import Dataset
import numpy as np

file_path="./data/spam.csv"
df = pd.read_csv(file_path,encoding = "ISO-8859-1")
df['labels'] = df.Category.map({'ham':0, 'spam':1})
train, eval = train_test_split(df,train_size=.75,shuffle=True,random_state=69)
train_dataset = Dataset.from_pandas(train)
eval_dataset = Dataset.from_pandas(eval)
# train_size represents the proportion of the dataset to include in the train split.
# random_state: Pass an int for reproducible output across multiple function calls
print('train and eval shape')
print(train_dataset.shape)
print(eval_dataset.shape)

AWS_S3_BUCKET = "sagemaker-us-east-1-670738750911"


AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")





tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(batch):
    return tokenizer(batch["Message"], padding="max_length", truncation=True, return_tensors="pt")

tokenized_dataset_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["Message"])
tokenized_dataset_eval = eval_dataset.map(tokenize_function, batched=True, remove_columns=["Message"])
# set format for pytorch
#tokenized_dataset_train =   tokenized_dataset_train.rename_column("label", "labels")
#tokenized_dataset_train.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
#tokenized_dataset_eval = tokenized_dataset_eval.rename_column("label", "labels")
#tokenized_dataset_eval.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

#  AutoModelForSequenceClassification has a classification head on top of the model outputs which can be easily trained with the base model
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased',num_labels=2).to(torch.device("mps"))

training_args = TrainingArguments(
    output_dir=".", 
    evaluation_strategy="epoch", 
    num_train_epochs=1,
    # PyTorch 2.0 specifics
    # bf16=True, # bfloat16 training BF16 Mixed precision training with AMP (`--bf16`) and BF16 half precision evaluation (`--bf16_full_eval`) can only be used on CUDA, XPU (with IPEX), NPU or CPU/TPU/NeuronCore devices.
	# torch_compile=True, # optimizations
    # optim="adamw_torch_fused", # improved optimizer
    )

metric = evaluate.load("accuracy") 

 
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels) 

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset_train,
    eval_dataset=tokenized_dataset_eval,
    compute_metrics=compute_metrics,
 )
#
#
trainer.train()

#model.save_pretrained("./sagemaker/model")
# alternatively save the trainer
trainer.save_model("./sagemaker/model2")
 
# tokenizer.save_pretrained("./sagemaker/model")
