
from transformers import AutoTokenizer,AutoModelForSequenceClassification
import s3fs

model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased',num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

AWS_S3_BUCKET = "sagemaker-us-east-1-670738750911"


model.save_pretrained(f"s3://{AWS_S3_BUCKET}/basemodel")
# alternatively save the trainer
# trainer.save_model("CustomModels/CustomHamSpam")
 
tokenizer.save_pretrained(f"s3://{AWS_S3_BUCKET}/basemodel")
