# load the model
from transformers import AutoModelForSequenceClassification,AutoTokenizer
import torch
import numpy as np
 
load_model = AutoModelForSequenceClassification.from_pretrained("./sagemaker/model")
 
load_tokenizer = AutoTokenizer.from_pretrained("./sagemaker/model")

text = ["XXXMobileMovieClub: To use your credit, click the WAP link in the next txt message or click here>> http://wap. xxxmobilemovieclub.com?n=QJKGIGHJJGCBL",
        "This is a normal text and should not be seen as spam!"
        ]
encoded_input = load_tokenizer(text, return_tensors='pt', padding=True)
print(encoded_input)
output = load_model(**encoded_input)
outputs = torch.nn.functional.softmax(output.logits, dim = -1).detach()
print(outputs.tolist())
print(load_model.config.id2label)
