import json
import os
from transformers import AutoModelForSequenceClassification,AutoTokenizer
import torch

def model_fn(model_dir):
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model_dict = {'model':model, 'tokenizer':tokenizer}
    print("model and tokenizer created.")
    return model_dict

def predict_fn(data, model_dict):
    print("Body: {}".format(data))
    tokenizer = model_dict['tokenizer']
    model = model_dict['model']
    sentences = [data[key] for key in data]
    encoded_input = tokenizer(sentences, return_tensors='pt', padding=True)
    output = model(**encoded_input)
    outputs = torch.nn.functional.softmax(output.logits, dim = -1).detach()
    print(outputs.tolist())
    keylist = list(data.keys())
    resp = {keylist[i]:json.dumps(outputs[i].tolist()) for i in range(len(outputs))}
    return resp