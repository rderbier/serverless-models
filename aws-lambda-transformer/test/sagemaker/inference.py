import json
import os
from sentence_transformers import SentenceTransformer

def model_fn(model_dir):
    model = SentenceTransformer(model_dir)
    return model

def predict_fn(data, model):
    print("Body: {}".format(data))
    sentences = [data[key] for key in data]
    outputs = model.encode(sentences)
    keylist = list(data.keys())
    resp = {keylist[i]:json.dumps(outputs[i].tolist()) for i in range(len(outputs))}

    return json.dumps(resp)