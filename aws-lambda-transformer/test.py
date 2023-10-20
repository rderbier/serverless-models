import faulthandler; faulthandler.enable()
import json
import os
os.environ['TRANSFORMERS_CACHE'] = '/tmp'
from sentence_transformers import SentenceTransformer

# change CACHE else we get following error at runtime in AWS Lambda:
# There was a problem when trying to write in your cache folder (/home/sbx_user1051/.cache/huggingface/hub).


model = SentenceTransformer('/var/task/model')

sentences = ["some sentence to transform"]
    
embeddings = model.encode(sentences)

print(embeddings)

