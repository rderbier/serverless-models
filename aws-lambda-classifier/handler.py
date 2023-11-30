import json
import os
from transformers import AutoModelForSequenceClassification,AutoTokenizer
import torch
os.environ['TRANSFORMERS_CACHE'] = '/tmp'


# change CACHE else we get following error at runtime in AWS Lambda:
# There was a problem when trying to write in your cache folder (/home/sbx_user1051/.cache/huggingface/hub).

model_dir = '/var/task/model'
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

def handler(event, context):
    # source = event.get('source')
    # type = event.get('detail-type')
    # if source == 'aws.events' and type == 'Scheduled Event': 
    #    return {"statusCode": 200, "body": "Lambda is warm!"}
    
    body = json.loads(event['body'])
    sentences = [body[key] for key in body]

    encoded_input = tokenizer(sentences, return_tensors='pt', padding=True)
    output = model(**encoded_input)
    outputs = torch.nn.functional.softmax(output.logits, dim = -1).detach()
    maxindex = torch.argmax(outputs, dim=1).tolist()
    final = outputs.tolist()
    print(final)
    keylist = list(body.keys())
    # id2label = model.config.id2label
    id2label = {0: 'NOT-SPAM', 1: 'SPAM'}


    resp = {keylist[i]:{'label': id2label[v], 'confidence':final[i][v], 'probabilities': [ {'label':id2label[j], 'probability':final[i][j] } for j,_ in enumerate(final[i]) ]} for i,v in enumerate(maxindex)}

    return {
        "statusCode": 200,
        "headers": {
            "Access-Control-Allow-Headers" : "Content-Type,X-Api-Key",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "OPTIONS,POST,GET"
        }, 
        "body": json.dumps(resp)
        }

