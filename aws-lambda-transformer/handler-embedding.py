import json
import os
import pydgraph
from pybars import Compiler
from sentence_transformers import SentenceTransformer

# change CACHE else we get following error at runtime in AWS Lambda:
# There was a problem when trying to write in your cache folder (/home/sbx_user1051/.cache/huggingface/hub).
os.environ['TRANSFORMERS_CACHE'] = '/var/cache'
model = SentenceTransformer('/var/task/model')
compiler = Compiler()
def handler(event, context):
    source = event.get('source')
    type = event.get('detail-type')
    if source == 'aws.events' and type == 'Scheduled Event': 
       return {"statusCode": 200, "body": "Lambda is warm!"}
    
    body = json.loads(event['body'])
    sentences = [body[key] for key in body]
    
    embeddings = model.encode(sentences)
    keylist = list(body.keys())
    resp = {keylist[i]:json.dumps(embeddings[i].tolist()) for i in range(len(embeddings))}

    return {"statusCode": 200, "body": json.dumps(resp)}

def computeEmbedding(predicate,data, template):
    print("Compute embeddings")
    nquad_list = []
    sentences = [template(e) for e in data]
    print(sentences)
    # embeddings = model.encode(sentences)
    
    #keylist =  [e['uid']  for e in data]
    #print(keylist)
    
    for i in range(len(sentences)):
       nquad_list.append(f'<> <{predicate}> "{sentences[i]}" .')
    return nquad_list

def embedding(event, context):
    body = json.loads(event['body'])
    entity = body['type']
    predicate = body['predicate']
    query  =  body['query']
    prompt = body['prompt']
    template = compiler.compile(prompt)
    client_stub = pydgraph.DgraphClientStub(body['grpc_endpoint'])
    client = pydgraph.DgraphClient(client_stub)
    
    query = f"{{promptData(func: type({entity}),first:100) {{ uid {query} }} }}"
    
    data = {}
    try:
        txn = client.txn(read_only=True)
        res = txn.query(query)
        data = json.loads(res.json)
        # print(json.dumps(data))
        
        nquads = computeEmbedding(predicate,data['promptData'],template)

    except Exception as inst:
        print(inst)
    finally:
        txn.discard()

    return {"statusCode": 200, "body": nquads}