# Load model from HF and save locally
from transformers import AutoTokenizer, T5ForConditionalGeneration 
# can use the generic class AutoModelForSeq2SeqLM instead of T5ForConditionalGeneration

# https://huggingface.co/docs/transformers/main_classes/text_generation

modelPath = "./model"
text = """ 
summarize: I0215 13:56:54.788721   11870 log.go:33] 1 received MsgPreVoteResp from 1 at term 1
I0215 13:56:54.788815   11870 log.go:33] 1 became candidate at term 2
I0215 13:56:54.788823   11870 log.go:33] 1 received MsgVoteResp from 1 at term 2
I0215 13:56:54.788844   11870 log.go:33] 1 became leader at term 2
I0215 13:56:54.788938   11870 log.go:33] raft.node: 1 elected leader 1 at term 2
I0215 13:56:54.789008   11870 raft.go:958] I've become the leader, updating leases.
I0215 13:56:54.789032   11870 assign.go:47] Updated UID: 1. Txn Ts: 1. NsID: 1.
E0215 13:56:55.791483   11870 raft.go:590] While proposing CID: Not Zero leader. Aborting proposal: cid:"a14c6475-e4dc-4c3f-aa47-74a020ea11b6" . Retrying...
W0215 13:56:56.624094   11870 node.go:710] [0x1] Read index context timed out
I0215 13:56:56.627402   11870 zero.go:536] Connected: cluster_info_only:true 
I0215 13:56:56.645168   11870 zero.go:511] Got connection request: addr:"localhost:7080" 
I0215 13:56:56.646432   11870 pool.go:165] CONN: Connecting to localhost:7080
I0215 13:56:56.646756   11870 zero.go:676] Connected: id:1 group_id:1 addr:"localhost:7080" 
I0215 13:56:58.791976   11870 raft.go:583] CID set for cluster: 4da2d34f-d0cc-4a6c-b3d3-55b6af1c1bf2
I0215 13:56:58.792136   11870 license_ee.go:48] Enterprise trial license proposed to the cluster: license:<maxNodes:18446744073709551615 expiryTs:1710626218 > 
I0215 13:57:52.794844   11870 raft.go:826] Skipping creating a snapshot. Num groups: 1, Num checkpoints: 0
E0216 08:21:43.263066   11870 pool.go:314] CONN: Unable to connect with localhost:7080 : rpc error: code = Unavailable desc = connection error: desc = "transport: Error while dialing: dial tcp [::1]:7080: connect: connection refused"
E0216 08:21:44.270882   11870 pool.go:314] CONN: Unable to connect with localhost:7080 : rpc error: code = Unavailable desc = connection error: desc = "transport: Error while dialing: dial tcp [::1]:7080: connect: connection refused"
E0216 08:21:45.273271   11870 pool.go:314] CONN: Unable to connect with localhost:7080 : rpc error: code = Unavailable desc = connection error: desc = "transport: Error while dialing: dial tcp [::1]:7080: connect: connection refused"
I0216 08:21:45.400656   11870 zero.go:511] Got connection request: cluster_info_only:true 
I0216 08:21:45.400803   11870 zero.go:536] Connected: cluster_info_only:true 
I0216 08:21:45.401267   11870 zero.go:511] Got connection request: id:1 addr:"localhost:7080" 
I0216 08:21:45.401346   11870 zero.go:658] Connected: id:1 addr:"localhost:7080" 
I0216 08:21:46.279793   11870 pool.go:332] CONN: Re-established connection with localhost:7080."""

tokenizer = AutoTokenizer.from_pretrained("Falconsai/text_summarization")
model = T5ForConditionalGeneration.from_pretrained(modelPath)

# model.save_pretrained(modelPath)

input_ids = tokenizer.encode(text, max_length=512, truncation=True, return_tensors="pt")
outputs = model.generate(input_ids, max_new_tokens=100, temperature=0.8, do_sample=True)
decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(decoded)
