tar -C model -zcvf model.tar.gz .
aws s3 cp ./model.tar.gz s3://hm-sentence-transformers
