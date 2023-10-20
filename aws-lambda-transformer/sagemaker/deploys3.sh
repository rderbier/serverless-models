tar -C model -zcvf sentence-transformer-all-minilm-l6-v2.tar.gz .
aws s3 cp ./sentence-transformer-all-minilm-l6-v2.tar.gz s3://hm-sentence-transformers --profile marketing
