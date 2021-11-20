python3 train_model.py 100 bigdataset_100 2>&1 | tee output_bigdataset_100
python3 train_model.py 1000 bigdataset_1000 2>&1 | tee output_bigdataset_1000
python3 train_model.py 5000 bigdataset_5000 2>&1 | tee output_bigdataset_5000
python3 train_model.py 10000 bigdataset_10000 2>&1 | tee output_bigdataset_10000
python3 train_model.py 20000 bigdataset_20000 2>&1 | tee output_bigdataset_20000
