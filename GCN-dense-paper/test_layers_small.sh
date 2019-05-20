nohup python3 train_layers.py --runs 50 --dataset cora     >layers_cora.txt &
nohup python3 train_layers.py --runs 50 --dataset citeseer >layers_citeseer.txt &
nohup python3 train_layers.py --runs 10 --dataset pubmed   >layers_pubmed.txt &
