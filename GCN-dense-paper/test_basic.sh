python3 train_res.py --seed 42 --epochs 200 --model gcn3 --runs 100 --dataset cora     >gcn_cora.txt
python3 train_res.py --seed 42 --epochs 200 --model gcn3 --runs 100 --dataset citeseer >gcn_citeseer.txt
python3 train_res.py --seed 42 --epochs 200 --model gcn3 --runs 100 --dataset pubmed   >gcn_pubmed.txt

python3 train_res.py --seed 42 --epochs 200 --model res3 --runs 100 --dataset cora     >res_cora.txt
python3 train_res.py --seed 42 --epochs 200 --model res3 --runs 100 --dataset citeseer >res_citeseer.txt
python3 train_res.py --seed 42 --epochs 200 --model res3 --runs 100 --dataset pubmed   >res_pubmed.txt

python3 train_res.py --seed 42 --epochs 200 --model gcn3norm --runs 100 --dataset cora     >gcnnorm_cora.txt
python3 train_res.py --seed 42 --epochs 200 --model gcn3norm --runs 100 --dataset citeseer >gcnnorm_citeseer.txt
python3 train_res.py --seed 42 --epochs 200 --model gcn3norm --runs 100 --dataset pubmed   >gcnnorm_pubmed.txt

python3 train_res.py --seed 42 --epochs 200 --model res3norm --runs 100 --dataset cora     >resnorm_cora.txt
python3 train_res.py --seed 42 --epochs 200 --model res3norm --runs 100 --dataset citeseer >resnorm_citeseer.txt
python3 train_res.py --seed 42 --epochs 200 --model res3norm --runs 100 --dataset pubmed   >resnorm_pubmed.txt

# ODE models are slower to train, and thus we test them with a lower number of runs

python3 train_res.py --seed 42 --epochs 200 --model ode3 --runs 100 --dataset cora     >ode_cora.txt
python3 train_res.py --seed 42 --epochs 200 --model ode3 --runs 100 --dataset citeseer >ode_citeseer.txt
python3 train_res.py --seed 42 --epochs 200 --model ode3 --runs 100 --dataset pubmed   >ode_pubmed.txt
