# ODE models are slower to train, and thus we test them with a lower number of runs

python3 train_res.py --seed 42 --epochs 200 --model ode3 --runs 50 --dataset cora     >ode_cora.txt
python3 train_res.py --seed 42 --epochs 200 --model ode3 --runs 50 --dataset citeseer >ode_citeseer.txt
python3 train_res.py --seed 42 --epochs 200 --model ode3 --runs 10 --dataset pubmed   >ode_pubmed.txt
