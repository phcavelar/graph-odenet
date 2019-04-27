python3 train_res.py --seed 42 --epochs 200 --model res3 --runs 2500 --dataset cora     >res_cora.txt
python3 train_res.py --seed 42 --epochs 200 --model res3 --runs 2500 --dataset citeseer >res_citeseer.txt
python3 train_res.py --seed 42 --epochs 200 --model res3 --runs 2500 --dataset pubmed   >res_pubmed.txt

python3 train_res.py --seed 42 --epochs 200 --model gcn3 --runs 2500 --dataset cora     >gcn_cora.txt
python3 train_res.py --seed 42 --epochs 200 --model gcn3 --runs 2500 --dataset citeseer >gcn_citeseer.txt
python3 train_res.py --seed 42 --epochs 200 --model gcn3 --runs 2500 --dataset pubmed   >gcn_pubmed.txt

python3 train_res.py --seed 42 --epochs 200 --model gcn3norm --runs 2500 --dataset cora     >gcnnorm_cora.txt
python3 train_res.py --seed 42 --epochs 200 --model gcn3norm --runs 2500 --dataset citeseer >gcnnorm_citeseer.txt
python3 train_res.py --seed 42 --epochs 200 --model gcn3norm --runs 2500 --dataset pubmed   >gcnnorm_pubmed.txt

python3 train_res.py --seed 42 --epochs 200 --model res3norm --runs 2500 --dataset cora     >resnorm_cora.txt
python3 train_res.py --seed 42 --epochs 200 --model res3norm --runs 2500 --dataset citeseer >resnorm_citeseer.txt
python3 train_res.py --seed 42 --epochs 200 --model res3norm --runs 2500 --dataset pubmed   >resnorm_pubmed.txt

# ODE models are slower to train, and thus we test them with a lower number of runs

python3 train_res.py --seed 42 --epochs 200 --model ode3 --runs 200 --dataset cora     >ode_cora.txt
python3 train_res.py --seed 42 --epochs 200 --model ode3 --runs 200 --dataset citeseer >ode_citeseer.txt
python3 train_res.py --seed 42 --epochs 200 --model ode3 --runs 50 --dataset pubmed   >ode_pubmed.txt

# The models below have failed to converge on preliminary tests and thus we train them only for a small number of runs to collect results

python3 train_res.py --seed 42 --epochs 200 --model res3fullnorm --runs 50 --dataset cora     >resfullnorm_cora.txt
python3 train_res.py --seed 42 --epochs 200 --model res3fullnorm --runs 50 --dataset citeseer >resfullnorm_citeseer.txt
python3 train_res.py --seed 42 --epochs 200 --model res3fullnorm --runs 50 --dataset pubmed   >resfullnorm_pubmed.txt

python3 train_res.py --seed 42 --epochs 200 --model ode3fullnorm --runs 10 --dataset cora     >odenorm_cora.txt
python3 train_res.py --seed 42 --epochs 200 --model ode3fullnorm --runs 10 --dataset citeseer >odenorm_citeseer.txt
python3 train_res.py --seed 42 --epochs 200 --model ode3fullnorm --runs 10 --dataset pubmed   >odenorm_pubmed.txt
