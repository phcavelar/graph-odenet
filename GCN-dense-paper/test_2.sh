python3 train_2.py --seed 42 --epochs 200 --model gcn2 --runs 100 --dataset cora     >gcn2_cora.txt
python3 train_2.py --seed 42 --epochs 200 --model res2 --runs 100 --dataset cora     >res2_cora.txt
python3 train_2.py --seed 42 --epochs 200 --model ode2 --runs 100 --dataset cora     >ode2_cora.txt

#python3 train_2.py --seed 42 --epochs 200 --model gcn2 --runs 100 --dataset citeseer >gcn2_citeseer.txt
#python3 train_2.py --seed 42 --epochs 200 --model res2 --runs 100 --dataset citeseer >res2_citeseer.txt
#python3 train_2.py --seed 42 --epochs 200 --model ode2 --runs 100 --dataset citeseer >ode2_citeseer.txt

#python3 train_2.py --seed 42 --epochs 200 --model res2 --runs 100 --dataset pubmed   >res2_pubmed.txt
#python3 train_2.py --seed 42 --epochs 200 --model gcn2 --runs 100 --dataset pubmed   >gcn2_pubmed.txt
#python3 train_2.py --seed 42 --epochs 200 --model ode2 --runs 100 --dataset pubmed   >ode2_pubmed.txt
