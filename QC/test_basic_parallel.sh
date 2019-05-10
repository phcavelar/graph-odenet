nohup python3 train_egcn.py --model enns2s --prefetch 2 --dataset qm9     >enns2s_qm9.txt &
nohup python3 train_egcn.py --model ennsum --prefetch 2 --dataset qm9     >ennsum_qm9.txt &
nohup python3 train_egcn.py --model egcn3s2s --prefetch 2 --dataset qm9    >egcn3s2s_qm9.txt &
nohup python3 train_egcn.py --model egcn3sum --prefetch 2 --dataset qm9    >egcn3sum_qm9.txt &
