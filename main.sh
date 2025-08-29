# python main.py --dataset mnist --noise_type symmetric --noise_rate 0.20 --times 4 
python main.py --dataset mnist --noise_type symmetric --noise_rate 0.50 --times 4 
python main.py --dataset mnist --noise_type symmetric --noise_rate 0.80 --times 4 
python main.py --dataset mnist --noise_type asymmetric --noise_rate 0.40 --times 4 

python main.py --dataset fmnist --noise_type symmetric --noise_rate 0.20 --times 4 
python main.py --dataset fmnist --noise_type symmetric --noise_rate 0.50 --times 4 
python main.py --dataset fmnist --noise_type symmetric --noise_rate 0.80 --times 4 
python main.py --dataset fmnist --noise_type asymmetric --noise_rate 0.40 --times 4 

python main.py --dataset cifar10 --noise_type symmetric --noise_rate 0.20 --times 4 
python main.py --dataset cifar10 --noise_type symmetric --noise_rate 0.50 --times 4 
python main.py --dataset cifar10 --noise_type symmetric --noise_rate 0.80 --times 4 
python main.py --dataset cifar10 --noise_type asymmetric --noise_rate 0.40 --times 4 

python main.py --dataset cifar100 --noise_type symmetric --noise_rate 0.20 --times 10 
python main.py --dataset cifar100 --noise_type symmetric --noise_rate 0.50 --times 10 
python main.py --dataset cifar100 --noise_type symmetric --noise_rate 0.80 --times 10 
python main.py --dataset cifar100 --noise_type asymmetric --noise_rate 0.40 --times 10 

