# -*- coding:utf-8 -*-
import os
import torch
import torchvision.transforms as transforms
from data.cifar import CIFAR10, CIFAR100
from data.mnist import MNIST
import argparse, sys
from algorithm.bss_ds import BSS
from data.fmnist import FMNIST
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--result_dir', type=str, help='dir to save result txt files', default='checkpoint')
parser.add_argument('--noise_rate', type=float, help='corruption rate, should be less than 1', default=0.4)
parser.add_argument('--forget_rate', type=float, help='forget rate', default=None)
parser.add_argument('--noise_type', type=str, help='[pairflip, symmetric, asymmetric]', default='asymmetric')
parser.add_argument('--num_gradual', type=int, default=10, help='can be 5, 10, 15. This parameter is equal to Tk for R(T) in paper.')
parser.add_argument('--exponent', type=float, default=1, help='0.5, 1, 2. This parameter is equal to c in Tc for R(T) in paper.')
parser.add_argument('--dataset', type=str, help='mnist, fmnist, cifar10, cifar100, food101', default='cifar10')
parser.add_argument('--n_epoch', type=int, default=200)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--epoch_decay_start', type=int, default=80)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--co_lambda', type=float, default=0.5)
parser.add_argument('--model_type', type=str, help='[mlp,cnn,rnet]', default='mlp')
parser.add_argument('--save_result', type=str, help='save result?', default="True")

parser.add_argument('--device', type=int, default=0)

parser.add_argument('--p_thresh', type=float, default=0.5)
parser.add_argument('--s_thresh', type=float, default=0.03)
parser.add_argument('--early_stop', type=int, default=20)
parser.add_argument('--times', type=int, default=4)
parser.add_argument('--w_alpha', type=float, default=0.7)

args = parser.parse_args()

batch_size = 128
learning_rate = args.lr


# Seed
if torch.cuda.is_available():
    device = torch.device('cuda:{}'.format(args.device))
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
else:
    device = torch.device('cpu')
    torch.manual_seed(123)

if args.dataset == 'fmnist':
    input_channel = 1
    num_classes = 10
    args.epoch_decay_start = 80
    args.model_type = "mlp"
    
    # args.s_thresh = 0.03
    # if args.noise_rate==0.8:
    #     args.s_thresh = 0.1   
        
    train_dataset = FMNIST(root='./data/', download=False,train=True,transform=transforms.ToTensor(),noise_type=args.noise_type,noise_rate=args.noise_rate)
    test_dataset = FMNIST(root='./data/', download=False, train=False,transform=transforms.ToTensor(),noise_type=None, noise_rate=args.noise_rate)

if args.dataset == 'mnist':
    input_channel = 1
    num_classes = 10
    args.epoch_decay_start = 80
    args.model_type = "mlp"

    # args.s_thresh = 0.03
    # if args.noise_rate==0.8:
    #     args.s_thresh = 0.1   
    
    train_dataset = MNIST(root='./data/',
                          download=False,
                          train=True,
                          transform=transforms.ToTensor(),
                          noise_type=args.noise_type,
                          noise_rate=args.noise_rate)

    test_dataset = MNIST(root='./data/',
                         download=False,
                         train=False,
                         transform=transforms.ToTensor(),
                         noise_type=args.noise_type,
                         noise_rate=args.noise_rate)

if args.dataset == 'cifar10':
    input_channel = 3
    num_classes = 10
    args.epoch_decay_start = 80
    args.model_type = "cnn"
    
    train_dataset = CIFAR10(root='./data/',
                            download=True,
                            train=True,
                            transform=transforms.ToTensor(),
                            noise_type=args.noise_type,
                            noise_rate=args.noise_rate)

    test_dataset = CIFAR10(root='./data/',
                           download=True,
                           train=False,
                           transform=transforms.ToTensor(),
                           noise_type=args.noise_type,
                           noise_rate=args.noise_rate)

if args.dataset == 'cifar100':
    input_channel = 3
    num_classes = 100
    args.epoch_decay_start = 100
    filter_outlier = False
    args.model_type = "cnn"
    #args.times = 10
    #args.w_alpha = 0.9
    train_dataset = CIFAR100(root='./data/',
                             download=True,
                             train=True,
                             transform=transforms.ToTensor(),
                             noise_type=args.noise_type,
                             noise_rate=args.noise_rate)

    test_dataset = CIFAR100(root='./data/',
                            download=True,
                            train=False,
                            transform=transforms.ToTensor(),
                            noise_type=args.noise_type,
                            noise_rate=args.noise_rate)



acc1_log=open(f'{args.result_dir.strip()}/dataste={args.dataset}_noise_rate={args.noise_rate}_noise_type={args.noise_type}_s={args.s_thresh}_pt={args.p_thresh}_alpha={args.w_alpha}_times={args.times}_m1_acc.txt','w')

acc2_log=open(f'{args.result_dir.strip()}/dataste={args.dataset}_noise_rate={args.noise_rate}_noise_type={args.noise_type}_s={args.s_thresh}_pt={args.p_thresh}_alpha={args.w_alpha}_times={args.times}_m2_acc.txt','w')

ssacc_log=open(f'{args.result_dir.strip()}/dataste={args.dataset}_noise_rate={args.noise_rate}_noise_type={args.noise_type}_s={args.s_thresh}_pt={args.p_thresh}_alpha={args.w_alpha}_times={args.times}_ss_acc.txt','w')

tsacc_log=open(f'{args.result_dir.strip()}/dataste={args.dataset}_noise_rate={args.noise_rate}_noise_type={args.noise_type}_s={args.s_thresh}_pt={args.p_thresh}_alpha={args.w_alpha}_times={args.times}_ts_acc.txt','w')

final_detection_log=open(f'{args.result_dir.strip()}/dataste={args.dataset}_noise_rate={args.noise_rate}_noise_type={args.noise_type}_s={args.s_thresh}_pt={args.p_thresh}_alpha={args.w_alpha}_times={args.times}_final_all.txt','w')

def main():
    print('loading dataset...')

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=4, drop_last=True, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=4, drop_last=True, shuffle=False)

    print('building model...')

    method = BSS(args, train_dataset, device, input_channel, num_classes)

    # for epoch in range(0, args.n_epoch):
    for epoch in range(0, args.early_stop):
        ssaccs, tsaccs, loss1, lossb, precision, recall, f1 = method.train(train_loader, epoch)
        test_acc1, test_acc2 = method.evaluate(test_loader)

        acc1_log.write('%.2f\n'%(test_acc1))
        acc1_log.flush() 

        acc2_log.write('%.2f\n'%(test_acc2))
        acc2_log.flush() 

        ssacc_log.write('%.2f\n'%(np.mean(ssaccs)))
        ssacc_log.flush()

        tsacc_log.write('%.2f\n'%(np.mean(tsaccs)))
        tsacc_log.flush() 

        final_detection_log.write('precision:%.2f recall:%.2f f1:%.2f \n'%(precision, recall, f1))
        final_detection_log.flush()

    # method.collect_and_draw(train_loader,epoch=100)


if __name__ == '__main__':
    main()


# test_acc1, test_acc2 = method.evaluate(test_loader)
# print('\nDataset: %s Noise: %s-%f Test Accuracy on the %s test images: Model1 %.2f %% Model2 %.2f ' % (args.dataset, args.noise_type, args.noise_rate, len(test_dataset), test_acc1, test_acc2))
