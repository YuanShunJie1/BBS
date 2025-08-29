import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from tqdm import tqdm
from model.cnn import CNN, TeacherCNN
from model.mlp import MLPNet, TeacherMLP


class Decoupling:
    def __init__(
            self, 
            config, 
            input_channel, 
            num_classes,
            device
        ):

        self.lr = config['lr']
        # Adjust learning rate and betas for Adam Optimizer
        mom1 = 0.9
        mom2 = 0.1
        self.alpha_plan = [self.lr] * config['n_epoch']
        self.beta1_plan = [mom1] * config['n_epoch']

        for i in range(config['epoch_decay_start'], config['n_epoch']):
            self.alpha_plan[i] = float(config['n_epoch'] - i) / (config['n_epoch'] - config['epoch_decay_start']) * self.lr
            self.beta1_plan[i] = mom2

        self.device = device
        self.epochs = config['n_epoch']

        if config['dataset'] == "mnist" or config['dataset'] == "fmnist":
            self.model1 = MLPNet()
            self.model2 = MLPNet()
        else:
            self.model1 = CNN(input_channel=input_channel, n_outputs=num_classes)
            self.model2 = CNN(input_channel=input_channel, n_outputs=num_classes)
        
        self.model1 = self.model1.to(self.device)
        self.model2 = self.model2.to(self.device)
        
        self.optimizer = torch.optim.Adam(list(self.model1.parameters()) + list(self.model2.parameters()), lr=self.lr)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.adjust_lr = config['adjust_lr']

    def evaluate(self, test_loader):
        print('Evaluating ...')
        self.model1.eval()  # Change model to 'eval' mode.
        self.model2.eval()  # Change model to 'eval' mode

        correct1 = 0
        total1 = 0
        for images, labels, _ in test_loader:
            images = Variable(images).to(self.device)
            logits1 = self.model1(images)
            outputs1 = F.softmax(logits1, dim=1)
            _, pred1 = torch.max(outputs1.data, 1)
            total1 += labels.size(0)
            correct1 += (pred1.cpu() == labels).sum()

        correct2 = 0
        total2 = 0
        for images, labels, _ in test_loader:
            images = Variable(images).to(self.device)
            logits2 = self.model2(images)
            outputs2 = F.softmax(logits2, dim=1)
            _, pred2 = torch.max(outputs2.data, 1)
            total2 += labels.size(0)
            correct2 += (pred2.cpu() == labels).sum()

        acc1 = 100 * float(correct1) / float(total1)
        acc2 = 100 * float(correct2) / float(total2)
        return acc1, acc2

    def train(self, train_loader, epoch):
        print('Training ...')
        self.model1.train()  
        self.model2.train()

        if self.adjust_lr == 1:
            self.adjust_learning_rate(self.optimizer, epoch)

        pbar = tqdm(train_loader)
        for (images, labels, indices) in pbar:
            images = Variable(images).to(self.device)
            labels = Variable(labels).to(self.device)

            logits1 = self.model1(images)
            _, pred1 = torch.max(logits1, dim=1)
            logits2 = self.model2(images)
            _, pred2 = torch.max(logits2, dim=1)

            inds = torch.where(pred1 != pred2)[0]  # 注意取 [0] 得到索引

            if len(inds) == 0:  
                # 没有分歧样本 → 跳过该 batch，避免 NaN
                pbar.set_description(
                    'Epoch [%d/%d], skipped (no disagreement samples)' 
                    % (epoch + 1, self.epochs))
                continue  

            loss_1 = self.loss_fn(logits1[inds], labels[inds])
            loss_2 = self.loss_fn(logits2[inds], labels[inds])

            self.optimizer.zero_grad()
            loss = loss_1 + loss_2  # 可以合并，梯度更稳定
            loss.backward()
            self.optimizer.step()

            pbar.set_description(
                'Epoch [%d/%d], Loss1: %.4f, Loss2: %.4f'
                % (epoch + 1, self.epochs, loss_1.item(), loss_2.item()))


    def adjust_learning_rate(self, optimizer, epoch):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.alpha_plan[epoch]
            param_group['betas'] = (self.beta1_plan[epoch], 0.999)  # Only change beta1
            




    # 