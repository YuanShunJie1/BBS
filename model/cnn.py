import math
import torch
import torch.nn as nn
import torch.nn.init as init 
import torch.nn.functional as F
import torch.optim as optim

def call_bn(bn, x):
    return bn(x)

class CNN(nn.Module):
    def __init__(self, input_channel=3, n_outputs=10, dropout_rate=0.25, momentum=0.1):
        self.dropout_rate = dropout_rate
        self.momentum = momentum
        super(CNN, self).__init__()
        self.c1=nn.Conv2d(input_channel, 64,kernel_size=3,stride=1, padding=1)
        self.c2=nn.Conv2d(64,64,kernel_size=3,stride=1, padding=1)
        self.c3=nn.Conv2d(64,128,kernel_size=3,stride=1, padding=1)
        self.c4=nn.Conv2d(128,128,kernel_size=3,stride=1, padding=1)
        self.c5=nn.Conv2d(128,196,kernel_size=3,stride=1, padding=1)
        self.c6=nn.Conv2d(196,16,kernel_size=3,stride=1, padding=1)
        self.linear1=nn.Linear(256, n_outputs)
        self.bn1=nn.BatchNorm2d(64, momentum=self.momentum)
        self.bn2=nn.BatchNorm2d(64, momentum=self.momentum)
        self.bn3=nn.BatchNorm2d(128, momentum=self.momentum)
        self.bn4=nn.BatchNorm2d(128, momentum=self.momentum)
        self.bn5=nn.BatchNorm2d(196, momentum=self.momentum)
        self.bn6=nn.BatchNorm2d(16, momentum=self.momentum)

    def forward(self, x, feature=False):
        h=x
        h=self.c1(h)
        h=F.relu(call_bn(self.bn1, h))
        h=self.c2(h)
        h=F.relu(call_bn(self.bn2, h))
        h=F.max_pool2d(h, kernel_size=2, stride=2)

        h=self.c3(h)
        h=F.relu(call_bn(self.bn3, h))
        h=self.c4(h)
        h=F.relu(call_bn(self.bn4, h))
        h=F.max_pool2d(h, kernel_size=2, stride=2)

        h=self.c5(h)
        h=F.relu(call_bn(self.bn5, h))
        h=self.c6(h)
        h=F.relu(call_bn(self.bn6, h))
        h=F.max_pool2d(h, kernel_size=2, stride=2)

        h = h.view(h.size(0), -1)
        logit=self.linear1(h)
        
        if feature:
            return logit, h
        return logit


class TeacherCNN(nn.Module):
    def __init__(self, num_classes, input_channel=3, n_outputs=1, dropout_rate=0.25, momentum=0.1):       
        self.dropout_rate = dropout_rate
        self.momentum = momentum
        self.num_classes = num_classes
        super(TeacherCNN, self).__init__()

        # load a pre-trained model for the feature extractor
        # self.feature_extractor = nn.Sequential(*list(StudentCNN().children())[:-1])#.eval()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channel, 64,kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25, inplace=False),


            nn.Conv2d(64, 128,kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25, inplace=False),


            nn.Conv2d(128, 196,kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(196, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(196,16,kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(16, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

        )

        self.fully_connect = nn.Sequential(
            nn.Linear(in_features=256 + self.num_classes, out_features=512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=128, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=n_outputs, bias=True),
            nn.Sigmoid()
        )

    def forward(self, images, labels):
        features = self.feature_extractor(images)
        features = features.view(features.size(0), -1)
        # x = torch.flatten(features, 1)
        label_emb = F.one_hot(labels, self.num_classes)
        logit = self.fully_connect(torch.cat([features, label_emb], dim = 1))
        return logit



