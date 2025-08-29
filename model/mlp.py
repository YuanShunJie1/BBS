import math
import torch
import torch.nn as nn
import torch.nn.init as init 
import torch.nn.functional as F
import torch.optim as optim

def call_bn(bn, x):
    return bn(x)

class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x, feature=False):
        x = x.view(-1, 28 * 28)
        h = F.relu(self.fc1(x))
        logit = self.fc2(h)
        if feature:
            return logit, h 
        return logit


class TeacherMLP(nn.Module):
    def __init__(self, num_classes):
        super(TeacherMLP, self).__init__()

        self.num_classes = num_classes
        self.feature_extractor = nn.Sequential(*list(MLPNet().children())[:-1])#.eval()
        self.fully_connect = nn.Sequential(
            nn.Linear(in_features=256 + self.num_classes, out_features=512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=128, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, images, labels):
        features = self.feature_extractor(images.view(-1, 28 * 28))
        x = torch.flatten(features, 1)
        label_emb = F.one_hot(labels, self.num_classes)
        logit = self.fully_connect(torch.cat([x, label_emb], dim = 1))
        return logit



