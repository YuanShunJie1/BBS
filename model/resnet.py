import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(3,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)


        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, lin=0, lout=5):
        out = x

        out = self.conv1(out)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.layer1(out)

        out = self.layer2(out)

        out = self.layer3(out)

        out = self.layer4(out)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes=10):
    return ResNet(PreActBlock, [2,2,2,2], num_classes=num_classes)

def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3,4,6,3], num_classes=num_classes)

def ResNet50(num_classes=10):
    return ResNet(Bottleneck, [3,4,6,3], num_classes=num_classes)

def ResNet101(num_classes=10):
    return ResNet(Bottleneck, [3,4,23,3], num_classes=num_classes)

def ResNet152(num_classes=10):
    return ResNet(Bottleneck, [3,8,36,3], num_classes=num_classes)




class TeacherResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(TeacherResNet, self).__init__()
        self.in_planes = 64
        self.num_classes = num_classes
        self.conv1 = conv3x3(3,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # self.linear = nn.Linear(512*block.expansion, num_classes)


        self.fully_connect = nn.Sequential(
            nn.Linear(in_features=512*block.expansion + self.num_classes, out_features=512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=128, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=1, bias=True),
            nn.Sigmoid())


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    def forward(self, images, labels):
        out = images
        out = self.conv1(out)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.layer1(out)

        out = self.layer2(out)

        out = self.layer3(out)

        out = self.layer4(out)

        out = F.avg_pool2d(out, 4)

        features = out.view(out.size(0), -1)
        label_emb = F.one_hot(labels, self.num_classes)

        logits = self.fully_connect(torch.cat([features, label_emb], dim = 1))
        return logits


def TeacherResNet18(num_classes=10):
    return TeacherResNet(PreActBlock, [2,2,2,2], num_classes=num_classes)

def TeacherResNet34(num_classes=10):
    return TeacherResNet(BasicBlock, [3,4,6,3], num_classes=num_classes)

def TeacherResNet50(num_classes=10):
    return TeacherResNet(Bottleneck, [3,4,6,3], num_classes=num_classes)

def TeacherResNet101(num_classes=10):
    return TeacherResNet(Bottleneck, [3,4,23,3], num_classes=num_classes)

def TeacherResNet152(num_classes=10):
    return TeacherResNet(Bottleneck, [3,8,36,3], num_classes=num_classes)


# class TeacherCNN(nn.Module):
#     def __init__(self, num_classes, input_channel=3, n_outputs=1, dropout_rate=0.25, momentum=0.1):       
#         self.dropout_rate = dropout_rate
#         self.momentum = momentum
#         self.num_classes = num_classes
#         super(TeacherCNN, self).__init__()

#         # load a pre-trained model for the feature extractor
#         # self.feature_extractor = nn.Sequential(*list(StudentCNN().children())[:-1])#.eval()

#         self.feature_extractor = nn.Sequential(
#             nn.Conv2d(input_channel, 64,kernel_size=3,stride=1, padding=1),
#             nn.BatchNorm2d(64, momentum=0.1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64,64,kernel_size=3,stride=1, padding=1),
#             nn.BatchNorm2d(64, momentum=0.1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Dropout(p=0.25, inplace=False),


#             nn.Conv2d(64, 128,kernel_size=3,stride=1, padding=1),
#             nn.BatchNorm2d(128, momentum=0.1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128,128,kernel_size=3,stride=1, padding=1),
#             nn.BatchNorm2d(128, momentum=0.1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Dropout(p=0.25, inplace=False),


#             nn.Conv2d(128, 196,kernel_size=3,stride=1, padding=1),
#             nn.BatchNorm2d(196, momentum=0.1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(196,16,kernel_size=3,stride=1, padding=1),
#             nn.BatchNorm2d(16, momentum=0.1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#         )

#         self.fully_connect = nn.Sequential(
#             nn.Linear(in_features=256 + self.num_classes, out_features=512, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_features=512, out_features=128, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_features=128, out_features=n_outputs, bias=True),
#             nn.Sigmoid()
#         )

#     def forward(self, images, labels):
#         features = self.feature_extractor(images)
#         features = features.view(features.size(0), -1)
#         # x = torch.flatten(features, 1)
#         label_emb = F.one_hot(labels, self.num_classes)
#         logit = self.fully_connect(torch.cat([features, label_emb], dim = 1))
#         return logit







# def test():
#     net = ResNet18()
#     y = net(Variable(torch.randn(1,3,32,32)))
#     print(y.size())
