import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import os

import torchvision
import torchvision.transforms
import graph

from torch.autograd import Variable
from utils import progress_bar

class Class_Balancing:
    def __init__(self,type="undersampling",transform=False):
        self.type = type
        self.transform = transform

    def class_balancing(self,x,y):
        if self.type == "undersampling":
            return self.class_balancing_undersampling(x,y)
        else:
            return self.class_balancing_replication(x,y)

    def class_balancing_replication(self,x,y):
        y_1 = np.where(y == 0)[0]
        y_5 = np.where(y == 1)[0]
        while y_1.shape[0] > y_5.shape[0]: 
            y_5 = np.where(y == 1)[0]
            np.random.shuffle(y_5)
            new = x[y_5[0]]
            new = self.transform(new) if self.transform else new 
            new = np.expand_dims(new,0)
            x = np.concatenate([x,new])
            y = np.concatenate([y,[1]])
            y_1 = np.where(y == 0)[0]
            y_5 = np.where(y == 1)[0]
        return x,y

    def class_balancing_undersampling(self,x,y):
        y_1 = np.where(y == 0)[0]
        y_5 = np.where(y == 1)[0]
        np.random.shuffle(y_1)
        y_1 = y_1[:y_5.shape[0]]
        indexes = torch.LongTensor(np.append(y_1,y_5))
        x = x[indexes]
        y = y[indexes]
        return x,y

    def __repr__(self):
        return self.type+" "+str(self.transform)

    def __str__(self):
        return self.type+" "+str(self.transform)

class MiniGraphPreActResNet(nn.Module):
    def __init__(self, conv_type,num_classes=2,dropout=False,layers=1,pooling=False):
        super(MiniGraphPreActResNet, self).__init__()
        
        self.conv = conv_type[0]
        self.conv_dict = conv_type[1]
        self.linear = self.conv == "linear"
        if self.linear:
            self.last_layer_size = 369
            self.layer_size = 64
        else:
            self.last_layer_size = 1
            self.layer_size = 64
        self.layers = list()
        self.pooling = pooling
        self.dropout = dropout

        self.dropout1 = nn.Dropout(0.1) if dropout else False
        self.dropout2 = nn.Dropout(0.1) if dropout else False


        for _ in range(layers):
            if self.linear:               
                self.layers.append(nn.Linear(self.last_layer_size, self.layer_size).cuda())
                self.layers.append(F.relu)
                self.last_layer_size = self.layer_size
                self.layer_size /= 2
                self.layer_size = int(self.layer_size)
                pass
            else:
                parameters_dict = dict(self.conv_dict,
                                    in_channels=self.last_layer_size,
                                    out_channels=self.layer_size)
                self.layers.append(self.conv(**parameters_dict).cuda())
                self.layers.append(F.relu)
                self.last_layer_size = self.layer_size
                self.layer_size *= 2
        if self.linear:
            self.linear_size = self.last_layer_size
        else:
            self.linear_size = 369*self.last_layer_size if not self.pooling else self.last_layer_size
        self.linear_layer = nn.Linear(self.linear_size, num_classes)



    def forward(self, x):
        if self.dropout:
            x = self.dropout1(x)
        if self.linear:
            x = x.view(-1,369)
        for layer in self.layers:
            x = layer(x)
        if not self.linear:
            if self.pooling:
                x = torch.mean(x,dim=2)
            else:
                x = x.view(-1,self.linear_size)
        if self.dropout:
            x = self.dropout2(x)
        x = self.linear_layer(x)
        return x


class PINESDataset(torch.utils.data.Dataset):

    def __init__(self, directory="../data/PINES/",mode="train",transform=False,class_balancing=False):
        """
        Args:
            directory (string): Path to the dataset.
            mode (str): train = 90% Train, validation=10% Train, train+validation=100% train else test.
            transform (callable, optional): Optional transform to be applied
                on a sample.

        """
        self.directory = directory
        self.mode = mode
        self.transform = transform

        np.random.seed(0)

        if self.mode=="train":
            x = np.load(os.path.join(directory,"X_train_vol16_1_5.npy"))
            y = np.load(os.path.join(directory,"y_train_vol16_1_5.npy"))
            np.random.shuffle(x)
            np.random.shuffle(y)
            examples_threshold = np.round(x.shape[0]*0.9,0).astype(np.int32)
            x = x[:examples_threshold]
            y = y[:examples_threshold]
        elif self.mode=="validation":
            x = np.load(os.path.join(directory,"X_train_vol16_1_5.npy"))
            y = np.load(os.path.join(directory,"y_train_vol16_1_5.npy"))
            np.random.shuffle(x)
            np.random.shuffle(y)
            examples_threshold = np.round(x.shape[0]*0.9,0).astype(np.int32)
            x = x[examples_threshold:]
            y = y[examples_threshold:]
        elif mode=="train+validation":
            x = np.load(os.path.join(directory,"X_train_vol16_1_5.npy"))
            y = np.load(os.path.join(directory,"y_train_vol16_1_5.npy"))
        else:
            x = np.load(os.path.join(directory,"X_test_vol16_1_5.npy"))
            y = np.load(os.path.join(directory,"y_test_vol16_1_5.npy"))
        self.X = torch.FloatTensor(np.expand_dims(x,1).astype(np.float32))
        self.Y = torch.LongTensor(y.astype(np.int64))

        if class_balancing and mode in ("train","train+validation"):            
            self.X, self.Y = class_balancing.class_balancing(self.X,self.Y)             
        print(self.mode,self.X.shape,np.bincount(self.Y))
            
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        sample = [self.X[idx], self.Y[idx]]
        if self.transform:
            sample[0] = self.transform(sample[0])
        return sample

def train_pines(class_balancing,layers,conv_type,
               dropout,epochs,pooling,data_augmentation,validation):

    use_cuda = torch.cuda.is_available()

    if data_augmentation:
        if isinstance(conv_type[1],dict) and "translations_path" in conv_type[1].keys():
            transform_train = graph.GraphCrop(conv_type[1]["translations_path"].replace(".pkl",""))
        else:
            transform_train = graph.GraphCrop()
    else:
        transform_train = False

    if validation:
        mode_train = "train"
        mode_test = "validation"
    else:
        mode_train = "train+validation"
        mode_test = "test"

    trainset = PINESDataset(mode=mode_train, transform=transform_train,class_balancing=class_balancing)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=8)

    testset = PINESDataset(mode=mode_test, transform=False,class_balancing=class_balancing)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)


    net = MiniGraphPreActResNet(conv_type=conv_type,
        dropout=dropout,layers=layers,pooling=pooling)

    if use_cuda:
        net = net.cuda(0)
        net = torch.nn.DataParallel(net, device_ids=[0])
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())

    def train(epoch):
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()


    def test():
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        # Save checkpoint.
        acc = 100.*correct/total
        return acc
    best_acc = 0
    last_acc = 0
    for epoch in range(epochs):
        train(epoch)
        acc = test()
        last_acc = acc
        if acc > best_acc:
            best_acc = acc
    return round(last_acc,2)
