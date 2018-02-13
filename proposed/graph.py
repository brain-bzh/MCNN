'''Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import ast
import numpy as np

from torch.autograd import Variable
from torch.nn.parameter import Parameter
import math
import pickle

TRANSLATIONS_PATH_CROP="translations/symmetric-translations"
TRANSLATIONS_PATH="translations/symmetric-translations.pkl"

class GraphCrop(object):
    """WIP
    """

    def __init__(self,translations_path=TRANSLATIONS_PATH_CROP,times_to_translate=4):
        self.translations_path = translations_path
        self.translations = ast.literal_eval(open(translations_path, "r").read())
        self.inverted_translations = list()
        for translation in self.translations:
            dict_ = dict()
            for k,v in translation.items():
                if not isinstance(v,list):
                    v = [v]
                for value in v:
                    dict_[value] = k
            self.inverted_translations.append(dict_)
        self.times_to_translate = times_to_translate

    def shift_direction(self,image,direction):
        image = image.view(3,1024)
        zeros = torch.unsqueeze(torch.zeros_like(image)[:,0],1)
        directions = (torch.zeros(image.size()[1])-1).type(torch.LongTensor)
        for index_node,inverted_dict in enumerate(self.inverted_translations):
            if direction in inverted_dict.keys():
                directions[index_node] = inverted_dict[direction]
            else:
                directions[index_node] = image.size()[1]
        image = torch.cat([image,zeros],dim=1)
        image = image[:,directions]
        image = image.view(3,32,32)
        return image



    def __call__(self, img):
        """
        Args:
            img (torch tensor): torch graph tensor to be cropped.
        Returns:
            torch tensor: cropped torch graph tensor
        """
        times = np.random.randint(0,self.times_to_translate+1)
        for _ in range(times):
            direction = np.random.randint(2,6)
            img = self.shift_direction(img,direction)
        return img

    def __repr__(self):
        return self.translations_path

    def __str__(self):
        return self.translations_path


class GraphPreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, number_of_strides,stride=False,translations_path=TRANSLATIONS_PATH):
        super(GraphPreActBlock, self).__init__()
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride
        self.translations = pickle.load(open(translations_path,"rb"))
        if self.stride:
            self.alive_indexes = torch.cuda.LongTensor(np.array(self.translations[number_of_strides]["alive"]))
        self.bn1 = nn.BatchNorm1d(in_planes)
        self.conv1 = TernaryLayer(in_channels=in_planes, number_of_strides=number_of_strides, out_channels=planes,translations_path=translations_path)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv2 = TernaryLayer(in_channels=planes, number_of_strides=number_of_strides,out_channels=planes,translations_path=translations_path)

        if in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, bias=False)
            )

    def forward(self, x):
        if self.stride:
            x = x[:,:,self.alive_indexes]

        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out.view(-1,self.in_planes,out.size()[2],1)).view(-1,self.planes,out.size()[2]) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class GraphPreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, translations_path=TRANSLATIONS_PATH):
        super(GraphPreActResNet, self).__init__()
        self.in_planes = 64
        self.number_of_strides = 0
        self.num_blocks = num_blocks
        self.translations_path = translations_path

        self.conv1 = TernaryLayer(in_channels=3, out_channels=64,translations_path=self.translations_path)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=False)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=True)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=True)
        self.last_layer = 256
        if self.num_blocks[3] > 0:
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=True)
            self.last_layer = 512
        self.linear = nn.Linear(self.last_layer, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [False]*(num_blocks-1)
        layers = []
        for stride in strides:
            if stride:
                self.number_of_strides += 1
            layers.append(block(self.in_planes, planes, self.number_of_strides,stride,translations_path=self.translations_path))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1,3,32**2)
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        if self.num_blocks[3] > 0:
            out = self.layer4(out)
        out = torch.mean(out,2)
        out = self.linear(out)
        return out

def GraphPreActResNet18(translations_path=TRANSLATIONS_PATH):
    return GraphPreActResNet(GraphPreActBlock, [2,2,2,2],translations_path=translations_path)

def GraphPreActResNet16(translations_path=TRANSLATIONS_PATH):
    return GraphPreActResNet(GraphPreActBlock, [2,2,2,0],translations_path=translations_path)

class TernaryLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, number_of_strides=0,translations_path=TRANSLATIONS_PATH):
        super(TernaryLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.translations_path = translations_path
        self.translations = pickle.load(open(self.translations_path,"rb"))[number_of_strides]
        self.build()

        weight = np.ones((self.in_channels*self.kernel_size, self.out_channels))

        self.weight = Parameter(torch.Tensor(weight))
        self.reset_parameters()

    def build(self):
        lines = self.translations["translations"]
        self.in_nodes = len(self.translations["alive"])
        assert self.in_nodes == len(lines)
        
        self.kernel_size = max([value for value in [max(list(cnnFilter.values())) for cnnFilter in lines]])
        if isinstance(self.kernel_size, list):
            self.kernel_size = self.kernel_size[0]
        
        indices = np.zeros((self.in_nodes, self.kernel_size))
        counts = np.zeros((self.in_nodes,))
        
        i = 0
        for translation in lines:
            for vertex,directions in translation.items():
                if not isinstance(directions,list):
                    directions = [directions]
                for direction in directions:
                    indices[i, (direction - 1) % self.kernel_size] = vertex + 1 
                    counts[i] += 1
            i += 1

        self.indices = torch.cuda.LongTensor(indices)
        for node in range(self.in_nodes):
            if not(counts[node]):
                raise RuntimeError('Node ' + str() + ' has no input connection.')

    def reset_parameters(self):
        n = self.in_channels
        n *= self.kernel_size
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        # checking shape
        in_shape = x.shape
        if len(in_shape) == 4:
            b, p, h, w = x.shape
            n = h*w
            if p != self.in_channels:
                raise RuntimeError('Second dimension of x is not equal to in_channels')
            if n != self.in_nodes:
                raise RuntimeError('Third*Fourth dimensions of x is not equal to in_nodes')
            x = x.view(b, p, n)
        elif len(in_shape) == 3:
            b, p, n = x.shape
            if p != self.in_channels:
                raise RuntimeError('Second dimension of x is not equal to in_channels')
            if n != self.in_nodes:
                raise RuntimeError('Third dimension of x is not equal to in_nodes')
        else:
            raise RuntimeError('In shape length is neither 3 of 4')

        # gathering col
        padded = F.pad(x, (1,0,0,0,0,0), 'constant', 0)
        col = padded[:,:,self.indices]

        # tensordot x<b,p,n,k> * w<p,k,q>
        col.transpose_(1,2).contiguous()
        k = self.kernel_size
        col = col.view(b*n,p*k)
        q = self.out_channels
        end = col.matmul(self.weight)
        end = end.view(b,n,q)
        end.transpose_(1,2).contiguous()

        # reshaping
        if len(in_shape) == 4:
            end = end.view(b,q,h,w)

        return end

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_channels=' + str(self.in_channels) \
            + ', out_channels=' + str(self.out_channels) \
            + ', translations_files=' + str(self.translations_path) \
            

