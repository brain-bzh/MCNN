'''ChebNet Pre-activation ResNet in PyTorch.

References:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
[2] Defferrard, Micha\"{e}l and Bresson, Xavier and Vandergheynst, Pierre
    Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering.
    Advances in Neural Information Processing Systems 29
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.autograd import Function

from .lib import *
import numpy as np

class my_sparse_mm(Function):
    """
    Implementation of a new autograd function for sparse variables, 
    called "my_sparse_mm", by subclassing torch.autograd.Function 
    and implementing the forward and backward passes.
    """
    
    def forward(self, W, x):  # W is SPARSE
        self.save_for_backward(W, x)
        y = torch.mm(W, x)
        return y
    
    def backward(self, grad_output):
        W, x = self.saved_tensors 
        grad_input = grad_output.clone()
        grad_input_dL_dW = torch.mm(grad_input, x.t()) 
        grad_input_dL_dx = torch.mm(W.t(), grad_input )
        return grad_input_dL_dW, grad_input_dL_dx

class ChebConv(nn.Module):

    def __init__(self, Fin, Fout, L, K, bias=False):
        super(ChebConv, self).__init__()
        self.L = L
        self.Fin = Fin
        self.Fout = Fout
        self.K = K
        self.bias = bias
        
        self.cl = nn.Linear(K*Fin, Fout, bias)
        scale = np.sqrt( 2.0/ (Fin+Fout) )
        self.cl.weight.data.uniform_(-scale, scale)
        if self.bias:
            self.cl.bias.data.fill_(0.0)

    def forward(self, x):
        #(x, cl, L, lmax, Fout, K):

        # parameters
        # B = batch size
        # V = nb vertices
        # Fin = nb input features
        # Fout = nb output features
        # K = Chebyshev order & support size

        x = x.permute(0,2,1).contiguous()
        B, V, Fin = x.size(); B, V, Fin = int(B), int(V), int(Fin) 

        # rescale Laplacian
        lmax = lmax_L(self.L)
        L = rescale_L(self.L, lmax) 
        
        # convert scipy sparse matric L to pytorch
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col)).T 
        indices = indices.astype(np.int64)
        indices = torch.from_numpy(indices)
        indices = indices.type(torch.LongTensor)
        L_data = L.data.astype(np.float32)
        L_data = torch.from_numpy(L_data) 
        L_data = L_data.type(torch.FloatTensor)
        L = torch.sparse.FloatTensor(indices, L_data, torch.Size(L.shape))
        L = Variable( L , requires_grad=False)
        if torch.cuda.is_available():
            L = L.cuda()
        
        # transform to Chebyshev basis
        x0 = x.permute(1,2,0).contiguous()  # V x Fin x B
        x0 = x0.view([V, Fin*B])            # V x Fin*B
        x = x0.unsqueeze(0)                 # 1 x V x Fin*B
        
        def concat(x, x_):
            x_ = x_.unsqueeze(0)            # 1 x V x Fin*B
            return torch.cat((x, x_), 0)    # K x V x Fin*B  
             
        if self.K > 1: 
            x1 = my_sparse_mm()(L,x0)              # V x Fin*B
            x = torch.cat((x, x1.unsqueeze(0)),0)  # 2 x V x Fin*B
        for k in range(2, self.K):
            x2 = 2 * my_sparse_mm()(L,x1) - x0  
            x = torch.cat((x, x2.unsqueeze(0)),0)  # M x Fin*B
            x0, x1 = x1, x2  
        
        x = x.view([self.K, V, Fin, B])           # K x V x Fin x B     
        x = x.permute(3,1,2,0).contiguous()       # B x V x Fin x K
        x = x.view([B*V, Fin*self.K])             # B*V x Fin*K
        
        # Compose linearly Fin features to get Fout features
        x = self.cl(x)                            # B*V x Fout  
        x = x.view([B, V, self.Fout])             # B x V x Fout
        x = x.permute(0, 2, 1).contiguous()
        
        return x

class ChebPreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, L1, K1, L2=None, K2=None, stride=1):
        super(ChebPreActBlock, self).__init__()
        if stride==1:
            assert (L2 is None and K2 is None) or (L2 is L1 and K2==K1)
            L2, K2 = L1, K1
        self.bn1 = nn.BatchNorm1d(in_planes)
        self.conv1 = ChebConv(in_planes, planes, L1, K1, bias=False) #nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.pool1 = nn.MaxPool1d(stride) #assuming vertices are correctly orderred
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv2 = ChebConv(planes, planes, L2, K2, bias=False) #nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion*planes, kernel_size=1, stride=1, bias=False))
            self.pool2 = nn.MaxPool1d(stride)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.pool2(self.shortcut(out)) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.pool1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out

class ChebPreActResNet(nn.Module):
    def __init__(self, block, num_blocks, L_list, K_list, num_classes=10):
        super(ChebPreActResNet, self).__init__()
        self.in_planes = 64
        self.L_list = L_list
        self.K_list = K_list

        self.conv1 = ChebConv(3, 64, L_list[0], K_list[0], bias=False) #nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, L1=L_list[0], K1=K_list[0])
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, L1=L_list[0], K1=K_list[0], L2=L_list[1], K2=K_list[1])
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, L1=L_list[1], K1=K_list[1], L2=L_list[2], K2=K_list[2])
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, L1=L_list[2], K1=K_list[2], L2=L_list[3], K2=K_list[3])
        #self.linear = nn.Linear(512*block.expansion, num_classes)
        self.linear = nn.Linear(512 * block.expansion * 172, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, L1, K1, L2=None, K2=None):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, L1, K1, L2, K2, stride))
            self.in_planes = planes * block.expansion
            if stride > 1:
                L1, K1 = L2, K2
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #out = F.avg_pool1d(out, 172) #F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ChebPreActResNet18(K=25):
    L_list, perm = build_grid_graph(4)
    K_list = [K]*len(L_list)
    return perm, ChebPreActResNet(ChebPreActBlock, [2,2,2,2], L_list, K_list)

def ChebPreActResNet34(K=25):
    L_list, perm = build_grid_graph(4)
    K_list = [K]*len(L_list)
    return perm, ChebPreActResNet(ChebPreActBlock, [3,4,6,3], L_list, K_list)

def build_grid_graph(coarsening_levels=4):
    grid_side = 32
    number_edges = 4
    metric = 'euclidean'
    A = grid_graph(grid_side,number_edges,metric) # create graph of Euclidean grid

    # Compute coarsened graphs
    L_list, perm = coarsen(A, coarsening_levels)

    # Compute max eigenvalue of graph Laplacians
    lmax = []
    for i in range(coarsening_levels):
        lmax.append(lmax_L(L_list[i]))
    print('lmax: ' + str([lmax[i] for i in range(coarsening_levels)]))

    return L_list, perm

def reorder(perm, train_data, test_data, val_data=None):
    if perm is None:
        return train_data, test_data, val_data

    # Reindex nodes to satisfy a binary tree structure
    train_data = perm_data(train_data, perm)
    test_data = perm_data(test_data, perm)
    if not(val_data is None):
        val_data = perm_data(val_data, perm)
    del perm
    return train_data, test_data, val_data

def reorder(perm, data,):
    # Reindex nodes to satisfy a binary tree structure
    data = perm_data(data, perm)
    return data


def test():
    net = ChebPreActResNet18()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())

# test()
