import numpy as np
import argparse
import torch, torchvision
import torchvision.transforms as transforms

#np.cov not working as expected
def cov(matrix):
    f = np.ones(matrix.shape[1])
    a = np.ones(matrix.shape[1])

    m = matrix.copy()
    w = f * a
    v1 = np.sum(w)
    v2 = np.sum(a)
    m -= np.sum(m * w, axis=1, keepdims=True) / v1
    _cov = np.dot(m * w, m.T) * v1 / (v1**2)    
    return _cov
    
def prepare_covariance_matrix(matrix):
    covariance = cov(matrix)
    np.fill_diagonal(covariance, 0)
    return covariance

def create_adjacence_matrix_from_covariance(matrix,k,symmetric):
    adjacence_matrix,weighted_adjacence_matrix = knn_over_matrix(matrix,k)
    if symmetric:
        return force_symmetry(adjacence_matrix), force_symmetry(weighted_adjacence_matrix)
    else:
        return adjacence_matrix, weighted_adjacence_matrix
        
    
def knn_over_matrix(matrix,k):
    temp = np.argsort(-matrix,axis=1)[:,k-1] # Get K biggest index
    thresholds = matrix[np.arange(matrix.shape[0]),temp].reshape(-1,1) # Transform matrix into a column matrix of maximums
    adjacence_matrix = (matrix >= thresholds)*1.0 # Create adjacence_matrix
    weighted_adjacence_matrix = adjacence_matrix * matrix # Create weigthed adjacence_matrix
    return adjacence_matrix, weighted_adjacence_matrix

def force_symmetry(matrix):
    return np.minimum(matrix+matrix.T,1)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def preprocessing(x_train,k,symmetric):
    covariance = prepare_covariance_matrix(x_train)
    adjacence_matrix, weighted_adjacence_matrix = create_adjacence_matrix_from_covariance(covariance,k=k,symmetric=symmetric)
    return covariance,convolution, adjacence_matrix, weighted_adjacence_matrix

def prepare_cifar_dataset():
    
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=False, num_workers=1)

    x_train = []
    for inputs, _ in trainloader:
        x_train.append(inputs.numpy())
    x_train = np.concatenate(x_train)
    x_train = np.transpose(x_train,[0,2,3,1])
    x_train = x_train.astype('float32')
    x_train = rgb2gray(x_train)
    x_train = x_train.reshape(-1,x_train.shape[1]*x_train.shape[2])
    return x_train


def save_adjacence_matrix(adjacence_matrix, filename):
    f = open(filename, 'w')
    size_string = adjacence_matrix.shape[0].__str__()
    f.write(size_string + "\n")
    for i in range(adjacence_matrix.shape[0]):
        line = adjacence_matrix[i,:]
        values = np.where(line > 0)[0]
        values_str = list(values).__str__()[1:-1].strip().replace(",","")
        f.write(values_str+"\n")
    f.close()
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate graph via covariance matrix')
    parser.add_argument('--verbose', '-v', action='store_true', help='show graph info in plots')
    parser.add_argument('--symmetric', '-s', action='store_true', help='force symmetry on graphs.')
    parser.add_argument('--k', '-k', default=4, type=float, help='k nearest neighbours for graph')
    parser.add_argument('--iaps', action='store_true', help='create iaps instead of cifar.')

    args = parser.parse_args()
    args.k = int(args.k)
    if args.iaps:
        filename = "iaps-radius-small"
    else:
        filename = "{}-{}nn-{}".format("cifar",
                                      args.k,
                                      "symmetric" if args.symmetric else "nonsymmetric")

    if not args.iaps:

        x_train = prepare_cifar_dataset()
        covariance, adjacence_matrix, weighted_adjacence_matrix = preprocessing(x_train.T,args.k,args.symmetric)
        print(adjacence_matrix)
        if args.verbose:
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            import seaborn
            plt.imshow(x_train[0].reshape(32,32), cmap = plt.get_cmap('Greys'))
            plt.figure(figsize=(10,10))
            seaborn.heatmap(covariance)
            plt.figure(figsize=(10,10))
            plt.imshow(adjacence_matrix,cmap="Greys")
            plt.show()

    else:
        coords = np.load("../data/IAPS/coords_sphere.npy")
        N = coords.shape[0]
        max_distance = 1.001001001
        adjacence_matrix = np.zeros((N,N))
        for index1,value in enumerate(coords):
            for index2,value2 in enumerate(coords):
                if index1 == index2:
                    continue
                distance = np.abs(value-value2).sum()
                if distance < max_distance:
                    adjacence_matrix[index1,index2] = 1
                    adjacence_matrix[index2,index1] = 1


    save_adjacence_matrix(adjacence_matrix,"graphs/{}".format(filename))
