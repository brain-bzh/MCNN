# MCNN

Matching Convolutional Neural Networks Without Priors About Data

## Paper

[short version under review at ICLR 2018 Workshop](https://openreview.net/forum?id=SJ9e4HJPM)

[long version (coming soon)](https://)

## Graph Processing

Graph generation and translation searching available in folder graph_processing. Instructions coming soon.

## CIFAR-10

All code is available at the pytorch folder.

### Architecture

We train all our models in [PreActResNet18](https://arxiv.org/abs/1603.05027). This model contais 3 strides, which in the case of the [Defferard method](https://arxiv.org/pdf/1606.09375.pdf) are converted to pooling.

The models are trained during 50 epochs with a learning rate of 0.1 and then trained for 50 more epochs with a learning rate of 0.001. An example of training our proposed method using our proposed data augmentation is:

~~~
python cifar.py -g -d &> log_covariance
python cifar.py -g -d --resume --lr 0.001 &>> log_covariance
~~~

 Further details can be seen in the source file under pytorch/models/preact_resnet.py

### Usage for training the proposed method and classical cnns
~~~~
usage: pytorch/cifar.py [-h] [--lr LR] [--epochs EPOCHS] [--resume] [--flip] [--no_da]
                [--graph_data_aug] [--graph_convolution] [--name NAME]
                [--translations_crop TRANSLATIONS_CROP]
                [--translations_conv TRANSLATIONS_CONV]

PyTorch CIFAR10 Training

optional arguments:
  -h, --help            show this help message and exit
  --lr LR               learning rate
  --epochs EPOCHS       epochs to train
  --resume, -r          resume from checkpoint
  --flip, -f            use flips for conventional data augmentation
  --no_da               don't use data augmentation
  --graph_data_aug, -d  use graph data augmentation
  --graph_convolution, -g
                        use graph convolutions instead of classical
                        convolutions
  --name NAME, -n NAME  name_of_the_checkpoint
  --translations_crop TRANSLATIONS_CROP
                        path to the translations for the crop
  --translations_conv TRANSLATIONS_CONV
                        path to the translations for the convolutions
~~~~

### Usage for training the Defferard method



## PINES/IAPS

Code needs cleaning and will be available soon.

Architecture (CNN, Defferard and Proposed): Input Layer -> Dropout (0.1) -> Conv(1->64) -> Conv(64->128) -> Dropout(0.1) -> Linear(128*369 -> 2)

## Visualizations

Coming soon

## Credits

Carlos Eduardo Rosar Kos Lassance, Jean-Charles Vialatte, Vincent Gripon and Nicolas Farrugia

Baseline for cifar10 - PreActResNet18: https://github.com/kuangliu/pytorch-cifar

Base code for the Defferard method: https://github.com/xbresson/graph_convnets_pytorch 

