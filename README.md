# MCNN

Matching Convolutional Neural Networks Without Priors About Data

## Paper

[short version under review at ICLR 2018 Workshop](https://openreview.net/forum?id=SJ9e4HJPM)

[long version (coming soon)](https://)

## Dependencies

We recommend the users to use a [anaconda](https://www.anaconda.com) environment to run the experiments. This environment can be created using:

~~~
conda env create -n environment_name -f environment.yml
~~~ 

## Graph Processing

Graph generation and translation searching available in folder graph_processing. A python version is available for generating the graphs and translations. If the python version takes too long to find the translations you can try the ocaml version of the program that is quicker, but does not generate the strides.

## CIFAR-10

### Architecture

We train all our models in [PreActResNet18](https://arxiv.org/abs/1603.05027). This model contais 3 strides, which in the case of the [Defferard method](https://arxiv.org/pdf/1606.09375.pdf) are converted to pooling.

The models are trained during 50 epochs with a learning rate of 0.1 and then trained for 50 more epochs with a learning rate of 0.001. An example of training our proposed method using our proposed data augmentation is:

~~~
cd proposed
python cifar.py -g -d &> log_covariance
python cifar.py -g -d --resume --lr 0.001 &>> log_covariance
~~~

Further details about the model can be seen in the source files proposed/graph.py and defferard/models/

### Usage for training the proposed method and classical cnns

~~~~
cd proposed
python cifar.py --help
usage: cifar.py [-h] [--lr LR] [--batch BATCH] [--epochs EPOCHS] [--resume]
                [--flip] [--no_da] [--graph_data_aug] [--graph_convolution]
                [--name NAME] [--translations_crop TRANSLATIONS_CROP]
                [--translations_conv TRANSLATIONS_CONV]

CIFAR10 CNN and Graph Translation Training

optional arguments:
  -h, --help            show this help message and exit
  --lr LR               learning rate
  --batch BATCH         batch size
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
~~~
cd defferard
python cifar.py --help
usage: cifar.py [-h] [--lr LR] [--batch BATCH] [--epochs EPOCHS] [--clip CLIP]
                [--k K] [--resume] [--flip]

CIFAR10 Defferard Training

optional arguments:
  -h, --help       show this help message and exit
  --lr LR          learning rate
  --batch BATCH    batch size
  --epochs EPOCHS  epochs to train
  --clip CLIP      gradient clipping value
  --k K            polynomial orders
  --resume, -r     resume from checkpoint
  --flip, -f       use flips for data augmentation
~~~

## PINES/IAPS

Architecture (CNN, Defferard and Proposed): Input Layer -> Dropout (0.1) -> Conv(1->64) -> Conv(64->128) -> Dropout(0.1) -> Linear(128*369 -> 2)

Experiments can be made by running the pines_test.py script in proposed and defferard folders. A script for generating the means is available at pines_summary.py.

## Credits

Carlos Eduardo Rosar Kos Lassance, Jean-Charles Vialatte, Vincent Gripon and Nicolas Farrugia

Baseline for cifar10 - PreActResNet18: https://github.com/kuangliu/pytorch-cifar

Base code for the Defferard method: https://github.com/xbresson/graph_convnets_pytorch 

