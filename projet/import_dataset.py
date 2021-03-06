
import torchvision.transforms as transforms

from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import CIFAR100, CIFAR10

normalize_scratch = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

# Data augmentation is needed in order to train from scratch
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(), # prend l'image de manière aléatoire et la tourne en mode miroir
    transforms.ToTensor(), # objigatoire pour pytorch: ne comprend que les tensors
    normalize_scratch, # centre-réduit chaque tensor de l'image
    ])
# A noter: ici, on ne fait que changer la même image, on ne met pas différentes versions de l'image, ce n'est pas vraiment du data augmentation
# Il aurait fallu prendre le dataset, le multiplier, et appliquer cette transformation (flip, crop) sur la moitié du dataset par exemple

transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize_scratch,
    ]) # On ne fait pas le flip et le crop pour le test


### The data from CIFAR100 will be downloaded in the following dataset
rootdir_cifar100 = './data/cifar100'

c100train = CIFAR100(rootdir_cifar100,train=True,download=True,transform=transform_train)
c100test = CIFAR100(rootdir_cifar100,train=False,download=True,transform=transform_test)

train_cifar100=DataLoader(c100train,batch_size=32)
test_cifar100=DataLoader(c100test,batch_size=32)

### The data from CIFAR10 will be downloaded in the following dataset
rootdir_cifar10 = './data/cifar10'

c10train = CIFAR10(rootdir_cifar10,train=True,download=True,transform=transform_train)
c10test = CIFAR10(rootdir_cifar10,train=False,download=True,transform=transform_test)

train_cifar10=DataLoader(c10train,batch_size=32)
test_cifar10=DataLoader(c10test,batch_size=32)