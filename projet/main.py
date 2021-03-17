from import_dataset import train_cifar10, train_cifar100, test_cifar10, test_cifar100
from utils import init_logging, init_parser
from pipeline import regularization, quantization, pruning

init_logging()
parser = init_parser()
args = parser.parse_args()
dataset = args.dataset
if dataset == "cifar10":
    n_classes = 10
    train_loader = train_cifar10
    test_loader = test_cifar10
elif dataset == "cifar100":
    n_classes = 100
    train_loader = train_cifar100
    test_loader = test_cifar100

#regularization(dataset, n_classes, train_loader, test_loader,200,0.3,"simple")
pruning(dataset, n_classes, train_loader, test_loader, 200,0.2,"thinet_normal")
quantization(dataset, n_classes, train_loader, test_loader, 200, 4)
