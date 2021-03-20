import logging
import os

from import_dataset import train_cifar10, train_cifar100, test_cifar10, test_cifar100
from utils import init_logging, init_parser
from pipeline import regularization, quantization, pruning, distillation

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
model_filename = args.model
if model_filename is not None:
    if os.path.exists(model_filename):
        logging.info(f"Loading model in {model_filename}")
    else:
        raise FileNotFoundError(f"The received file argument {model_filename} doesn't exist")

# regularization(dataset, "ResNet18", n_classes, train_loader, test_loader,200,0.3,"spectral_isometry")
pruning(model_filename, dataset, n_classes, train_loader, test_loader, 200, 0.5, "thinet_normal")
quantization(model_filename, dataset, n_classes, train_loader, test_loader, 125, 4)
distillation(model_filename, dataset, "models_quantized", n_classes, train_loader, test_loader, 125)
