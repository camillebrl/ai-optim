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

# regularization(dataset=dataset,
#                model_name="ResNet18",
#                n_classes=n_classes,
#                train_loader=train_loader,
#                test_loader=test_loader,
#                n_epochs=130,
#                regul_coef=0.5,
#                regul_function="spectral_isometry",
#                learning_rate=0.0001,
#                gradient_method="Adam",
#                scheduler=None)
# pruning(model_fname="cifar10/models/models_regularized/Regul_spectral_isometry_0.5_ResNet18_CrossEntropyLoss()_0.0001_0.95_5e-05_SGD_None.run",
#         dataset=dataset,
#         n_classes=n_classes,
#         train_loader=train_loader,
#         test_loader=test_loader,
#         n_epochs=130,
#         pruning_rate=0.5,
#         pruning_type="thinet_normal")
# quantization(model_fname=model_filename,
#              dataset=dataset,
#              n_classes=n_classes,
#              train_loader=train_loader,
#              test_loader=test_loader,
#              n_epochs=125,
#              nb_bits=2)
distillation(model_fname="cifar10/models/models_quantized/Quantization_4_ResNet18_CrossEntropyLoss()_0.001_0.95_5e-05_SGD_CosineAnnealingLR_thinet_normal_0.2_Regul_simple_0.3.run",
             dataset=dataset,
             n_classes=n_classes,
             train_loader=train_loader,
             test_loader=test_loader,
             n_epochs=125)