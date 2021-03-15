import torch

# la convention est de mettre les constantes en majuscule

DEVICE= 'cuda' if torch.cuda.is_available() else 'cpu'