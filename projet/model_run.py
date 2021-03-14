class ModelRun():
    """
    This class holds a model's state dict, its hparams, and
    maybe it's optimizer's state in the future
    """
    def __init__(self, state_dict, hyperparameters):
        self.state_dict = state_dict
        self.hyperparameters = hyperparameters