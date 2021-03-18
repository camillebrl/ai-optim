from datetime import datetime


class Hyperparameters():
    def __init__(self, learning_rate, weight_decay, momentum, loss_function, gradient_method,
                 model_name, scheduler):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.loss_function = loss_function
        self.gradient_method = gradient_method
        self.model_name = model_name
        self.scheduler = scheduler

    def build_name(self):
        fname = f"{self.model_name}_{self.loss_function}_{self.learning_rate}_{self.momentum}_" \
                f"{self.weight_decay}_{self.gradient_method}_{self.scheduler}"
        return fname

    def get_tensorboard_name(self):
        now_as_str = datetime.now().strftime("%d_%m_%H:%M:%S_")
        return now_as_str + self.build_name()


class PruningHyperparameters(Hyperparameters):
    def __init__(self, learning_rate, weight_decay, momentum, loss_function, gradient_method,
                 model_name, scheduler, pruning_rate, pruning_type):
        super().__init__(learning_rate, weight_decay, momentum, loss_function, gradient_method,
                         model_name, scheduler)
        self.pruning_rate = pruning_rate
        self.pruning_type = pruning_type

    def build_name(self):
        base_name = super().build_name()
        return f'Pruning_{self.pruning_type}_{self.pruning_rate}_' + base_name


class QuantizationHyperparameters(Hyperparameters):
    def __init__(self, learning_rate, weight_decay, momentum, loss_function, gradient_method,
                 model_name, scheduler, nb_bits):
        super().__init__(learning_rate, weight_decay, momentum, loss_function, gradient_method,
                         model_name, scheduler)
        self.nb_bits = nb_bits

    def build_name(self):
        base_name = super().build_name()
        return f'Quantization_{self.nb_bits}_' + base_name


class RegularizationHyperparameters(Hyperparameters):
    def __init__(self, learning_rate, weight_decay, momentum, loss_function, gradient_method,
                 model_name, scheduler, regul_coef, regul_function):
        super().__init__(learning_rate, weight_decay, momentum, loss_function, gradient_method,
                         model_name, scheduler)
        self.regul_coef = regul_coef
        self.regul_function = regul_function

    def build_name(self):
        base_name = super().build_name()
        return f'Regul_{self.regul_function}_{self.regul_coef}_' + base_name

class ClusteringHyperparameters(Hyperparameters):
    def __init__(self, learning_rate, weight_decay, momentum, loss_function, gradient_method,
                 model_name, scheduler, nb_clusters):
        super().__init__(learning_rate, weight_decay, momentum, loss_function, gradient_method,
                         model_name, scheduler)
        self.nb_clusters = nb_clusters

    def build_name(self):
        base_name = super().build_name()
        return f'Clustering_{self.nb_clusters}_' + base_name