from architecture_ResNet import nums_blocks

class Hyperparameters():
    def __init__(self,learning_rate,weight_decay,momentum,loss_function,gradient_method,regul_coef,regul_function,model_name):
        self.learning_rate=learning_rate
        self.weight_decay=weight_decay
        self.momentum=momentum
        self.loss_function=loss_function
        self.scheduler=scheduler
        self.optimizer=optimizer
        self.regul_coef=regul_coef
        self.regul_function=regul_function
    def build_name(self):
        fname = f"{self.model_name}_{self.loss_function}_regCoefOf_{self.regul_coef}_{self.regul_function}" \
                    f"_lr_{self.learning_rate}_momentumOf_{self.momentum}_weightDecayOf_{self.weight_decay}_gradMethodOf_" \
                    f"{self.gradient_method}_{self.scheduler}"
        return fname

class Pruning_Hyperparameters(Hyperparameters):
    def __init__(self,learning_rate,weight_decay,momentum,loss_function,gradient_method,pruning_rate,pruning_function):
        super().__init__(self,learning_rate,weight_decay,momentum,loss_function,gradient_method)
        self.pruning_rate=pruning_rate
        self.pruning_function=pruning_function
    def build_name(self,model_name):
        base_name=super().build_name(model_name)
        return f'{self.pruning_function}_{self.pruning_rate}_'+base_name


class Quantization_Hyperparameters(Hyperparameters):
    def __init__(self,learning_rate,weight_decay,momentum,loss_function,gradient_method,nb_bits):
        super().__init__(self,learning_rate,weight_decay,momentum,loss_function,gradient_method)
        self.nb_bits=nb_bits
    def build_name(self):
        base_name=super().build_name(self.model_name)
        return f'Quantization_{self.nb_bits}_'+base_name