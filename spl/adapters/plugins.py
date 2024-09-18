import math


class StandardPlugin:
    def __init__(
        self,
        model_adapter,
        model_config,
        dataset,
        tokenizer,
        num_steps,
        examples_per_step,
        outer_max_lr=0.7,
        outer_min_lr=0.7,
        outer_weight_decay=0.00,
        tensor_version_interval=60,
        expected_worker_time=55,
        max_concurrent_iterations=4,
        inner_max_lr=0.001,
        inner_min_lr=0.0001,
        inner_T_0=200,
    ):
        self.model_adapter = model_adapter
        self.model_config = model_config
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.num_steps = num_steps
        self.batch_size = num_steps * examples_per_step
        self.outer_max_lr = outer_max_lr
        self.outer_min_lr = outer_min_lr
        self.outer_weight_decay = outer_weight_decay
        self.tensor_version_interval = tensor_version_interval
        self.expected_worker_time = expected_worker_time
        self.max_concurrent_iterations = max_concurrent_iterations
        self.inner_max_lr = inner_max_lr
        self.inner_min_lr = inner_min_lr
        self.inner_T_0 = inner_T_0
    
    def get_master_learning_hyperparameters(self, current_master_iteration):
        return {
            'steps': self.num_steps,
            'max_lr': self.inner_max_lr,
            'min_lr': self.inner_min_lr,
            'T_0': self.inner_T_0
        }
    
    def get_sot_learning_hyperparameters(self, current_iteration):
        """
        Calculate the learning rate using cosine annealing with warm restarts.

        Args:
            current_iteration (int): Current iteration number.

        Returns:
            dict: A dictionary containing the learning rate and Adam optimizer parameters.
        """
        T_0 = 20  # Initial number of iterations for the first cycle
        T_mult = 2  # Factor to increase the cycle length after each restart
        eta_max = self.outer_max_lr  # Initial learning rate (maximum)
        eta_min = self.outer_min_lr  # Minimum learning rate

        # Determine the current cycle length
        cycle_length = T_0
        t = current_iteration
        while current_iteration >= cycle_length:
            current_iteration -= cycle_length
            cycle_length *= T_mult

        # Calculate the learning rate using the cosine annealing formula
        lr = eta_min + (eta_max - eta_min) * (1 + math.cos(math.pi * current_iteration / cycle_length)) / 2
        
        print(f'Calculated LR: {lr}')

        return {
            'learning_rate': lr,
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-8,
            'weight_decay': self.outer_weight_decay,
            't': t,
        }
