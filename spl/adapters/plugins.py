import math
import asyncio


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
        inner_weight_decay=0.0,
        preload_batch_count=4
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
        self.inner_weight_decay = inner_weight_decay
        self.preload_batch_count = preload_batch_count
    
    def get_master_learning_hyperparameters(self):
        return {
            'steps': self.num_steps,
            'max_lr': self.inner_max_lr,
            'min_lr': self.inner_min_lr,
            'T_0': self.inner_T_0,
            'weight_decay': self.inner_weight_decay,
            'tensor_version_interval': self.tensor_version_interval,
            'expected_worker_time': self.expected_worker_time,
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
        
    async def call_submodule(self, target, func_name, *args, **kwargs):
        """
        Call an arbitrary function on model_adapter or dataset, handling async functions.

        Args:
            target (str): Either "model_adapter" or "dataset".
            func_name (str): The name of the function to call.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.
        
        Returns:
            The result of the function call.
        """
        if target not in ["model_adapter", "dataset"]:
            raise ValueError("Target must be 'model_adapter' or 'dataset'")

        obj = getattr(self, target)
        if not hasattr(obj, func_name):
            raise AttributeError(f"{target} has no function '{func_name}'")

        func = getattr(obj, func_name)

        # check if the function is coroutine
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    def get(self, key):
        """
        Get a value from the plugin object.

        Args:
            key (str): The key to retrieve.

        Returns:
            The value associated with the key.
        """
        return getattr(self, key)