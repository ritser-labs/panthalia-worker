import math
import asyncio


class StandardPlugin:
    def __init__(
        self,
        model_adapter,
        model_config,
        sot_adapter,
        dataset,
        num_steps,
        examples_per_accumulation,
        accumulations_per_step,
        tokenizer=None,
        outer_max_lr=0.7,
        outer_min_lr=0.7,
        outer_weight_decay=0.00,
        tensor_version_interval=60,
        max_concurrent_iterations=4,
        preload_batch_count=4,
        chunk_shape=(64, 64),
        k=1,
    ):
        self.model_adapter = model_adapter
        self.model_config = model_config
        self.sot_adapter = sot_adapter
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.num_steps = num_steps
        self.batch_size = num_steps * examples_per_accumulation
        self.examples_per_accumulation = examples_per_accumulation
        self.accumulations_per_step = accumulations_per_step
        self.outer_max_lr = outer_max_lr
        self.outer_min_lr = outer_min_lr
        self.outer_weight_decay = outer_weight_decay
        self.tensor_version_interval = tensor_version_interval
        self.max_concurrent_iterations = max_concurrent_iterations
        self.preload_batch_count = preload_batch_count
        self.chunk_shape = chunk_shape
        self.k = k
        
        self.model_adapter.plugin = self
    
    def get_master_learning_hyperparameters(self):
        return {
            'steps': self.num_steps,
            'accumulations_per_step': self.accumulations_per_step,
            'tensor_version_interval': self.tensor_version_interval,
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
            'chunk_shape': self.chunk_shape,
            'k': self.k,
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
        allowed_targets = ["model_adapter", "dataset", "sot_adapter"]
        if target not in allowed_targets:
            raise ValueError(f"Target must be one of {allowed_targets}")

        obj = getattr(self, target)
        if not hasattr(obj, func_name):
            raise AttributeError(f"{target} has no function '{func_name}'")

        func = getattr(obj, func_name)

        # check if the function is coroutine
        if asyncio.iscoroutinefunction(func):
            result = await func(*args, **kwargs)
        else:
            result =  func(*args, **kwargs)
        
        if result is None:
            # No more data scenario. Instead of treating it as an error,
            # we return a structured response or propagate a known status.
            return {"status": "no_more_data"}
        return result

    def get(self, key):
        """
        Get a value from the plugin object.

        Args:
            key (str): The key to retrieve.

        Returns:
            The value associated with the key.
        """
        return getattr(self, key)