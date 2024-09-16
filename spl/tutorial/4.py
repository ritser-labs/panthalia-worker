from .adapters.model_adapter import TransformerModelAdapter

# Previous code

# Defining the Panthalia model adapter
class NanoGPTModelAdapter(TransformerModelAdapter):
    def forward_and_loss(self, model, inputs, targets):
        return model.forward(inputs, targets)[1]
    
    def forward(self, model, inputs):
        return model.forward(inputs)[0]

    def generate(self, model, input, max_new_tokens=None, temperature=0.8, top_k=200):
        if max_new_tokens is None:
            max_new_tokens = self.model_config.get_max_seq_len()
        return model.generate(input, max_new_tokens, temperature=temperature, top_k=top_k)