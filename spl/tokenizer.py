import tiktoken

class Tokenizer:
    """
    Tokenizing and encoding/decoding text using the Tiktoken tokenizer.
    """

    def __init__(self, encoding_name: str):
        """
        Initializes the Tokenizer with a Tiktoken encoding.

        Args:
            encoding_name (str): The name of the Tiktoken encoding.
        """
        self.tokenizer = tiktoken.get_encoding(encoding_name)
        self.pad_id = self.tokenizer.encode("<pad>")[0]  # Assign a unique ID for pad
        self.bos_id = self.tokenizer.encode("<bos>")[0]  # Assign a unique ID for BOS
        self.eos_id = self.tokenizer.encode("<eos>")[0]  # Assign a unique ID for EOS

    def get_vocab_size(self):
        return self.tokenizer.n_vocab

    def encode(self, s: str, bos: bool = False, eos: bool = False):
        tokens = self.tokenizer.encode(s)
        if bos:
            tokens.insert(0, self.bos_id)
        if eos:
            tokens.append(self.eos_id)
        return tokens

    def decode(self, t: list):
        return self.tokenizer.decode(t)
