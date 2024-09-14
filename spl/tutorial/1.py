# Defining our tokenizer

class CharacterLevelTokenizer:
    def __init__(self, pad_id: int = 0):
        self.pad_id = pad_id
        
    def encode(self, text: str) -> list:
        return [ord(char) for char in text]
    
    def decode(self, ascii_list: list) -> str:
        return ''.join([chr(value) for value in ascii_list])

    def get_vocab_size(self) -> int:
        return 128


tokenizer = CharacterLevelTokenizer()
