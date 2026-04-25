class CharTokenizer:
    def __init__(self, text: str):
        """
        Build a character-level tokenizer from the input text.
        
        Args:
            text: A string used to build the vocabulary.
        """
        self.stoi = {'<BOS>': 0, '<EOS>': 1}
        self.itos = {0: '<BOS>', 1: '<EOS>'}
        self.vocab_size = 2

        for c in sorted(set(text)):
            self.stoi[c] = self.vocab_size
            self.itos[self.vocab_size] = c
            self.vocab_size += 1

    def encode(self, text: str) -> list:
        """
        Encode a string into a list of token indices.
        
        Args:
            text: The string to encode.
        Returns:
            List of integer indices.
        """            
        return [0] + [self.stoi[c] for c in text] + [1] 

    def decode(self, indices: list) -> str:
        """
        Decode a list of token indices back into a string.
        
        Args:
            indices: List of integer indices.
        Returns:
            Decoded string.
        """
        return "<BOS>" + "".join(self.itos[i] for i in indices[1:-1]) + "<EOS>"