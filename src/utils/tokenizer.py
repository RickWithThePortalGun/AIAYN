import re
from typing import List
import torch
from collections import Counter


class Tokenizer:
    """Simple tokenizer for machine translation."""
    
    def __init__(self, vocab_size=50000):
        self.vocab_size = vocab_size
        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = Counter()
        
        # Special tokens
        self.PAD = '<PAD>'
        self.UNK = '<UNK>'
        self.SOS = '<SOS>'
        self.EOS = '<EOS>'
        
    def build_vocab(self, texts: List[str]):
        """Build vocabulary from texts."""
        self.word_counts = Counter()
        
        for text in texts:
            words = self._tokenize(text)
            self.word_counts.update(words)
        
        # Get most common words
        most_common = self.word_counts.most_common(self.vocab_size - 4)
        
        # Add special tokens
        self.word2idx = {
            self.PAD: 0,
            self.UNK: 1,
            self.SOS: 2,
            self.EOS: 3
        }
        
        # Add vocabulary
        for word, _ in most_common:
            self.word2idx[word] = len(self.word2idx)
        
        # Create reverse mapping
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Simple word tokenization
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        return words
    
    def encode(self, text: str, add_sos_eos: bool = True) -> List[int]:
        """Encode text to indices."""
        words = self._tokenize(text)
        indices = []
        
        if add_sos_eos:
            indices.append(self.word2idx[self.SOS])
        
        for word in words:
            idx = self.word2idx.get(word, self.word2idx[self.UNK])
            indices.append(idx)
        
        if add_sos_eos:
            indices.append(self.word2idx[self.EOS])
        
        return indices
    
    def decode(self, indices: List[int]) -> str:
        """Decode indices to text."""
        words = []
        for idx in indices:
            word = self.idx2word.get(idx, self.UNK)
            if word in [self.PAD, self.SOS, self.EOS]:
                continue
            words.append(word)
        return ' '.join(words)
    
    def __len__(self):
        return len(self.word2idx)
